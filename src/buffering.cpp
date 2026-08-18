// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The table-buffering half of the protocol: sink every input batch, then
// produce.

#include <stdexcept>

#include <arrow/array.h>
#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>

#include "dispatcher.h"
#include "methods.h"
#include "vgi/settings.h"
#include "vgi/storage.h"

#include "wire.h"

namespace vgi {
namespace {

vgi_rpc::Result envelope_of(const std::shared_ptr<arrow::RecordBatch>& payload) {
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(payload))
                                      .finish());
}

// The `state_ids` parameter is a list<binary>; read it back as strings.
std::vector<std::string> read_state_ids(const std::shared_ptr<arrow::RecordBatch>& dto,
                                        const std::string& field) {
    std::vector<std::string> ids;
    auto column = dto->GetColumnByName(field);
    auto list = std::dynamic_pointer_cast<arrow::ListArray>(column);
    if (!list || list->length() == 0 || list->IsNull(0)) return ids;
    auto values = std::dynamic_pointer_cast<arrow::BinaryArray>(list->values());
    if (!values) return ids;
    const int64_t begin = list->value_offset(0);
    const int64_t end = list->value_offset(1);
    for (int64_t i = begin; i < end; ++i) {
        if (!values->IsNull(i)) ids.push_back(values->GetString(i));
    }
    return ids;
}

}  // namespace

ProcessParams Dispatcher::buffering_params(const std::shared_ptr<arrow::RecordBatch>& dto) const {
    ProcessParams params;
    params.catalog_name = catalog_.name;
    params.schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    params.execution_id = wire::get_optional_binary(dto, "execution_id").value_or("");
    params.storage = default_storage();

    // Reload what bind stashed. These calls carry no arguments, settings or
    // COPY destination of their own, and may run on a worker that never bound.
    const auto stashed = [&](const char* key) {
        return params.storage->kv_get(params.execution_id, key).value_or("");
    };
    params.arguments = Arguments::parse(stashed("bind.arguments"));
    params.settings = Settings::parse(stashed("bind.settings"));
    params.secrets = Secrets::parse(stashed("bind.secrets"));
    if (auto format = stashed("bind.copy_to_format"); !format.empty()) {
        params.copy_to_format = format;
    }
    if (auto path = stashed("bind.copy_to_path"); !path.empty()) params.copy_to_path = path;
    params.output_schema = wire::decode_schema(stashed("bind.schema"));
    // A buffering call carries no arguments or output schema of its own; both
    // were settled at bind and belong to the execution, not the batch.
    return params;
}

vgi_rpc::Result Dispatcher::table_buffering_process(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("table_buffering_process: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    auto fn = require_buffering(function_name, schema_name);

    auto batch = wire::decode_ipc(wire::get_binary(dto, "input_batch"));
    auto params = buffering_params(dto);
    auto state_id = batch ? fn->process(params, batch) : std::string{};

    return envelope_of(wire::ResultBuilder(payload_schema_of("table_buffering_process"))
                           .set_binary("state_id", state_id)
                           .fill_defaults()
                           .finish());
}

vgi_rpc::Result Dispatcher::table_buffering_combine(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("table_buffering_combine: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    auto fn = require_buffering(function_name, schema_name);

    auto params = buffering_params(dto);
    auto finalize_ids = fn->combine(params, read_state_ids(dto, "state_ids"));

    return envelope_of(wire::ResultBuilder(payload_schema_of("table_buffering_combine"))
                           .set_binary_list("finalize_state_ids", finalize_ids)
                           .fill_defaults()
                           .finish());
}

vgi_rpc::Result Dispatcher::table_buffering_destructor(const vgi_rpc::Request& request) {
    // Release the execution's shared state. Best effort and never raising:
    // the protocol requires that, and a failure here would abort a query that
    // has already produced its answer.
    try {
        if (auto dto = wire::get_ipc(request.batch(), "request")) {
            const auto execution_id =
                wire::get_optional_binary(dto, "execution_id").value_or("");
            if (!execution_id.empty()) default_storage()->clear(execution_id);
        }
    } catch (const std::exception&) {
    }
    auto empty = arrow::RecordBatch::MakeEmpty(arrow::schema({})).ValueOrDie();
    return envelope_of(empty);
}

}  // namespace vgi
