// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The table-buffering half of the protocol: sink every input batch, then
// produce.

#include <functional>
#include <stdexcept>

#include <arrow/array.h>
#include <vgi_rpc/call_context.h>
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

}  // namespace

vgi_rpc::LogLevel to_rpc_level(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return vgi_rpc::LogLevel::DEBUG;
        case LogLevel::Warning: return vgi_rpc::LogLevel::WARN;
        case LogLevel::Error: return vgi_rpc::LogLevel::ERROR;
        case LogLevel::Info: break;
    }
    return vgi_rpc::LogLevel::INFO;
}

std::function<void(LogLevel, const std::string&)> client_log_sink(
    vgi_rpc::CallContext* context) {
    // A no-op rather than an empty function when there is no channel, so a
    // function may log without first asking which call it is in.
    if (!context) return [](LogLevel, const std::string&) {};
    return [context](LogLevel level, const std::string& message) {
        context->client_log(to_rpc_level(level), message);
    };
}

ProcessParams Dispatcher::buffering_params(const std::shared_ptr<arrow::RecordBatch>& dto,
                                           vgi_rpc::CallContext* context) const {
    ProcessParams params;
    params.client_log = client_log_sink(context);
    const auto attachment = attachment_of(dto);
    params.catalog_name = attachment.catalog;
    params.attachment_id = attachment.id;
    params.attach_options = attachment.options;
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

vgi_rpc::Result Dispatcher::table_buffering_process(const vgi_rpc::Request& request,
                                                    vgi_rpc::CallContext& context) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("table_buffering_process: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const Scope scope{attachment_of(dto).catalog, schema_name};
    auto fn = require_buffering(function_name, scope);

    auto batch = wire::decode_ipc(wire::get_binary(dto, "input_batch"));
    auto params = buffering_params(dto, &context);
    params.input_batch_index = wire::get_optional_int64(dto, "batch_index");
    auto state_id = batch ? fn->process(params, batch) : std::string{};

    return envelope_of(wire::ResultBuilder(payload_schema_of("table_buffering_process"))
                           .set_binary("state_id", state_id)
                           .fill_defaults()
                           .finish());
}

vgi_rpc::Result Dispatcher::table_buffering_combine(const vgi_rpc::Request& request,
                                                    vgi_rpc::CallContext& context) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("table_buffering_combine: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const Scope scope{attachment_of(dto).catalog, schema_name};
    auto fn = require_buffering(function_name, scope);

    auto params = buffering_params(dto, &context);
    // Through `wire::` rather than a local reader: the shared one accepts both
    // binary widths and raises on a shape it does not recognise, where a
    // hand-rolled one that quietly returned an empty list would have combine
    // finalize nothing and the sink produce no rows.
    auto finalize_ids = fn->combine(params, wire::get_binary_list(dto, "state_ids"));

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
