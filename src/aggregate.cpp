// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The aggregate half of the protocol.
//
// An aggregate is not a stream. The engine drives it through five unary calls
// and the worker holds the per-group state between them, keyed by an
// execution id the engine mints at bind and echoes on every later call.

#include <stdexcept>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>

#include "dispatcher.h"
#include "methods.h"
#include "wire.h"

namespace vgi {
namespace {

// Split an aggregate input batch into its group-id column and the value
// columns, which is every column except that one.
std::pair<std::shared_ptr<arrow::Int64Array>, std::vector<std::shared_ptr<arrow::Array>>>
split_group_ids(const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto group_ids =
        std::dynamic_pointer_cast<arrow::Int64Array>(batch->GetColumnByName(kGroupColumnName));
    if (!group_ids) {
        throw std::runtime_error(std::string("aggregate batch has no int64 '") +
                                 kGroupColumnName + "' column");
    }
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (int i = 0; i < batch->num_columns(); ++i) {
        if (batch->schema()->field(i)->name() == kGroupColumnName) continue;
        columns.push_back(batch->column(i));
    }
    return {group_ids, columns};
}

// An int64 column by name, falling back to position — the engine has used both
// spellings for the combine batch's two columns.
std::shared_ptr<arrow::Int64Array> int64_column(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& name, int position) {
    auto column = batch->GetColumnByName(name);
    if (!column && position < batch->num_columns()) column = batch->column(position);
    auto typed = std::dynamic_pointer_cast<arrow::Int64Array>(column);
    if (!typed) throw std::runtime_error("aggregate combine: no int64 '" + name + "' column");
    return typed;
}

vgi_rpc::Result empty_envelope() {
    // A void-ish aggregate call still answers with the envelope; the payload
    // is an empty batch rather than nothing, because the envelope's `result`
    // column is not nullable.
    auto empty = arrow::RecordBatch::MakeEmpty(arrow::schema({})).ValueOrDie();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(empty))
                                      .finish());
}

}  // namespace

vgi_rpc::Result Dispatcher::aggregate_bind(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_bind: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    auto fn = require_aggregate(function_name, schema_name);

    BindParams params;
    params.input_schema = wire::get_schema(dto, "input_schema");
    params.arguments = Arguments::parse(wire::get_optional_binary(dto, "arguments").value_or(""));
    params.settings = Settings::parse(wire::get_optional_binary(dto, "settings").value_or(""));
    params.secrets = Secrets::parse(wire::get_optional_binary(dto, "secrets").value_or(""));
    params.catalog_name = catalog_.name;
    params.schema_name = schema_name;

    auto output_schema = fn->bind(params);
    if (!output_schema) {
        throw std::runtime_error("aggregate '" + function_name + "' bound to no schema");
    }

    // The execution id keys this aggregation's state for its whole life. It
    // has to be unique across concurrent aggregations in one worker, which a
    // counter gives us — the engine treats it as opaque.
    auto execution_id = next_execution_id();
    aggregate_states_[execution_id];  // create the (empty) state map

    auto payload = wire::ResultBuilder(payload_schema_of("aggregate_bind"))
                       .set_binary("output_schema", wire::encode_schema(output_schema))
                       .set_binary("execution_id", execution_id)
                       .fill_defaults()
                       .finish();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(payload))
                                      .finish());
}

vgi_rpc::Result Dispatcher::aggregate_update(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_update: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const auto execution_id = wire::get_binary(dto, "execution_id");
    auto fn = require_aggregate(function_name, schema_name);

    auto batch = wire::decode_ipc(wire::get_binary(dto, "input_batch"));
    if (!batch) return empty_envelope();
    auto [group_ids, columns] = split_group_ids(batch);

    auto& stored = aggregate_states_[execution_id];

    // Pre-load only groups that *already* have state. Seeding a fresh group
    // here would give a group of all-NULLs a zero where SQL requires NULL —
    // the function decides when a group first acquires state, by folding a
    // value into it.
    std::map<int64_t, std::string> states;
    for (int64_t i = 0; i < group_ids->length(); ++i) {
        const int64_t id = group_ids->Value(i);
        if (states.count(id)) continue;
        auto it = stored.find(id);
        if (it != stored.end()) states.emplace(id, it->second);
    }

    fn->update(states, *group_ids, columns);

    for (auto& [id, state] : states) stored[id] = std::move(state);
    return empty_envelope();
}

vgi_rpc::Result Dispatcher::aggregate_combine(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_combine: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const auto execution_id = wire::get_binary(dto, "execution_id");
    auto fn = require_aggregate(function_name, schema_name);

    auto batch = wire::decode_ipc(wire::get_binary(dto, "merge_batch"));
    if (!batch) return empty_envelope();

    auto sources = int64_column(batch, "source_group_id", 0);
    auto targets = int64_column(batch, "target_group_id", 1);

    auto& stored = aggregate_states_[execution_id];
    for (int64_t i = 0; i < sources->length(); ++i) {
        const int64_t source_id = sources->Value(i);
        const int64_t target_id = targets->Value(i);
        auto source = stored.find(source_id);
        auto target = stored.find(target_id);

        const bool has_source = source != stored.end();
        const bool has_target = target != stored.end();
        // Neither side has state — an all-NULL group under default null
        // handling. Leaving the target stateless is what makes it finalize to
        // NULL instead of a seeded zero.
        if (!has_source && !has_target) continue;
        if (!has_source) continue;  // target already holds the answer
        stored[target_id] = has_target ? fn->combine(target->second, source->second)
                                       : source->second;
    }
    return empty_envelope();
}

vgi_rpc::Result Dispatcher::aggregate_finalize(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_finalize: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const auto execution_id = wire::get_binary(dto, "execution_id");
    auto fn = require_aggregate(function_name, schema_name);

    auto output_schema = wire::get_schema(dto, "output_schema");
    if (!output_schema) throw std::runtime_error("aggregate_finalize: no output_schema");

    auto ids_batch = wire::decode_ipc(wire::get_binary(dto, "group_ids_batch"));
    std::shared_ptr<arrow::Int64Array> group_ids;
    if (ids_batch && ids_batch->num_columns() > 0) {
        group_ids = std::dynamic_pointer_cast<arrow::Int64Array>(ids_batch->column(0));
    }
    if (!group_ids) {
        arrow::Int64Builder empty;
        std::shared_ptr<arrow::Array> array;
        (void)empty.Finish(&array);
        group_ids = std::static_pointer_cast<arrow::Int64Array>(array);
    }

    auto& stored = aggregate_states_[execution_id];
    std::vector<std::optional<std::string>> states;
    states.reserve(static_cast<size_t>(group_ids->length()));
    for (int64_t i = 0; i < group_ids->length(); ++i) {
        auto it = stored.find(group_ids->Value(i));
        states.push_back(it == stored.end() ? std::nullopt
                                            : std::optional<std::string>(it->second));
    }

    auto result_batch = fn->finalize(output_schema, *group_ids, states);
    auto payload = wire::ResultBuilder(payload_schema_of("aggregate_finalize"))
                       .set_binary("result_batch", wire::encode_ipc(result_batch))
                       .fill_defaults()
                       .finish();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(payload))
                                      .finish());
}

vgi_rpc::Result Dispatcher::aggregate_destructor(const vgi_rpc::Request& request) {
    // The protocol requires this call not to raise, and *everything* below can
    // — decoding the request, reading a field, decoding the group batch. The
    // try has to wrap all of it, not just the erase, or a malformed request
    // fails a query that has already produced its answer.
    try {
        auto dto = wire::get_ipc(request.batch(), "request");
        if (!dto) return empty_envelope();
        const auto execution_id = wire::get_optional_binary(dto, "execution_id").value_or("");

        // The whole execution is released, whatever group ids were named.
        //
        // Releasing only the named groups leaves the execution's own map entry
        // behind forever, and in a pooled worker "forever" is the life of the
        // process. Both references ignore the group list here and clear the
        // execution outright.
        aggregate_states_.erase(execution_id);
    } catch (const std::exception&) {
    }
    return empty_envelope();
}

}  // namespace vgi
