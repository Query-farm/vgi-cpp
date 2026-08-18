// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The aggregate half of the protocol.
//
// An aggregate is not a stream. The engine drives it through five unary calls
// and the worker holds the per-group state between them, keyed by an
// execution id the engine mints at bind and echoes on every later call.

#include <stdexcept>

#include <arrow/array.h>

#include <cstdlib>
#include <arrow/array/builder_primitive.h>
#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>

#include "dispatcher.h"
#include "methods.h"
#include "vgi/storage.h"

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

std::string state_key(int64_t group_id) {
    return "aggregate.state." + std::to_string(group_id);
}

std::string partition_key(int64_t partition_id) {
    return "window.partition." + std::to_string(partition_id);
}

std::string schema_key(int64_t partition_id) {
    return "window.schema." + std::to_string(partition_id);
}

std::string mask_key(int64_t partition_id) {
    return "window.mask." + std::to_string(partition_id);
}

std::string rows_key(int64_t partition_id) {
    return "window.rows." + std::to_string(partition_id);
}

// Least-significant bit first, which is how Arrow packs a validity bitmap and
// how the engine packs this one.
std::vector<bool> unpack_mask(const std::string& packed, int64_t row_count) {
    if (packed.empty() || row_count <= 0) return {};
    std::vector<bool> mask(static_cast<size_t>(row_count), false);
    for (int64_t i = 0; i < row_count; ++i) {
        const size_t byte = static_cast<size_t>(i) / 8;
        if (byte >= packed.size()) break;
        mask[static_cast<size_t>(i)] =
            (static_cast<unsigned char>(packed[byte]) >> (i % 8)) & 1;
    }
    return mask;
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
    const Scope scope{attachment_of(dto).catalog, schema_name};
    auto fn = require_aggregate(function_name, scope);

    BindParams params;
    params.input_schema = wire::get_schema(dto, "input_schema");
    params.arguments = Arguments::parse(wire::get_optional_binary(dto, "arguments").value_or(""));
    params.settings = Settings::parse(wire::get_optional_binary(dto, "settings").value_or(""));
    params.secrets = Secrets::parse(wire::get_optional_binary(dto, "secrets").value_or(""));
    params.catalog_name = scope.catalog;
    params.schema_name = schema_name;

    auto output_schema = fn->bind(params);
    if (!output_schema) {
        throw std::runtime_error("aggregate '" + function_name + "' bound to no schema");
    }

    // The execution id keys this aggregation's state for its whole life, and
    // that state lives on disk under $TMPDIR — shared by every worker process
    // the pool spawns. So it has to be unique across *processes*, not just
    // within one: `next_execution_id` qualifies its counter with the pid and
    // two clocks for exactly that reason. The engine treats it as opaque.
    auto execution_id = next_execution_id();
    // Stashed for finalize, which carries no arguments of its own.
    if (auto arguments = wire::get_optional_binary(dto, "arguments"); arguments) {
        default_storage()->kv_put(execution_id, "aggregate.arguments", *arguments);
    }

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
    const Scope scope{attachment_of(dto).catalog, schema_name};
    const auto execution_id = wire::get_binary(dto, "execution_id");
    auto fn = require_aggregate(function_name, scope);

    auto batch = wire::decode_ipc(wire::get_binary(dto, "input_batch"));
    if (!batch) return empty_envelope();
    auto [group_ids, columns] = split_group_ids(batch);

    auto* store = default_storage().get();

    // Pre-load only groups that *already* have state. Seeding a fresh group
    // here would give a group of all-NULLs a zero where SQL requires NULL —
    // the function decides when a group first acquires state, by folding a
    // value into it.
    std::map<int64_t, std::string> states;
    for (int64_t i = 0; i < group_ids->length(); ++i) {
        const int64_t id = group_ids->Value(i);
        if (states.count(id)) continue;
        if (auto held = store->kv_get(execution_id, state_key(id))) states.emplace(id, *held);
    }

    fn->update(states, *group_ids, columns);

    for (auto& [id, state] : states) store->kv_put(execution_id, state_key(id), state);
    return empty_envelope();
}

vgi_rpc::Result Dispatcher::aggregate_combine(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_combine: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const Scope scope{attachment_of(dto).catalog, schema_name};
    const auto execution_id = wire::get_binary(dto, "execution_id");
    auto fn = require_aggregate(function_name, scope);

    auto batch = wire::decode_ipc(wire::get_binary(dto, "merge_batch"));
    if (!batch) return empty_envelope();

    auto sources = int64_column(batch, "source_group_id", 0);
    auto targets = int64_column(batch, "target_group_id", 1);

    auto* store = default_storage().get();
    for (int64_t i = 0; i < sources->length(); ++i) {
        const int64_t source_id = sources->Value(i);
        const int64_t target_id = targets->Value(i);
        auto source = store->kv_get(execution_id, state_key(source_id));
        // Neither side has state — an all-NULL group under default null
        // handling. Leaving the target stateless is what makes it finalize to
        // NULL instead of a seeded zero.
        if (!source) continue;  // target already holds the answer, if any
        auto target = store->kv_get(execution_id, state_key(target_id));
        store->kv_put(execution_id, state_key(target_id),
                      target ? fn->combine(*target, *source) : *source);
    }
    return empty_envelope();
}

vgi_rpc::Result Dispatcher::aggregate_finalize(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_finalize: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const Scope scope{attachment_of(dto).catalog, schema_name};
    const auto execution_id = wire::get_binary(dto, "execution_id");
    auto fn = require_aggregate(function_name, scope);

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

    auto* store = default_storage().get();
    std::vector<std::optional<std::string>> states;
    states.reserve(static_cast<size_t>(group_ids->length()));
    for (int64_t i = 0; i < group_ids->length(); ++i) {
        states.push_back(store->kv_get(execution_id, state_key(group_ids->Value(i))));
    }

    // Bind-time arguments come from the stash, since a finalize request
    // carries none and may run on a worker that never bound.
    auto arguments = Arguments::parse(
        default_storage()->kv_get(execution_id, "aggregate.arguments").value_or(""));
    auto result_batch =
        fn->finalize_with_arguments(output_schema, *group_ids, states, arguments);
    auto payload = wire::ResultBuilder(payload_schema_of("aggregate_finalize"))
                       .set_binary("result_batch", wire::encode_ipc(result_batch))
                       .fill_defaults()
                       .finish();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(payload))
                                      .finish());
}

vgi_rpc::Result Dispatcher::aggregate_streaming_open(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_streaming_open: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const Scope scope{attachment_of(dto).catalog, schema_name};
    (void)require_aggregate(function_name, scope);

    // The session's shape is fixed at open and echoed on every chunk, which
    // carries none of it — so it is stashed, like every other cross-call
    // context here.
    auto execution_id = next_execution_id();
    auto* store = default_storage().get();
    store->kv_put(execution_id, "streaming.partition_keys",
                  std::to_string(wire::get_optional_int64(dto, "partition_key_count").value_or(0)));
    store->kv_put(execution_id, "streaming.order_keys",
                  std::to_string(wire::get_optional_int64(dto, "order_key_count").value_or(0)));
    store->kv_put(execution_id, "streaming.output_schema",
                  wire::get_optional_binary(dto, "output_schema").value_or(""));

    auto payload = wire::ResultBuilder(payload_schema_of("aggregate_streaming_open"))
                       .set_binary("execution_id", execution_id)
                       .fill_defaults()
                       .finish();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(payload))
                                      .finish());
}

vgi_rpc::Result Dispatcher::aggregate_streaming_chunk(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_streaming_chunk: empty request");

    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const Scope scope{attachment_of(dto).catalog, schema_name};
    const auto execution_id = wire::get_binary(dto, "execution_id");
    auto fn = require_aggregate(function_name, scope);

    auto* store = default_storage().get();
    const auto count = [&](const char* key) -> size_t {
        return static_cast<size_t>(
            std::strtoull(store->kv_get(execution_id, key).value_or("0").c_str(), nullptr, 10));
    };
    auto output_schema =
        wire::decode_schema(store->kv_get(execution_id, "streaming.output_schema").value_or(""));

    auto chunk = wire::decode_ipc(wire::get_binary(dto, "input_batch"));
    if (!chunk) throw std::runtime_error("aggregate_streaming_chunk: empty chunk");

    // Per-partition state, carried across chunks. Loaded whole rather than per
    // key because a chunk touches few partitions and the map is small.
    std::map<std::string, std::string> states;
    for (const auto& [id, blob] : store->scan(execution_id, "streaming", "", 0, SIZE_MAX)) {
        (void)id;
        const auto separator = blob.find('\0');
        if (separator == std::string::npos) continue;
        states[blob.substr(0, separator)] = blob.substr(separator + 1);
    }

    auto values = fn->streaming_chunk(chunk, count("streaming.partition_keys"),
                                      count("streaming.order_keys"), states);

    // Rewritten wholesale: the log is append-only, so the current state is the
    // newest entry per key and stale ones are simply never read.
    for (const auto& [key, state] : states) {
        store->append(execution_id, "streaming", "", key + std::string(1, '\0') + state);
    }

    auto result_batch = arrow::RecordBatch::Make(
        output_schema ? output_schema
                      : arrow::schema({arrow::field("result", values->type(), true)}),
        values->length(), {values});
    auto payload = wire::ResultBuilder(payload_schema_of("aggregate_streaming_chunk"))
                       .set_binary("result_batch", wire::encode_ipc(result_batch))
                       .fill_defaults()
                       .finish();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(payload))
                                      .finish());
}

vgi_rpc::Result Dispatcher::aggregate_streaming_close(const vgi_rpc::Request& request) {
    // Must not raise; the session's state is the only thing to release.
    try {
        if (auto dto = wire::get_ipc(request.batch(), "request")) {
            const auto execution_id =
                wire::get_optional_binary(dto, "execution_id").value_or("");
            if (!execution_id.empty()) default_storage()->clear(execution_id);
        }
    } catch (const std::exception&) {
    }
    return empty_envelope();
}

vgi_rpc::Result Dispatcher::aggregate_window_init(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_window_init: empty request");

    const auto execution_id = wire::get_binary(dto, "execution_id");
    const auto partition_id = wire::get_int64(dto, "partition_id");
    auto partition = wire::decode_ipc(wire::get_binary(dto, "partition_batch"));
    auto output_schema = wire::get_schema(dto, "output_schema");

    // Stored, not held in memory.
    //
    // The engine ships the partition once and then asks for many output rows
    // over it — but those requests can land on a *different* worker process,
    // exactly as buffering's do. An in-process map is silently empty there,
    // and the symptom is "no partition for this execution" rather than a wrong
    // answer, which is at least honest but still a broken query.
    auto* store = default_storage().get();
    store->kv_put(execution_id, partition_key(partition_id),
                  partition ? wire::encode_ipc(partition) : std::string{});
    store->kv_put(execution_id, schema_key(partition_id),
                  output_schema ? wire::encode_schema(output_schema) : std::string{});
    // The FILTER mask travels as packed bits, one per partition row.
    store->kv_put(execution_id, mask_key(partition_id),
                  wire::get_optional_binary(dto, "filter_mask").value_or(""));
    store->kv_put(execution_id, rows_key(partition_id),
                  std::to_string(wire::get_optional_int64(dto, "row_count").value_or(0)));
    return empty_envelope();
}

vgi_rpc::Result Dispatcher::aggregate_window(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_window: empty request");
    return window_result(dto, /*batched=*/false);
}

vgi_rpc::Result Dispatcher::aggregate_window_batch(const vgi_rpc::Request& request) {
    auto dto = wire::get_ipc(request.batch(), "request");
    if (!dto) throw std::runtime_error("aggregate_window_batch: empty request");
    return window_result(dto, /*batched=*/true);
}

vgi_rpc::Result Dispatcher::aggregate_window_destructor(const vgi_rpc::Request& request) {
    // Must not raise; a stored partition is large, so failing to release it
    // matters more than the error would.
    try {
        if (auto dto = wire::get_ipc(request.batch(), "request")) {
            const auto execution_id =
                wire::get_optional_binary(dto, "execution_id").value_or("");
            if (execution_id.empty()) return empty_envelope();
            // Only the named partition. An execution holds several at once —
            // the engine evicts them as it finishes each — so clearing the
            // whole scope destroys partitions still in use, which surfaces as
            // "unknown partition N" on the next call.
            const auto partition_id = wire::get_optional_int64(dto, "partition_id");
            if (!partition_id) return empty_envelope();
            auto* store = default_storage().get();
            store->kv_del(execution_id, partition_key(*partition_id));
            store->kv_del(execution_id, schema_key(*partition_id));
            store->kv_del(execution_id, mask_key(*partition_id));
            store->kv_del(execution_id, rows_key(*partition_id));
        }
    } catch (const std::exception&) {
    }
    return empty_envelope();
}

// One or many output rows over a held partition.
//
// The batched call exists because a window over N rows is N round trips
// otherwise; `frames_per_row` says how many sub-frames belong to each output
// row, so one request covers a run of them.
vgi_rpc::Result Dispatcher::window_result(const std::shared_ptr<arrow::RecordBatch>& dto,
                                          bool batched) {
    const auto function_name = wire::get_string(dto, "function_name");
    const auto schema_name = wire::get_optional_string(dto, "schema_name").value_or("main");
    const Scope scope{attachment_of(dto).catalog, schema_name};
    const auto execution_id = wire::get_binary(dto, "execution_id");
    const auto partition_id = wire::get_int64(dto, "partition_id");
    auto fn = require_aggregate(function_name, scope);

    auto* store = default_storage().get();
    auto stored = store->kv_get(execution_id, partition_key(partition_id));
    if (!stored) {
        throw std::runtime_error("aggregate_window: unknown partition " +
                                 std::to_string(partition_id));
    }
    auto partition = wire::decode_ipc(*stored);
    auto output_schema =
        wire::decode_schema(store->kv_get(execution_id, schema_key(partition_id)).value_or(""));
    if (!output_schema) {
        // `window_init` is what stores it. Reaching `window` without one means
        // the init never arrived or carried no schema, and building a batch
        // against a null schema dereferences it.
        throw std::runtime_error("aggregate_window: partition " +
                                 std::to_string(partition_id) + " has no output schema");
    }

    const auto starts = wire::get_int64_list(dto, "frame_starts");
    const auto ends = wire::get_int64_list(dto, "frame_ends");
    std::vector<std::vector<std::pair<int64_t, int64_t>>> frames;

    if (batched) {
        const auto per_row = wire::get_int64_list(dto, "frames_per_row");
        size_t next = 0;
        for (int64_t count : per_row) {
            std::vector<std::pair<int64_t, int64_t>> row;
            for (int64_t i = 0; i < count && next < starts.size() && next < ends.size();
                 ++i, ++next) {
                row.emplace_back(starts[next], ends[next]);
            }
            frames.push_back(std::move(row));
        }
    } else {
        std::vector<std::pair<int64_t, int64_t>> row;
        for (size_t i = 0; i < starts.size() && i < ends.size(); ++i) {
            row.emplace_back(starts[i], ends[i]);
        }
        frames.push_back(std::move(row));
    }

    // Unpacked here rather than in every implementation: the bit order is a
    // wire detail, and getting it backwards silently inverts the filter.
    const auto row_count =
        std::strtoll(store->kv_get(execution_id, rows_key(partition_id)).value_or("0").c_str(),
                     nullptr, 10);
    auto mask = unpack_mask(store->kv_get(execution_id, mask_key(partition_id)).value_or(""),
                            row_count);

    auto values = fn->window(partition, output_schema, frames, mask);
    auto result_batch = arrow::RecordBatch::Make(output_schema, values->length(), {values});
    auto payload = wire::ResultBuilder(payload_schema_of(batched ? "aggregate_window_batch"
                                                                : "aggregate_window"))
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
        if (!execution_id.empty()) default_storage()->clear(execution_id);
    } catch (const std::exception&) {
    }
    return empty_envelope();
}

}  // namespace vgi
