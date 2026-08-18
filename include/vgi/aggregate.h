// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/arguments.h"

#include "vgi/function.h"
#include "vgi/types.h"

namespace vgi {

// The column carrying the engine's group id on every aggregate input batch.
inline constexpr const char* kGroupColumnName = "__vgi_group_id";

// An aggregate function.
//
// Unlike the other kinds this is not a stream: the engine drives it through
// five separate unary calls — bind, update, combine, finalize, destructor —
// and the worker holds per-group state between them, keyed by execution.
//
// State is opaque bytes rather than a typed object because it has to survive
// those round trips and, on the HTTP transport, possibly a different worker
// process. Serializing it is the contract, not an implementation detail.
class AggregateFunction {
public:
    virtual ~AggregateFunction() = default;

    virtual std::string name() const = 0;
    virtual FunctionMetadata metadata() const = 0;
    virtual std::vector<ArgSpec> argument_specs() const = 0;

    // The output schema, always a single column named `result`.
    virtual std::shared_ptr<arrow::Schema> bind(const BindParams& params) const = 0;

    // Fold `columns` into `states`, keyed by group id.
    //
    // `states` arrives holding only groups that already have state — a group
    // seen for the first time is *absent*, not seeded. That distinction is
    // load-bearing: an implementation that inserts on first sight turns a
    // group of all-NULLs into a zero, where SQL requires NULL. Insert only
    // when a value is actually folded in.
    virtual void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                        const std::vector<std::shared_ptr<arrow::Array>>& columns) const = 0;

    // Merge `source` into `target`, returning the new target state.
    virtual std::string combine(const std::string& target, const std::string& source) const = 0;

    // One output row per entry of `group_ids`. `states[i]` is the state for
    // `group_ids[i]`, absent if that group never accumulated anything.
    virtual std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array& group_ids,
        const std::vector<std::optional<std::string>>& states) const = 0;

    // Like `finalize`, with the bind-time arguments in hand.
    //
    // Needed by an aggregate whose *result* depends on a constant argument —
    // a percentile's fraction, say — which the group states cannot carry
    // because they are folded before it is known. The default ignores them, so
    // an aggregate that does not care overrides only `finalize`.
    virtual std::shared_ptr<arrow::RecordBatch> finalize_with_arguments(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array& group_ids,
        const std::vector<std::optional<std::string>>& states, const Arguments& arguments) const {
        (void)arguments;
        return finalize(output_schema, group_ids, states);
    }

    // ── Windowed aggregation ─────────────────────────────────────────────
    //
    // A different shape from grouped aggregation: the engine ships a whole
    // partition once, then asks for one value per output row over a list of
    // sub-frames within it. Only an aggregate declaring `supports_window`
    // is asked, and the default refuses rather than returning something wrong.

    virtual bool supports_window() const { return false; }

    // ── Streaming-partitioned aggregation ────────────────────────────────
    //
    // A third shape: the engine opens a session, feeds chunks whose columns
    // are `[partition keys…, order keys…, values…]`, and expects one output
    // value per *input row*. State is carried across chunks per partition key,
    // which is what lets a running total continue across a chunk boundary.

    virtual bool streaming_partitioned() const { return false; }

    // Process one chunk, returning one value per row. `states` is the
    // cross-chunk map, keyed by the partition key's bytes; the framework loads
    // and persists it around this call.
    virtual std::shared_ptr<arrow::Array> streaming_chunk(
        const std::shared_ptr<arrow::RecordBatch>& chunk, size_t partition_key_count,
        size_t order_key_count, std::map<std::string, std::string>& states) const {
        (void)chunk;
        (void)partition_key_count;
        (void)order_key_count;
        (void)states;
        throw std::runtime_error("streaming_chunk() not supported by this aggregate");
    }

    // Evaluate the aggregate over `frames` of `partition`, one value per
    // entry. Each frame is a half-open [begin, end) row range.
    //
    // `filter_mask`, when non-empty, is one flag per partition row from a
    // `FILTER (WHERE …)` clause. It cannot be applied by pre-filtering the
    // partition, because frame bounds index the *original* rows — so an
    // implementation must skip masked-out rows as it walks each frame.
    // Ignoring it makes a filtered window silently aggregate everything.
    virtual std::shared_ptr<arrow::Array> window(
        const std::shared_ptr<arrow::RecordBatch>& partition,
        const std::shared_ptr<arrow::Schema>& output_schema,
        const std::vector<std::vector<std::pair<int64_t, int64_t>>>& frames,
        const std::vector<bool>& filter_mask) const {
        (void)partition;
        (void)output_schema;
        (void)frames;
        (void)filter_mask;
        throw std::runtime_error("window() not supported by this aggregate");
    }
};

}  // namespace vgi
