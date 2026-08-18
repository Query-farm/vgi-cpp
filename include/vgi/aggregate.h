// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

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
    virtual void update(std::map<int64_t, std::string>& states,
                        const arrow::Int64Array& group_ids,
                        const std::vector<std::shared_ptr<arrow::Array>>& columns) const = 0;

    // Merge `source` into `target`, returning the new target state.
    virtual std::string combine(const std::string& target,
                                const std::string& source) const = 0;

    // One output row per entry of `group_ids`. `states[i]` is the state for
    // `group_ids[i]`, absent if that group never accumulated anything.
    virtual std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema,
        const arrow::Int64Array& group_ids,
        const std::vector<std::optional<std::string>>& states) const = 0;
};

}  // namespace vgi
