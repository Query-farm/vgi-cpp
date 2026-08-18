// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/arguments.h"
#include "vgi/cache_control.h"
#include "vgi/pushdown.h"
#include "vgi/settings.h"
#include "vgi/storage.h"
#include "vgi/types.h"

namespace vgi {

// What the engine knows at bind time: the resolved argument types and the
// values of any constant arguments.  A function that returns a fixed type can
// ignore all of it.
struct BindParams {
    // The types of the columns the call site will ship. A function whose
    // result type follows its input reads it here.
    std::shared_ptr<arrow::Schema> input_schema;
    // Argument values known at bind: every constant argument, and the declared
    // type of every argument whether constant or not.
    Arguments arguments;
    Settings settings;
    Secrets secrets;
    // Whether the engine has already resolved this function's secret lookups.
    // A bind that asks again once they are resolved would loop.
    bool secrets_resolved = false;
    // The COPY destination, when this bind is part of a `COPY … TO`. A writer
    // scopes its secret lookups to the path.
    std::optional<std::string> copy_to_format;
    std::optional<std::string> copy_to_path;
    std::string catalog_name;
    std::string schema_name;

    // The declared type of positional argument `index`, preferring the
    // argument list and falling back to the input schema.
    //
    // Both are needed: a constant argument never appears in the input schema,
    // and a column argument's field in the argument list may be a placeholder.
    std::shared_ptr<arrow::DataType> input_type(size_t index) const;
};

// The ORDER BY and TABLESAMPLE clauses the optimizer folded into this scan.
//
// Hints, not obligations: the engine still sorts and still limits, so a scan
// that ignores the ordering ones is merely slower. Sampling is the exception —
// declaring `sampling_pushdown` makes DuckDB drop its own sampling operator, so
// a function that advertises it and then ignores `tablesample_percentage`
// returns every row.
//
// Which of these arrive is decided by DuckDB's optimizers, not by the query
// text: the ordering hints need a LIMIT above an ORDER BY over a plain column
// reference, and only TABLESAMPLE SYSTEM is ever pushed.
struct ScanHints {
    std::optional<std::string> order_by_column;
    // Wire spellings, echoed as they arrive: "ASC" / "DESC", and
    // "NULLS_FIRST" / "NULLS_LAST".
    std::optional<std::string> order_by_direction;
    std::optional<std::string> order_by_null_order;
    // Rows the scan may stop after — LIMIT plus OFFSET, since the offset rows
    // must still be produced to be skipped.
    std::optional<int64_t> order_by_limit;
    std::optional<double> tablesample_percentage;
    std::optional<int64_t> tablesample_seed;
};

// What a bind produced, handed back to every process() call for that
// statement.  Splitting it from BindParams is what lets one bound function
// serve many batches without re-deciding its output shape.
struct ProcessParams {
    std::shared_ptr<arrow::Schema> output_schema;
    Arguments arguments;
    Settings settings;
    Secrets secrets;
    std::string catalog_name;
    std::string schema_name;
    // The engine's id for this function execution, echoed on every call that
    // belongs to it. A function holding state across calls — a buffering sink,
    // an aggregate — keys on this; a stateless one can ignore it.
    std::string execution_id;
    // The predicates the engine pushed into this scan. Empty when none were,
    // or when the function did not declare `filter_pushdown`.
    PushdownFilters pushdown_filters;
    // The ORDER BY / TABLESAMPLE clauses the optimizer folded into this scan.
    // Every field is empty when the optimizer pushed nothing.
    ScanHints scan_hints;
    // The COPY destination, when this call is part of a `COPY … TO`.
    //
    // The path is not an option: it comes from the COPY statement itself, so a
    // writer reads it here rather than from its declared arguments.
    std::optional<std::string> copy_to_format;
    std::optional<std::string> copy_to_path;
    // Cross-process state, scoped by `execution_id`. Set for every call.
    //
    // Needed rather than optional: the engine parallelizes a buffering sink
    // across worker *processes*, so in-memory accumulation is silently empty
    // by the time finalize runs in a different one.
    std::shared_ptr<FunctionStorage> storage;
};

// A scalar function: one output row per input row.
//
// The engine calls bind() once per statement to settle the output schema, then
// process() once per input batch.  Neither is allowed to block indefinitely —
// dispatch is single-threaded, as it is in the RPC framework beneath.
class ScalarFunction {
public:
    virtual ~ScalarFunction() = default;

    virtual std::string name() const = 0;
    virtual FunctionMetadata metadata() const = 0;
    virtual std::vector<ArgSpec> argument_specs() const = 0;

    // Settle the output schema for this call site.  The default answers from
    // metadata().return_type, which is enough for any function whose result
    // type does not depend on its arguments.
    virtual std::shared_ptr<arrow::Schema> bind(const BindParams& params) const;

    virtual std::shared_ptr<arrow::RecordBatch> process(
        const ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const = 0;

    // The secrets this call needs resolved, when which ones depends on the
    // arguments — a scope taken from a path argument, say.
    //
    // Answered from `metadata()` by default, which is enough for a fixed list.
    // A function whose scope is an argument cannot use that list: `metadata()`
    // is asked before any call site exists.
    virtual std::vector<SecretLookup> secret_lookups(const BindParams&) const {
        return metadata().required_secrets;
    }

    // A result-cache advertisement for this function's output.
    //
    // Only sound for a deterministic map: the engine may serve a later call
    // the earlier answer, so a function whose result depends on anything
    // outside its arguments must not advertise. Nothing by default.
    virtual std::optional<CacheControl> cache_control() const { return std::nullopt; }
};

}  // namespace vgi
