// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/arguments.h"
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
};

}  // namespace vgi
