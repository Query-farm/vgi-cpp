// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/types.h"

namespace vgi {

// What the engine knows at bind time: the resolved argument types and the
// values of any constant arguments.  A function that returns a fixed type can
// ignore all of it.
struct BindParams {
    std::shared_ptr<arrow::Schema> input_schema;
    // One row: the constant arguments, in ArgSpec order.  Null when the
    // function declares none.
    std::shared_ptr<arrow::RecordBatch> constants;
    std::string catalog_name;
    std::string schema_name;
};

// What a bind produced, handed back to every process() call for that
// statement.  Splitting it from BindParams is what lets one bound function
// serve many batches without re-deciding its output shape.
struct ProcessParams {
    std::shared_ptr<arrow::Schema> output_schema;
    std::string catalog_name;
    std::string schema_name;
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
