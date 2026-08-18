// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/function.h"
#include "vgi/types.h"

namespace vgi {

// A function that takes a table and produces a table.
//
// Driven as an exchange stream, but unlike a scalar function it is not
// row-for-row: one input batch may produce zero, one, or many output batches.
// That is the whole difference in the contract, and it is why `process`
// returns a vector rather than a batch.
class TableInOutFunction {
public:
    virtual ~TableInOutFunction() = default;

    virtual std::string name() const = 0;
    virtual FunctionMetadata metadata() const = 0;
    virtual std::vector<ArgSpec> argument_specs() const = 0;

    // The output schema. Defaults to passing the input schema through, which
    // is what a filter or a row-wise transform wants.
    virtual std::shared_ptr<arrow::Schema> bind(const BindParams& params) const;

    virtual std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const = 0;

    // Whether the function accumulates across the whole stream and flushes at
    // the end. Declaring it drives the engine's FINALIZE phase; without it,
    // `finish` is never called.
    virtual bool has_finish() const { return false; }

    virtual std::vector<std::shared_ptr<arrow::RecordBatch>> finish(
        const ProcessParams&) const {
        return {};
    }
};

// Project `batch` onto `schema` by column *name*, filling absent columns with
// nulls.
//
// By name and not by position: the engine narrows a table-in-out's output
// schema when a projection is pushed down, and the surviving columns keep
// their names but not their indices.
std::shared_ptr<arrow::RecordBatch> project_batch(
    const std::shared_ptr<arrow::RecordBatch>& batch,
    const std::shared_ptr<arrow::Schema>& schema);

}  // namespace vgi
