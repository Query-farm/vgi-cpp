// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Table-in-out fixtures: functions that take a relation and produce one.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// `echo(data)` — emits each input batch unchanged.
//
// The default `process` projects the input onto the bound output schema, which
// is exactly passthrough when nothing was pushed down and exactly the right
// narrowing when a projection was.
class Echo : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "echo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Passthrough function that emits each input batch unchanged";
        md.categories = {"utility", "debug"};
        md.tags = {{"category", "debug"}, {"type", "passthrough"}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return {vgi::project_batch(batch, params.output_schema)};
    }
};

// `echo_witness(data)` — reports how many columns survived projection
// pushdown, by filling every column with that count.
//
// The value is the *observed* output schema's width, which is the only way a
// test can see what the engine actually asked for.
class EchoWitness : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "echo_witness"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Emits len(observed_output_schema) per column — projection probe";
        md.categories = {"test", "pushdown"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t observed = params.output_schema->num_fields();
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(observed));

        for (int i = 0; i < params.output_schema->num_fields(); ++i) {
            arrow::Int64Builder builder;
            (void)builder.Reserve(batch->num_rows());
            for (int64_t row = 0; row < batch->num_rows(); ++row) {
                (void)builder.Append(observed);
            }
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            columns.push_back(cast_to(array, params.output_schema->field(i)->type()));
        }
        return {arrow::RecordBatch::Make(params.output_schema, batch->num_rows(), columns)};
    }
};

}  // namespace

void register_table_in_out(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<Echo>());
    worker.register_table_in_out(std::make_shared<EchoWitness>());
}

}  // namespace example
