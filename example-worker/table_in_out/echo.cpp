// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Table-in-out fixtures: functions that take a relation and produce one.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>

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
        // Declared so the bound output schema arrives already narrowed: the
        // plan-shape assertions check that DuckDB pushed the projection in
        // rather than stacking a narrowing PROJECTION above the operator.
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::vector<vgi::EmittedBatch> process(
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
        // The witness only says something when the narrowing actually reaches
        // the worker, which is what declaring this buys.
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::vector<vgi::EmittedBatch> process(
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

// `repeat_inputs(repeat_count, data)` — duplicate each input batch N times.
//
// The count is a bind-time constant, so column 0 of the input batch is the
// relation's first column, not the count.
class RepeatInputs : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "repeat_inputs"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Duplicates each input batch N times";
        md.categories = {"transform", "augmentation"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("repeat_count", 0, "int64",
                                           "Times to repeat each input"),
                vgi::ArgSpec::table("data", 1, "Input relation")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        const int64_t count = params.arguments.const_int64(0).value_or(1);
        // Rejected at bind rather than silently clamped: a caller asking for
        // zero copies means something, and quietly giving them one is worse
        // than telling them.
        if (count < 1) throw std::invalid_argument("Repeat count must be at least 1");
        if (!params.input_schema) {
            throw std::invalid_argument("input_schema is required but was None");
        }
        return params.input_schema;
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto count =
            std::max<int64_t>(1, params.arguments.const_int64(0).value_or(1));
        auto projected = vgi::project_batch(batch, params.output_schema);

        // Concatenated into one batch rather than emitted N times: the engine
        // treats each emitted batch as a separate unit downstream, and the
        // fixture's tests expect one.
        std::vector<std::shared_ptr<arrow::RecordBatch>> copies(
            static_cast<size_t>(count), projected);
        auto table = arrow::Table::FromRecordBatches(projected->schema(), copies);
        if (!table.ok()) throw std::runtime_error("repeat_inputs: " + table.status().message());
        auto combined = table.ValueUnsafe()->CombineChunks();
        if (!combined.ok()) {
            throw std::runtime_error("repeat_inputs: " + combined.status().message());
        }
        arrow::TableBatchReader reader(*combined.ValueUnsafe());
        std::shared_ptr<arrow::RecordBatch> out;
        if (!reader.ReadNext(&out).ok() || !out) {
            // An empty input repeated is still empty, which is a legitimate
            // answer rather than a failure.
            return {projected};
        }
        return {out};
    }
};

}  // namespace

void register_table_in_out(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<RepeatInputs>());
    worker.register_table_in_out(std::make_shared<Echo>());
    worker.register_table_in_out(std::make_shared<EchoWitness>());
}

}  // namespace example
