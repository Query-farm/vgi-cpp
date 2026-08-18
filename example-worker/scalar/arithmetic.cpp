// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Arithmetic scalar fixtures. Each one exercises a distinct part of the
// contract: a bind-decided result type, a constant argument, varargs, or a
// type bound.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>

#include <vgi/numeric.h>
#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// `double(value)` — the result type is the input's, widened for the carry, so
// it cannot be declared and has to come from bind.
class Double : public vgi::ScalarFunction {
public:
    std::string name() const override { return "double"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Doubles numeric values";
        md.examples = {
            {"SELECT double(21)", "Double an integer literal", "42"},
            {"SELECT double(value) FROM numbers", "Double every value in a column", std::nullopt},
        };
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto value = vgi::ArgSpec::any_column("value", 0, "Numeric value to double");
        value.with_bound(vgi::bounds::multipliable());
        return {value};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        return arrow::schema({arrow::field(
            "result", vgi::promote_for_addition(params.input_type(0)), /*nullable=*/true)});
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return vgi::double_first(params, batch);
    }
};

// `add_values(a, b)` — the result type is the common type of both inputs.
class AddValues : public vgi::ScalarFunction {
public:
    std::string name() const override { return "add_values"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Adds two numeric values";
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto first = vgi::ArgSpec::any_column("col1", 0, "First numeric value");
        first.with_bound(vgi::bounds::addable());
        auto second = vgi::ArgSpec::any_column("col2", 1, "Second numeric value");
        second.with_bound(vgi::bounds::addable());
        return {first, second};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        auto type = vgi::common_type_for_addition(params.input_type(0), params.input_type(1));
        return arrow::schema({arrow::field("result", type, /*nullable=*/true)});
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return vgi::add_two(params, batch);
    }
};

// `multiply(value, factor)` — `factor` is a bind-time constant, so it never
// appears in the input batch; column 0 is `value`, not `factor`.
class Multiply : public vgi::ScalarFunction {
public:
    std::string name() const override { return "multiply"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Multiplies a value by a constant factor";
        md.return_type = arrow::int64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Integer value to multiply"),
                vgi::ArgSpec::constant_arg("factor", 1, "int64", "Multiplication factor")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t factor = params.arguments.const_int64(1).value_or(1);
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(0), arrow::int64()));

        arrow::Int64Builder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(values->Value(i) * factor);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `sum_values(...)` — varargs, summed in the bound type.
class SumValues : public vgi::ScalarFunction {
public:
    std::string name() const override { return "sum_values"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Sum multiple numeric values";
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto values = vgi::ArgSpec::any_column("values", 0, "Numeric values to sum");
        values.with_varargs().with_bound(vgi::bounds::addable());
        return {values};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        return arrow::schema({arrow::field(
            "result", vgi::promote_for_addition(params.input_type(0)), /*nullable=*/true)});
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        if (batch->num_columns() == 0) {
            throw std::invalid_argument("sum_values requires at least 1 value");
        }
        const auto type = output_type(params);
        auto accumulator = cast_to(batch->column(0), type);
        for (int i = 1; i < batch->num_columns(); ++i) {
            auto column = cast_to(batch->column(i), type);
            auto sum = arrow::compute::CallFunction("add", {accumulator, column});
            if (!sum.ok()) throw std::runtime_error("sum_values: " + sum.status().message());
            accumulator = sum.MoveValueUnsafe().make_array();
        }
        return result(params, cast_to(accumulator, type));
    }
};

}  // namespace

void register_arithmetic(vgi::Worker& worker) {
    worker.register_scalar(std::make_shared<Double>());
    worker.register_scalar(std::make_shared<AddValues>());
    worker.register_scalar(std::make_shared<Multiply>());
    worker.register_scalar(std::make_shared<SumValues>());
}

}  // namespace example
