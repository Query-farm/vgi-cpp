// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Overloaded formatting fixtures.
//
// One name, several argument shapes: the engine resolves which overload a call
// means from the arity and types at the call site, so each shape is a separate
// registration under the same name. That resolution is what these fixtures
// exist to check — the formatting itself is incidental.

#include <charconv>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

vgi::FunctionMetadata format_metadata(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.return_type = arrow::utf8();
    md.categories = {"string", "utility"};
    return md;
}

// The call's numeric column, as doubles. A column that cannot be read as a
// number yields all-null rather than failing: the engine has already resolved
// the overload, so a cast failure here is a fixture bug, not a user error.
std::shared_ptr<arrow::DoubleArray> numeric_column(
    const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (batch->num_columns() == 0) return nullptr;
    return std::dynamic_pointer_cast<arrow::DoubleArray>(
        cast_to(batch->column(batch->num_columns() - 1), arrow::float64()));
}

std::string fixed(double value, int precision) {
    std::ostringstream out;
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(precision);
    out << value;
    return out.str();
}

// Rendered the way Python's `str(float)` renders it: a whole number keeps its
// `.0`, everything else takes the shortest round-tripping form. The tests
// compare the text, so the spelling is the contract.
std::string python_float(double value) {
    if (std::isfinite(value) && value == std::floor(value)) return fixed(value, 1);
    // Shortest round-tripping form, which is what Python prints. An
    // ostringstream at 17 digits would render 3.14 as 3.1400000000000001.
    char digits[32];
    const auto end = std::to_chars(digits, digits + sizeof(digits), value).ptr;
    return std::string(digits, end);
}

// `format_number(value)` / `(precision, value)` / `(precision, prefix, value)`.
class FormatNumber : public vgi::ScalarFunction {
public:
    enum class Shape { Default, Precision, Prefixed };

    explicit FormatNumber(Shape shape) : shape_(shape) {}

    std::string name() const override { return "format_number"; }

    vgi::FunctionMetadata metadata() const override {
        switch (shape_) {
            case Shape::Precision: return format_metadata("Format number with specified precision");
            case Shape::Prefixed: return format_metadata("Format number with precision and prefix");
            case Shape::Default: break;
        }
        return format_metadata("Format number with default precision (0 decimals)");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto precision = vgi::ArgSpec::constant_arg("precision", 0, "int64", "Decimals");
        precision.with_range(0, 10);
        switch (shape_) {
            case Shape::Precision:
                return {precision, vgi::ArgSpec::column("value", 1, "float64", "Number")};
            case Shape::Prefixed:
                return {precision, vgi::ArgSpec::constant_arg("prefix", 1, "varchar", "Prefix"),
                        vgi::ArgSpec::column("value", 2, "float64", "Number")};
            case Shape::Default: break;
        }
        return {vgi::ArgSpec::column("value", 0, "float64", "Number")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int precision = shape_ == Shape::Default
                                  ? 0
                                  : static_cast<int>(std::max<int64_t>(
                                        0, params.arguments.const_int64(0).value_or(0)));
        const std::string prefix =
            shape_ == Shape::Prefixed ? params.arguments.const_string(1).value_or("") : "";

        auto values = numeric_column(batch);
        arrow::StringBuilder formatted;
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            if (!values || values->IsNull(i)) {
                (void)formatted.AppendNull();
            } else {
                (void)formatted.Append(prefix + fixed(values->Value(i), precision));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)formatted.Finish(&array);
        return result(params, array);
    }

private:
    Shape shape_;
};

// `smart_format(width, value)` / `(prefix, value)` — two overloads that differ
// only in the *type* of their first constant, which is the case the engine
// resolves by type rather than by arity.
class SmartFormat : public vgi::ScalarFunction {
public:
    enum class Shape { Width, Prefix };

    explicit SmartFormat(Shape shape) : shape_(shape) {}

    std::string name() const override { return "smart_format"; }

    vgi::FunctionMetadata metadata() const override {
        return format_metadata("Smart-format a value");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {shape_ == Shape::Width
                    ? vgi::ArgSpec::constant_arg("width", 0, "int64", "Field width")
                    : vgi::ArgSpec::constant_arg("prefix", 0, "varchar", "Prefix"),
                vgi::ArgSpec::any_column("value", 1, "Value")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto width = shape_ == Shape::Width
                               ? static_cast<size_t>(std::max<int64_t>(
                                     0, params.arguments.const_int64(0).value_or(0)))
                               : 0;
        const std::string prefix =
            shape_ == Shape::Prefix ? params.arguments.const_string(0).value_or("") : "";

        auto values = numeric_column(batch);
        arrow::StringBuilder formatted;
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            if (!values || values->IsNull(i)) {
                (void)formatted.AppendNull();
                continue;
            }
            auto rendered = python_float(values->Value(i));
            if (shape_ == Shape::Width) {
                if (rendered.size() < width) {
                    rendered.insert(rendered.begin(), width - rendered.size(), ' ');
                }
            } else {
                rendered = prefix + rendered;
            }
            (void)formatted.Append(rendered);
        }
        std::shared_ptr<arrow::Array> array;
        (void)formatted.Finish(&array);
        return result(params, array);
    }

private:
    Shape shape_;
};

}  // namespace

void register_format_fixtures(vgi::Worker& worker) {
    worker.register_scalar(std::make_shared<FormatNumber>(FormatNumber::Shape::Default));
    worker.register_scalar(std::make_shared<FormatNumber>(FormatNumber::Shape::Precision));
    worker.register_scalar(std::make_shared<FormatNumber>(FormatNumber::Shape::Prefixed));
    worker.register_scalar(std::make_shared<SmartFormat>(SmartFormat::Shape::Width));
    worker.register_scalar(std::make_shared<SmartFormat>(SmartFormat::Shape::Prefix));
}

}  // namespace example
