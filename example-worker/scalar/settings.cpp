// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Fixtures that read DuckDB settings.
//
// A setting only reaches the worker if the function *declares* it in
// `required_settings`. Setting it in SQL is not enough — an undeclared setting
// is silently absent, which is why these fixtures fall back to an identity
// value rather than failing.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// `multiply_by_setting(value)` — multiplies by the `multiplier` setting.
class MultiplyBySetting : public vgi::ScalarFunction {
public:
    std::string name() const override { return "multiply_by_setting"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Multiply the input value by a setting value";
        md.return_type = arrow::int64();
        md.required_settings = {"multiplier"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Integer value to multiply")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        // 1 rather than an error: an unset setting means "no scaling", and a
        // function that refused to run without one would be unusable.
        const int64_t multiplier = params.settings.get_int64("multiplier").value_or(1);
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(0), arrow::int64()));

        arrow::Int64Builder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(values->Value(i) * multiplier);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `scale_by_setting(value)` — the float counterpart.
class ScaleBySetting : public vgi::ScalarFunction {
public:
    std::string name() const override { return "scale_by_setting"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Scale the input value by the float setting `scale_factor`";
        md.return_type = arrow::float64();
        md.required_settings = {"scale_factor"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "float64", "Value to scale")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const double scale = params.settings.get_double("scale_factor").value_or(1.0);
        auto values = std::static_pointer_cast<arrow::DoubleArray>(
            cast_to(batch->column(0), arrow::float64()));

        arrow::DoubleBuilder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(values->Value(i) * scale);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

}  // namespace

void register_settings_fixtures(vgi::Worker& worker) {
    worker.register_scalar(std::make_shared<MultiplyBySetting>());
    worker.register_scalar(std::make_shared<ScaleBySetting>());
}

}  // namespace example
