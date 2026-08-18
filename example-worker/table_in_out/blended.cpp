// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Blended functions: table-in-out functions whose input rows come from their
// *arguments* rather than from a table-valued argument.
//
// That is what `input_from_args` declares, and it is what makes the same
// registration serve both `geo_encode(52, 13)` as a plain scan and
// `FROM t, geo_encode(t.lat, t.lon)` correlated against a table. Because the
// arguments are real value types rather than a TABLE, DuckDB also permits
// several arities under one name.

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

vgi::FunctionMetadata blended_metadata(std::string description,
                                       std::vector<std::string> categories) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = std::move(categories);
    md.input_from_args = true;
    return md;
}

// Render a double the way Python's `str(float)` does over this fixture's
// range: an integral value keeps its trailing `.0` (`52.0`, not `52`).
//
// The tests pin the exact strings the canonical Python fixture emits, so this
// is a compatibility requirement rather than a formatting preference.
std::string render_double(double value) {
    if (value == std::floor(value) && std::isfinite(value)) {
        std::ostringstream out;
        out << static_cast<long long>(value) << ".0";
        return out.str();
    }
    std::ostringstream out;
    out.precision(17);
    out << value;
    std::string text = out.str();
    // Trim to the shortest form that still round-trips, which is what Python
    // prints; 17 digits is the safe upper bound, not the desired output.
    for (int digits = 1; digits < 17; ++digits) {
        std::ostringstream candidate;
        candidate.precision(digits);
        candidate << value;
        if (std::stod(candidate.str()) == value) return candidate.str();
    }
    return text;
}

double round_to(double value, int64_t places) {
    const double scale = std::pow(10.0, static_cast<double>(places));
    return std::round(value * scale) / scale;
}

// `geo_encode(latitude, longitude[, altitude])` — a per-row encoder, in two
// arities under one name.
class GeoEncode : public vgi::TableInOutFunction {
public:
    explicit GeoEncode(bool with_altitude) : with_altitude_(with_altitude) {}

    std::string name() const override { return "geo_encode"; }

    vgi::FunctionMetadata metadata() const override {
        return blended_metadata(with_altitude_
                                    ? "Blended per-row geo encoder (lat, lon, alt -> geohash)"
                                    : "Blended per-row geo encoder (lat, lon -> geohash)",
                                {"geo", "blended"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        std::vector<vgi::ArgSpec> specs{
            vgi::ArgSpec::column("latitude", 0, "double", "Latitude input column"),
            vgi::ArgSpec::column("longitude", 1, "double", "Longitude input column")};
        if (with_altitude_) {
            specs.push_back(
                vgi::ArgSpec::column("altitude", 2, "double", "Altitude input column"));
        }
        specs.push_back(vgi::ArgSpec::named("precision", "int64", "Rounding precision"));
        return specs;
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("geohash", arrow::utf8(), /*nullable=*/true)});
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t precision = params.arguments.named_int64("precision").value_or(4);

        std::vector<std::shared_ptr<arrow::DoubleArray>> inputs;
        static const char* kNames[] = {"latitude", "longitude", "altitude"};
        const int wanted = with_altitude_ ? 3 : 2;
        for (int i = 0; i < wanted; ++i) {
            // By name where the engine supplies it, by position otherwise: a
            // correlated call names the columns, a literal scan does not.
            auto column = batch->GetColumnByName(kNames[i]);
            if (!column && i < batch->num_columns()) column = batch->column(i);
            if (!column) throw std::runtime_error(std::string("geo_encode: no '") + kNames[i] +
                                                  "' column");
            inputs.push_back(std::static_pointer_cast<arrow::DoubleArray>(
                cast_to(column, arrow::float64())));
        }

        arrow::StringBuilder out;
        (void)out.Reserve(batch->num_rows());
        for (int64_t row = 0; row < batch->num_rows(); ++row) {
            bool any_null = false;
            std::string code;
            for (const auto& input : inputs) {
                if (input->IsNull(row)) {
                    any_null = true;
                    break;
                }
                if (!code.empty()) code += ":";
                code += render_double(round_to(input->Value(row), precision));
            }
            if (any_null) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(code);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return {arrow::RecordBatch::Make(params.output_schema, array->length(), {array})};
    }

private:
    bool with_altitude_;
};

// `blended_explode(n)` — emits `0..n-1` per input row, the 1:N blended shape.
class BlendedExplode : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "blended_explode"; }

    vgi::FunctionMetadata metadata() const override {
        return blended_metadata("Blended 1:N fan-out emitting range(n) per input row",
                                {"blended", "test"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("n", 0, "int64", "Rows to emit per input row")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("i", arrow::int64(), /*nullable=*/true)});
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto counts = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(batch->column(0), arrow::int64()));
        arrow::Int64Builder out;
        for (int64_t row = 0; row < counts->length(); ++row) {
            if (counts->IsNull(row)) continue;
            for (int64_t i = 0; i < counts->Value(row); ++i) (void)out.Append(i);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return {arrow::RecordBatch::Make(params.output_schema, array->length(), {array})};
    }
};

}  // namespace

void register_blended(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<GeoEncode>(/*with_altitude=*/false));
    worker.register_table_in_out(std::make_shared<GeoEncode>(/*with_altitude=*/true));
    worker.register_table_in_out(std::make_shared<BlendedExplode>());
}

}  // namespace example
