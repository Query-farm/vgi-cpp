// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Blended functions: table-in-out functions whose input rows come from their
// *arguments* rather than from a table-valued argument.
//
// That is what `input_from_args` declares, and it is what makes the same
// registration serve both `geo_encode(52, 13)` as a plain scan and
// `FROM t, geo_encode(t.lat, t.lon)` correlated against a table. Because the
// arguments are real value types rather than a TABLE, DuckDB also permits
// several arities under one name.

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <arrow/util/base64.h>

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
//
// Two registrations advertise the two signatures, but the worker resolves a
// call by name alone, so whichever of them answers has to serve either shape.
// The arity therefore comes from the batch the call site actually shipped, not
// from the registration that was asked.
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
            specs.push_back(vgi::ArgSpec::column("altitude", 2, "double", "Altitude input column"));
        }
        specs.push_back(vgi::ArgSpec::named("precision", "int64", "Rounding precision"));
        return specs;
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("geohash", arrow::utf8(), /*nullable=*/true)});
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t precision = params.arguments.named_int64("precision").value_or(4);

        std::vector<std::shared_ptr<arrow::DoubleArray>> inputs;
        static const char* kNames[] = {"latitude", "longitude", "altitude"};
        const int wanted = std::min(3, batch->num_columns());
        if (wanted < 2) throw std::runtime_error("geo_encode: needs latitude and longitude");
        for (int i = 0; i < wanted; ++i) {
            // By name where the engine supplies it, by position otherwise: a
            // correlated call names the columns, a literal scan does not.
            auto column = batch->GetColumnByName(kNames[i]);
            if (!column && i < batch->num_columns()) column = batch->column(i);
            if (!column)
                throw std::runtime_error(std::string("geo_encode: no '") + kNames[i] + "' column");
            inputs.push_back(
                std::static_pointer_cast<arrow::DoubleArray>(cast_to(column, arrow::float64())));
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

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto counts =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(0), arrow::int64()));
        arrow::Int64Builder out;
        // Which input row each output row came from. Without it the engine has
        // no way to stamp the correlated columns onto a fan-out, and refuses
        // the batch rather than guess an identity map that cannot hold.
        std::vector<int32_t> parents;
        for (int64_t row = 0; row < counts->length(); ++row) {
            if (counts->IsNull(row)) continue;
            for (int64_t i = 0; i < counts->Value(row); ++i) {
                (void)out.Append(i);
                parents.push_back(static_cast<int32_t>(row));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);

        vgi::EmittedBatch emitted{
            arrow::RecordBatch::Make(params.output_schema, array->length(), {array})};
        emitted.parent_rows = std::move(parents);
        return {std::move(emitted)};
    }
};

// `row_sum(v1, v2, …[, absolute := false])` — a varargs per-row sum.
//
// A varargs parameter names no columns, so the inputs are read positionally —
// the one blended shape where a name lookup has nothing to find. A row with any
// NULL among its columns sums to NULL.
class RowSum : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "row_sum"; }

    vgi::FunctionMetadata metadata() const override {
        return blended_metadata("Blended per-row varargs sum", {"numeric", "blended"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto values = vgi::ArgSpec::column("values", 0, "double", "Numeric input columns");
        values.with_varargs();
        auto absolute = vgi::ArgSpec::named("absolute", "boolean", "Sum absolute values");
        absolute.default_value = "false";
        return {std::move(values), std::move(absolute)};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("row_sum", arrow::float64(), /*nullable=*/true)});
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        bool absolute = false;
        if (auto flag = params.arguments.named("absolute")) {
            const auto& flags =
                static_cast<const arrow::BooleanArray&>(*cast_to(flag, arrow::boolean()));
            absolute = flags.length() > 0 && !flags.IsNull(0) && flags.Value(0);
        }

        std::vector<double> totals(static_cast<size_t>(batch->num_rows()), 0.0);
        std::vector<bool> valid(static_cast<size_t>(batch->num_rows()), true);
        for (int column = 0; column < batch->num_columns(); ++column) {
            const auto& values = static_cast<const arrow::DoubleArray&>(
                *cast_to(batch->column(column), arrow::float64()));
            for (int64_t row = 0; row < batch->num_rows(); ++row) {
                const auto index = static_cast<size_t>(row);
                if (values.IsNull(row)) {
                    valid[index] = false;
                    continue;
                }
                const double value = values.Value(row);
                totals[index] += absolute ? std::fabs(value) : value;
            }
        }

        arrow::DoubleBuilder out;
        (void)out.Reserve(batch->num_rows());
        for (size_t row = 0; row < totals.size(); ++row) {
            if (valid[row]) {
                (void)out.Append(totals[row]);
            } else {
                (void)out.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return {arrow::RecordBatch::Make(params.output_schema, array->length(), {array})};
    }
};

// `blended_drop(x)` — emits a 0-row batch for its input row.
//
// The 1->0 end of the blended range: the literal call shape has to reach true
// end-of-stream on an empty-but-not-final batch rather than re-feeding its
// synthesized input row forever.
class BlendedDrop : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "blended_drop"; }

    vgi::FunctionMetadata metadata() const override {
        return blended_metadata(
            "Blended 1->0 map emitting a single 0-row batch (literal scan-mode)",
            {"blended", "test"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("x", 0, "double", "Input column (ignored)")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("v", arrow::int64(), /*nullable=*/true)});
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>&) const override {
        return {arrow::RecordBatch::MakeEmpty(params.output_schema).ValueOrDie()};
    }
};

// `projectable_blended(x)` — a 1->1 map with two output columns that opts into
// projection pushdown.
//
// The pair matters: with one output column a narrowing bug is invisible, and
// without pushdown there is nothing to narrow. Building against the bound
// schema rather than emitting both columns is what proves the worker read the
// projection instead of relying on the engine to slice column 0.
class ProjectableBlended : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "projectable_blended"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = blended_metadata("Blended 1->1 map with projection_pushdown + two output columns",
                                   {"blended", "test"});
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("x", 0, "int64", "Input column")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("a", arrow::int64(), /*nullable=*/true),
                              arrow::field("b", arrow::int64(), /*nullable=*/true)});
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto& values =
            static_cast<const arrow::Int64Array&>(*cast_to(batch->column(0), arrow::int64()));

        std::vector<std::shared_ptr<arrow::Array>> columns;
        for (int i = 0; i < params.output_schema->num_fields(); ++i) {
            const auto& name = params.output_schema->field(i)->name();
            const int64_t factor = name == "b" ? 100 : 10;
            arrow::Int64Builder out;
            (void)out.Reserve(values.length());
            for (int64_t row = 0; row < values.length(); ++row) {
                if (values.IsNull(row)) {
                    (void)out.AppendNull();
                } else {
                    (void)out.Append(values.Value(row) * factor);
                }
            }
            std::shared_ptr<arrow::Array> array;
            (void)out.Finish(&array);
            columns.push_back(std::move(array));
        }
        return {arrow::RecordBatch::Make(params.output_schema, values.length(), columns)};
    }
};

}  // namespace

// `hostile_provenance(n, mode := …)` — provenance the engine must refuse.
//
// The worker is a semi-trusted party: the parent indices it sends are used to
// index the input batch, so a malformed payload has to be caught rather than
// followed. Each mode poisons the payload differently, and every one of them
// must throw on both transports.
class HostileProvenance : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "hostile_provenance"; }

    vgi::FunctionMetadata metadata() const override {
        return blended_metadata("Blended map emitting deliberately malformed provenance",
                                {"testing", "blended"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("n", 0, "int64", "Input value, echoed"),
                vgi::ArgSpec::named("mode", "varchar",
                                    "Which malformation to emit: range, length or base64")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("hv", arrow::int64(), /*nullable=*/true)});
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto rows = batch->num_rows();
        arrow::Int64Builder out;
        (void)out.Reserve(rows);
        for (int64_t i = 0; i < rows; ++i) (void)out.Append(i);
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);

        vgi::EmittedBatch emitted{arrow::RecordBatch::Make(params.output_schema, rows, {array})};

        // Written as raw metadata, not through `parent_rows`: the typed field
        // cannot express any of these, which is the point of it.
        const auto mode = params.arguments.named_string("mode").value_or("range");
        std::vector<int32_t> parents;
        if (mode == "base64") {
            emitted.metadata[kParentRowKey] = "!!!not base64!!!";
            return {std::move(emitted)};
        }
        for (int64_t i = 0; i < rows; ++i) {
            // One past the last valid index, which a range check must catch.
            parents.push_back(static_cast<int32_t>(mode == "range" ? rows : i));
        }
        // One int32 too many, which a length check must catch.
        if (mode == "length") parents.push_back(0);
        emitted.metadata[kParentRowKey] = encode_parent_rows(parents);
        return {std::move(emitted)};
    }

private:
    static constexpr const char* kParentRowKey = "vgi_rpc.parent_row#b64";

    // The same little-endian packing the SDK does for `parent_rows`, spelled
    // out here because this fixture has to produce payloads the SDK would
    // refuse to build.
    static std::string encode_parent_rows(const std::vector<int32_t>& parents) {
        std::string raw(parents.size() * sizeof(int32_t), '\0');
        for (size_t i = 0; i < parents.size(); ++i) {
            const auto value = static_cast<uint32_t>(parents[i]);
            raw[i * 4 + 0] = static_cast<char>(value & 0xFF);
            raw[i * 4 + 1] = static_cast<char>((value >> 8) & 0xFF);
            raw[i * 4 + 2] = static_cast<char>((value >> 16) & 0xFF);
            raw[i * 4 + 3] = static_cast<char>((value >> 24) & 0xFF);
        }
        return arrow::util::base64_encode(raw);
    }
};

void register_blended(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<GeoEncode>(/*with_altitude=*/false));
    worker.register_table_in_out(std::make_shared<GeoEncode>(/*with_altitude=*/true));
    worker.register_table_in_out(std::make_shared<BlendedExplode>());
    worker.register_table_in_out(std::make_shared<RowSum>());
    worker.register_table_in_out(std::make_shared<BlendedDrop>());
    worker.register_table_in_out(std::make_shared<ProjectableBlended>());
    worker.register_table_in_out(std::make_shared<HostileProvenance>());
}

}  // namespace example
