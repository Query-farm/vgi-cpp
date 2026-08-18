// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// `vgi_percentile(value, percentile)` — the constant-argument aggregate.
//
// The only fixture here whose *result* depends on a value the group states
// cannot carry: the rows are folded long before the percentile is applied, so
// the constant has to reach finalize on its own. That is what
// finalize_with_arguments exists for, and what this fixture exercises.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// The median, matching the Python fixture's default for the same parameter.
constexpr double kDefaultPercentile = 0.5;

// 17 significant digits round-trips an IEEE double exactly. It matters here
// because the state is text on the wire and is re-parsed by whichever process
// finalizes.
std::string render(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

std::vector<double> parse_values(const std::string& state) {
    std::vector<double> values;
    size_t start = 0;
    while (start < state.size()) {
        const size_t end = std::min(state.find(',', start), state.size());
        values.push_back(std::strtod(state.substr(start, end - start).c_str(), nullptr));
        start = end + 1;
    }
    return values;
}

// The folded constant, rejected here rather than where it is used.
//
// Every one of these used to surface from deep inside finalize as a raw
// conversion failure, after the query had already scanned its input; bind is
// the first point at which the value is known and the last at which the
// complaint can still name the function.
double checked_percentile(const vgi::Arguments& arguments) {
    auto supplied = arguments.positional(0);
    // Registration binds with no arguments at all, which is not a call site
    // and must not be rejected.
    if (!supplied || supplied->length() == 0) return kDefaultPercentile;
    if (supplied->IsNull(0)) {
        throw std::invalid_argument("vgi_percentile: percentile must not be NULL");
    }
    auto value = arguments.const_double(0);
    if (!value) {
        throw std::invalid_argument("vgi_percentile: percentile must be a number, got " +
                                    supplied->type()->ToString());
    }
    if (!std::isfinite(*value)) {
        throw std::invalid_argument("vgi_percentile: percentile must be a finite number, got " +
                                    render(*value));
    }
    if (*value < 0.0 || *value > 1.0) {
        throw std::invalid_argument("vgi_percentile: percentile must be in [0, 1], got " +
                                    render(*value));
    }
    return *value;
}

class Percentile : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_percentile"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Approximate percentile (demonstrates a constant argument)";
        md.return_type = arrow::float64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "float64", "Values"),
                vgi::ArgSpec::constant_arg("percentile", 1, "double", "Percentile (0-1)")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        (void)checked_percentile(params.arguments);
        return arrow::schema({arrow::field("result", arrow::float64(), /*nullable=*/true)});
    }

    // The values themselves, comma-joined: a percentile is not decomposable,
    // so no running summary can stand in for them.
    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values =
            std::static_pointer_cast<arrow::DoubleArray>(cast_to(columns[0], arrow::float64()));
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            if (!state.empty()) state += ",";
            state += render(values->Value(i));
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        if (target.empty()) return source;
        if (source.empty()) return target;
        return target + "," + source;
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array& group_ids,
        const std::vector<std::optional<std::string>>& states) const override {
        return finalize_with_arguments(output_schema, group_ids, states, vgi::Arguments{});
    }

    std::shared_ptr<arrow::RecordBatch> finalize_with_arguments(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states,
        const vgi::Arguments& arguments) const override {
        // Already accepted at bind; re-read rather than re-derived, since a
        // finalize may run in a process that never bound.
        const double percentile = arguments.const_double(0).value_or(kDefaultPercentile);

        arrow::DoubleBuilder out;
        (void)out.Reserve(static_cast<int64_t>(states.size()));
        for (const auto& state : states) {
            if (!state || state->empty()) {
                (void)out.AppendNull();
                continue;
            }
            auto values = parse_values(*state);
            std::sort(values.begin(), values.end());
            // Nearest-rank rather than interpolated, clamped so percentile 1.0
            // lands on the last value instead of one past it.
            const auto index =
                std::min(static_cast<size_t>(percentile * static_cast<double>(values.size())),
                         values.size() - 1);
            (void)out.Append(values[index]);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }
};

}  // namespace

void register_percentile(vgi::Worker& worker) {
    worker.register_aggregate(std::make_shared<Percentile>());
}

}  // namespace example
