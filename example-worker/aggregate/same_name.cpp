// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Same-name-in-two-schemas aggregate fixtures.
//
// The aggregate member of the family whose other two members are
// scalar/same_name.cpp and table_in_out/echo.cpp. Aggregates are the widest
// surface of the three: every aggregate RPC — update, combine, finalize,
// destructor, the window calls, the streaming ones — resolves through one
// by-name entry point in the worker, so a request that omits the schema runs
// whichever implementation the by-name lookup finds first.
//
// The tag is stamped at finalize while the accumulation happens in update, so
// a call that mis-routes only partway — bound to one implementation and
// finalized by the other — reads as the wrong tag just as an outright
// mis-route does.

#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Identical for both implementations — the collision is the point.
constexpr const char* kFunctionName = "test_same_name_agg";

// Decimal text rather than packed bytes: the state is one small integer and
// the probe is about routing, so a state a human can read in a trace is worth
// more here than a compact one.
int64_t decode_total(const std::string& state) {
    return state.empty() ? 0 : std::strtoll(state.c_str(), nullptr, 10);
}

class SameNameAgg : public vgi::AggregateFunction {
public:
    explicit SameNameAgg(std::string schema) : schema_(std::move(schema)) {}

    std::string name() const override { return kFunctionName; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Schema-disambiguation probe; the " + schema_ + "-schema aggregate";
        md.return_type = arrow::utf8();
        // SPECIAL, so a group that folded nothing still finalizes to a tagged
        // zero. NULL would carry no tag, and the tag is the whole assertion.
        md.null_handling = vgi::NullHandling::Special;
        md.examples = {
            {"SELECT example." + schema_ + "." + kFunctionName + "(n) FROM range(3) t(n)",
             "Returns '" + schema_ + ":3'", schema_ + ":3"}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Integer value to accumulate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("result", arrow::utf8(), /*nullable=*/true)});
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(columns[0], arrow::int64()));
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            state = std::to_string(decode_total(state) + values->Value(i));
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        return std::to_string(decode_total(target) + decode_total(source));
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        arrow::StringBuilder out;
        (void)out.Reserve(static_cast<int64_t>(states.size()));
        for (const auto& state : states) {
            (void)out.Append(schema_ + ":" +
                             std::to_string(state ? decode_total(*state) : 0));
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }

private:
    std::string schema_;
};

}  // namespace

void register_same_name_aggregates(vgi::Worker& worker) {
    // The primary catalog, not a hardcoded name: one binary stands in for
    // several fixtures, and these belong to whichever it is serving.
    const auto& primary = worker.catalog().name;
    worker.register_aggregate_in(primary, "main", std::make_shared<SameNameAgg>("main"));
    worker.register_aggregate_in(primary, "data", std::make_shared<SameNameAgg>("data"));
}

}  // namespace example
