// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/pushdown.h"

#include <algorithm>

#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>
#include <nlohmann/json.hpp>

#include "wire.h"

namespace vgi {

// The filter tree as it arrives. Kept out of the header: the public `Filter`
// is a flattened view, and callers should not have to walk an AST to answer
// "is this column constrained".
struct PushdownFilters::Spec {
    std::string kind;
    std::string column_name;
    std::string op;
    std::optional<size_t> value_ref;
    std::vector<std::shared_ptr<Spec>> children;
};

using Spec = PushdownFilters::Spec;

namespace {

using nlohmann::json;

std::shared_ptr<Spec> parse_spec(const json& node) {
    auto spec = std::make_shared<Spec>();
    spec->kind = node.value("type", "");
    spec->column_name = node.value("column_name", "");
    if (node.contains("op") && node["op"].is_string()) spec->op = node["op"].get<std::string>();
    if (node.contains("value_ref") && node["value_ref"].is_number_unsigned()) {
        spec->value_ref = node["value_ref"].get<size_t>();
    }
    if (node.contains("children") && node["children"].is_array()) {
        for (const auto& child : node["children"]) spec->children.push_back(parse_spec(child));
    }
    if (node.contains("child_filter") && node["child_filter"].is_object()) {
        spec->children.push_back(parse_spec(node["child_filter"]));
    }
    return spec;
}

Filter::Kind kind_of(const std::string& kind) {
    if (kind == "constant") return Filter::Kind::Constant;
    if (kind == "in") return Filter::Kind::In;
    if (kind == "is_null") return Filter::Kind::IsNull;
    if (kind == "is_not_null") return Filter::Kind::IsNotNull;
    if (kind == "join_keys") return Filter::Kind::JoinKeys;
    return Filter::Kind::Other;
}

void flatten(const std::shared_ptr<Spec>& spec, std::vector<Filter>& out) {
    if (spec->kind == "and" || spec->kind == "or") {
        for (const auto& child : spec->children) flatten(child, out);
        return;
    }
    out.push_back({kind_of(spec->kind), spec->column_name, spec->op});
}

// The comparison this op denotes, in Arrow's vocabulary. The protocol has used
// several spellings for the same operator over time, so all are accepted.
const char* comparison_kernel(const std::string& op) {
    if (op == "eq" || op == "=" || op == "==") return "equal";
    if (op == "ne" || op == "!=" || op == "<>") return "not_equal";
    if (op == "lt" || op == "<") return "less";
    if (op == "le" || op == "lteq" || op == "<=") return "less_equal";
    if (op == "gt" || op == ">") return "greater";
    if (op == "ge" || op == "gteq" || op == ">=") return "greater_equal";
    return nullptr;
}

std::optional<int64_t> as_int64(const std::shared_ptr<arrow::Array>& array) {
    if (!array || array->length() == 0 || array->IsNull(0)) return std::nullopt;
    auto casted = arrow::compute::Cast(*array, arrow::int64());
    if (!casted.ok()) return std::nullopt;
    return std::static_pointer_cast<arrow::Int64Array>(casted.MoveValueUnsafe())->Value(0);
}

}  // namespace

PushdownFilters PushdownFilters::parse(const std::string& ipc_bytes) {
    PushdownFilters filters;
    if (ipc_bytes.empty()) return filters;

    auto batch = wire::decode_ipc(ipc_bytes);
    if (!batch || batch->num_columns() == 0) return filters;

    auto encoded = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(0));
    if (!encoded || encoded->length() == 0 || encoded->IsNull(0)) return filters;

    json tree;
    try {
        tree = json::parse(encoded->GetString(0));
    } catch (const json::exception&) {
        // A filter blob we cannot read means "no filters", not a failed scan.
        // The engine re-checks every predicate, so ignoring one costs speed
        // and never correctness.
        return filters;
    }
    if (!tree.is_array()) return filters;

    for (const auto& node : tree) filters.specs_.push_back(parse_spec(node));
    // value_ref N is column N + 1; column 0 held the tree.
    for (int i = 1; i < batch->num_columns(); ++i) {
        filters.values_.push_back(batch->column(i));
    }
    for (const auto& spec : filters.specs_) flatten(spec, filters.filters_);
    return filters;
}

std::vector<Filter> PushdownFilters::column_filters(const std::string& column) const {
    std::vector<Filter> found;
    for (const auto& filter : filters_) {
        if (filter.column_name == column) found.push_back(filter);
    }
    return found;
}

ColumnBounds PushdownFilters::column_bounds(const std::string& column) const {
    ColumnBounds bounds;
    std::vector<std::shared_ptr<Spec>> stack = specs_;
    while (!stack.empty()) {
        auto spec = stack.back();
        stack.pop_back();
        if (spec->kind == "and" || spec->kind == "or") {
            stack.insert(stack.end(), spec->children.begin(), spec->children.end());
            continue;
        }
        if (spec->kind != "constant" || spec->column_name != column) continue;
        if (!spec->value_ref || *spec->value_ref >= values_.size()) continue;
        auto value = as_int64(values_[*spec->value_ref]);
        if (!value) continue;

        const auto& op = spec->op;
        if (op == "gt" || op == "ge" || op == "gteq" || op == ">" || op == ">=") {
            bounds.min = bounds.min ? std::min(*bounds.min, *value) : *value;
        } else if (op == "lt" || op == "le" || op == "lteq" || op == "<" || op == "<=") {
            bounds.max = bounds.max ? std::max(*bounds.max, *value) : *value;
        } else {
            bounds.min = *value;
            bounds.max = *value;
        }
    }
    return bounds;
}

std::shared_ptr<arrow::RecordBatch> PushdownFilters::apply(
    const std::shared_ptr<arrow::RecordBatch>& batch) const {
    if (!batch || specs_.empty()) return batch;

    auto surviving = batch;
    std::vector<std::shared_ptr<Spec>> stack = specs_;
    while (!stack.empty()) {
        auto spec = stack.back();
        stack.pop_back();

        // Only conjunctions are decomposed. An `or` cannot be applied one
        // branch at a time without dropping rows the whole disjunction keeps,
        // so it is left to the engine.
        if (spec->kind == "and") {
            stack.insert(stack.end(), spec->children.begin(), spec->children.end());
            continue;
        }

        auto column = surviving->GetColumnByName(spec->column_name);
        if (!column) continue;

        arrow::Result<arrow::Datum> mask;
        if (spec->kind == "is_null") {
            mask = arrow::compute::IsNull(column);
        } else if (spec->kind == "is_not_null") {
            mask = arrow::compute::IsValid(column);
        } else if (spec->kind == "constant" || spec->kind == "in") {
            if (!spec->value_ref || *spec->value_ref >= values_.size()) continue;
            auto value = values_[*spec->value_ref];
            if (spec->kind == "in") {
                arrow::compute::SetLookupOptions options(value);
                mask = arrow::compute::CallFunction("is_in", {column}, &options);
            } else {
                const char* kernel = comparison_kernel(spec->op.empty() ? "eq" : spec->op);
                if (!kernel) continue;
                // The constant is one element; comparing needs it as a scalar
                // so Arrow broadcasts rather than requiring equal lengths.
                if (value->length() == 0) continue;
                auto scalar = value->GetScalar(0);
                if (!scalar.ok()) continue;
                mask = arrow::compute::CallFunction(kernel, {column, scalar.MoveValueUnsafe()});
            }
        } else {
            continue;
        }
        if (!mask.ok()) continue;

        // A null result is neither true nor false; SQL drops those rows, and
        // Filter's default emits them, so nulls are made explicit false first.
        // A null comparison result is neither true nor false; SQL drops those
        // rows, and Filter's default *emits* them, so nulls are resolved to
        // false first. Getting this wrong keeps rows the predicate rejected.
        auto resolved = arrow::compute::CallFunction(
            "fill_null", {mask.MoveValueUnsafe(),
                          arrow::Datum(std::make_shared<arrow::BooleanScalar>(false))});
        if (!resolved.ok()) continue;
        auto filtered = arrow::compute::Filter(surviving, resolved.MoveValueUnsafe());
        if (!filtered.ok()) continue;
        surviving = filtered.MoveValueUnsafe().record_batch();
    }
    return surviving;
}

}  // namespace vgi
