// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "vgi/pushdown.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>

#include <arrow/array/builder_primitive.h>
#include <arrow/array/concatenate.h>
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
    // A `join_keys` filter names a column in the side batches rather than
    // carrying its values inline.
    std::string keys_column;
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
    spec->keys_column = node.value("keys_column", "");
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

bool is_equality(const std::string& op) {
    return op.empty() || op == "eq" || op == "=" || op == "==";
}

std::optional<int64_t> as_int64(const std::shared_ptr<arrow::Array>& array) {
    if (!array || array->length() == 0 || array->IsNull(0)) return std::nullopt;
    auto casted = arrow::compute::Cast(*array, arrow::int64());
    if (!casted.ok()) return std::nullopt;
    return std::static_pointer_cast<arrow::Int64Array>(casted.MoveValueUnsafe())->Value(0);
}

}  // namespace

PushdownFilters PushdownFilters::parse(const std::string& ipc_bytes,
                                       const std::vector<std::string>& join_key_batches) {
    PushdownFilters filters;
    for (const auto& blob : join_key_batches) {
        auto batch = wire::decode_ipc(blob);
        if (!batch) continue;
        for (int i = 0; i < batch->num_columns(); ++i) {
            // Last batch wins for a repeated column name. Safe under the
            // contract that extra rows are only slower — a scan that misses a
            // key emits more than it had to, and the engine still filters.
            filters.join_keys_[batch->schema()->field(i)->name()] = batch->column(i);
        }
    }
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
    // VGI_FILTER_DEBUG=1 prints the filter tree. Which predicates DuckDB
    // actually pushes is not obvious — an `IN` list arrives as a `join_keys`
    // filter with no inline values at all — and this is the quickest way to
    // find out.
    if (std::getenv("VGI_FILTER_DEBUG")) {
        std::fprintf(stderr, "[vgi-filter] %s\n", encoded->GetString(0).c_str());
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

// The values a spec refers to: inline for most kinds, from the side batches
// for `join_keys`.
std::shared_ptr<arrow::Array> PushdownFilters::values_for(const Spec& spec) const {
    if (spec.kind == "join_keys") {
        auto it = join_keys_.find(spec.keys_column.empty() ? spec.column_name : spec.keys_column);
        return it == join_keys_.end() ? nullptr : it->second;
    }
    if (!spec.value_ref || *spec.value_ref >= values_.size()) return nullptr;
    return values_[*spec.value_ref];
}

std::vector<Filter> PushdownFilters::column_filters(const std::string& column) const {
    std::vector<Filter> found;
    for (const auto& filter : filters_) {
        if (filter.column_name == column) found.push_back(filter);
    }
    return found;
}

bool PushdownFilters::has_filter_for_column(const std::string& column) const {
    return std::any_of(filters_.begin(), filters_.end(),
                       [&](const Filter& filter) { return filter.column_name == column; });
}

std::vector<std::string> PushdownFilters::filtered_columns() const {
    std::vector<std::string> columns;
    for (const auto& filter : filters_) {
        // An `and`/`or` node is flattened away, but a leaf with no column —
        // there is no such thing on the wire today — would name the empty
        // string and read as a column.
        if (!filter.column_name.empty()) columns.push_back(filter.column_name);
    }
    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    return columns;
}

// The widest interval every predicate on `column` agrees a value could be in.
//
// Deliberately loosening rather than tightening: bounds from separate branches
// are combined with min/max, so an OR widens and an AND does not narrow. A
// scan that trusts these emits a superset, and the engine re-checks every
// predicate — where a tightened bound would drop rows the query asked for.
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
        auto value = as_int64(values_for(*spec));
        if (!value) continue;

        const auto& op = spec->op;
        if (op == "gt" || op == "ge" || op == "gteq" || op == ">" || op == ">=") {
            bounds.min = bounds.min ? std::min(*bounds.min, *value) : *value;
        } else if (op == "lt" || op == "le" || op == "lteq" || op == "<" || op == "<=") {
            bounds.max = bounds.max ? std::max(*bounds.max, *value) : *value;
        } else {
            // Equality pins both ends — but only ever outward, or `n = 5 OR
            // n = 9` would come back as the single point whichever branch the
            // stack happened to pop last.
            bounds.min = bounds.min ? std::min(*bounds.min, *value) : *value;
            bounds.max = bounds.max ? std::max(*bounds.max, *value) : *value;
        }
    }
    return bounds;
}

std::vector<std::shared_ptr<Spec>> PushdownFilters::column_specs(const std::string& column) const {
    std::vector<std::shared_ptr<Spec>> found;
    for (const auto& spec : specs_) {
        if (spec->column_name != column) continue;
        if (spec->kind == "and") {
            for (const auto& child : spec->children) {
                if (child->column_name == column) found.push_back(child);
            }
        } else {
            found.push_back(spec);
        }
    }
    return found;
}

std::shared_ptr<arrow::Array> PushdownFilters::or_column_values(const Spec& spec,
                                                                const std::string& column) const {
    arrow::ArrayVector branches;
    for (const auto& child : spec.children) {
        // A branch constraining a different column — or none — leaves this
        // column free to take any value within that branch.
        if (child->column_name != column) return nullptr;
        const bool discrete = child->kind == "in" || child->kind == "join_keys" ||
                              (child->kind == "constant" && is_equality(child->op));
        if (!discrete) return nullptr;
        auto values = values_for(*child);
        if (!values || values->length() == 0) return nullptr;
        branches.push_back(child->kind == "constant" ? values->Slice(0, 1) : values);
    }
    if (branches.empty()) return nullptr;

    auto combined = arrow::Concatenate(branches);
    if (!combined.ok()) return nullptr;
    // Arrow's `unique` keeps first-appearance order, which is what makes the
    // rendered set stable enough for a test to compare.
    auto deduped = arrow::compute::Unique(combined.MoveValueUnsafe());
    if (!deduped.ok()) return nullptr;
    return deduped.MoveValueUnsafe();
}

std::shared_ptr<arrow::Array> PushdownFilters::column_values(const std::string& column) const {
    for (const auto& spec : column_specs(column)) {
        if (spec->kind == "constant" && is_equality(spec->op)) {
            auto values = values_for(*spec);
            // Sliced rather than returned whole: the values column holds one
            // entry per referenced constant, not one per filter.
            if (values && values->length() > 0) return values->Slice(0, 1);
        } else if (spec->kind == "in" || spec->kind == "join_keys") {
            if (auto values = values_for(*spec)) return values;
        } else if (spec->kind == "or") {
            if (auto values = or_column_values(*spec, column)) return values;
        }
    }
    return nullptr;
}

namespace {

const char* op_symbol(const std::string& op) {
    if (op == "eq") return "=";
    if (op == "ne") return "!=";
    if (op == "lt") return "<";
    if (op == "le") return "<=";
    if (op == "gt") return ">";
    if (op == "ge") return ">=";
    return "?";
}

// Rendered as the Python fixtures render it: strings single-quoted, booleans
// `True`/`False`, nulls `NULL`, everything else via its own display. The tests
// compare this text, so the spelling is the contract.
std::string format_scalar(const std::shared_ptr<arrow::Array>& array, int64_t i) {
    if (!array || i >= array->length() || array->IsNull(i)) return "NULL";
    switch (array->type()->id()) {
        case arrow::Type::STRING:
            return "'" + static_cast<const arrow::StringArray&>(*array).GetString(i) + "'";
        case arrow::Type::LARGE_STRING:
            return "'" + static_cast<const arrow::LargeStringArray&>(*array).GetString(i) + "'";
        case arrow::Type::BOOL:
            return static_cast<const arrow::BooleanArray&>(*array).Value(i) ? "True" : "False";
        default: break;
    }
    auto casted = arrow::compute::Cast(*array->Slice(i, 1), arrow::utf8());
    if (!casted.ok()) return {};
    return std::static_pointer_cast<arrow::StringArray>(casted.MoveValueUnsafe())->GetString(0);
}

}  // namespace

std::string PushdownFilters::format() const {
    if (specs_.empty()) return "(none)";

    // Bound to the member so the recursion can reach `values_`; a free
    // function would have to take them as a parameter at every level.
    std::function<std::string(const std::shared_ptr<Spec>&, const std::string&)> render =
        [&](const std::shared_ptr<Spec>& spec, const std::string& column) -> std::string {
        const std::string& name = column.empty() ? spec->column_name : column;
        auto value = [&]() -> std::shared_ptr<arrow::Array> { return values_for(*spec); };

        if (spec->kind == "is_null") return name + " IS NULL";
        if (spec->kind == "is_not_null") return name + " IS NOT NULL";
        if (spec->kind == "constant") {
            return name + " " + op_symbol(spec->op.empty() ? "eq" : spec->op) + " " +
                   format_scalar(value(), 0);
        }
        if (spec->kind == "in" || spec->kind == "join_keys") {
            auto values = value();
            if (!values) return name + " IN ()";
            // A long IN list is collapsed: the point is that a filter arrived,
            // and printing thousands of values makes the output unreadable and
            // the test unwriteable.
            if (values->length() > 20) {
                return name + " IN (" + std::to_string(values->length()) + " values)";
            }
            std::string items;
            for (int64_t i = 0; i < values->length(); ++i) {
                if (i) items += ", ";
                items += format_scalar(values, i);
            }
            return name + " IN (" + items + ")";
        }
        if (spec->kind == "and" || spec->kind == "or") {
            const std::string joiner = spec->kind == "and" ? " AND " : " OR ";
            std::string parts;
            for (size_t i = 0; i < spec->children.size(); ++i) {
                if (i) parts += joiner;
                parts += render(spec->children[i], "");
            }
            // Parenthesized, so a nested group reads unambiguously and the
            // text matches what the reference implementations emit.
            return "(" + parts + ")";
        }
        return spec->kind;
    };

    std::string out;
    for (size_t i = 0; i < specs_.size(); ++i) {
        if (i) out += " AND ";
        out += render(specs_[i], "");
    }
    return out.empty() ? "(none)" : out;
}

std::string PushdownFilters::format_repr() const {
    if (specs_.empty()) return "(none)";

    // Bound to the member for the same reason `format` is: the recursion has
    // to reach `values_`.
    std::function<std::string(const std::shared_ptr<Spec>&, const std::string&)> render =
        [&](const std::shared_ptr<Spec>& spec, const std::string& column) -> std::string {
        const std::string& name = column.empty() ? spec->column_name : column;

        if (spec->kind == "is_null") return "IsNullFilter(" + name + " IS NULL)";
        if (spec->kind == "is_not_null") return "IsNotNullFilter(" + name + " IS NOT NULL)";
        if (spec->kind == "constant") {
            return "ConstantFilter(" + name + " " + op_symbol(spec->op.empty() ? "eq" : spec->op) +
                   " " + format_scalar(values_for(*spec), 0) + ")";
        }
        if (spec->kind == "in" || spec->kind == "join_keys") {
            // Every value, unlike `format`: this rendering names the kinds,
            // and a fixture asserting on it wants the whole set.
            auto values = values_for(*spec);
            std::string items;
            for (int64_t i = 0; values && i < values->length(); ++i) {
                if (i) items += ", ";
                items += format_scalar(values, i);
            }
            return "InFilter(" + name + " IN [" + items + "])";
        }
        if (spec->kind == "and" || spec->kind == "or") {
            const std::string label = spec->kind == "and" ? "AndFilter(" : "OrFilter(";
            const std::string joiner = spec->kind == "and" ? " AND " : " OR ";
            std::string parts;
            for (size_t i = 0; i < spec->children.size(); ++i) {
                if (i) parts += joiner;
                parts += render(spec->children[i], "");
            }
            return label + parts + ")";
        }
        return spec->kind;
    };

    std::string parts;
    for (size_t i = 0; i < specs_.size(); ++i) {
        if (i) parts += ", ";
        parts += render(specs_[i], "");
    }
    return "PushdownFilters([" + parts + "])";
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

        // A dictionary column is decoded before comparing. DuckDB types a
        // dictionary<*, utf8> column without ENUM metadata as plain VARCHAR
        // and pushes a string literal, and every comparison kernel rejects the
        // (dictionary, string) pair; casting the literal *up* to the
        // dictionary type is what throws.
        if (column->type_id() == arrow::Type::DICTIONARY) {
            const auto& value_type =
                static_cast<const arrow::DictionaryType&>(*column->type()).value_type();
            auto decoded = arrow::compute::Cast(*column, value_type);
            if (!decoded.ok()) continue;
            column = *decoded;
        }

        arrow::Result<arrow::Datum> mask;
        if (spec->kind == "is_null") {
            mask = arrow::compute::IsNull(column);
        } else if (spec->kind == "is_not_null") {
            mask = arrow::compute::IsValid(column);
        } else if (spec->kind == "constant" || spec->kind == "in" || spec->kind == "join_keys") {
            auto value = values_for(*spec);
            if (!value) continue;
            if (spec->kind != "constant") {
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
        // DROP, explicitly. A null comparison result is neither true nor
        // false and SQL excludes those rows; `EMIT_NULL` would keep them,
        // which is a wrong answer rather than a slow one. Named rather than
        // left to the default so the choice survives an Arrow upgrade.
        arrow::compute::FilterOptions options(
            arrow::compute::FilterOptions::NullSelectionBehavior::DROP);
        auto filtered = arrow::compute::Filter(surviving, mask.MoveValueUnsafe(), options);
        if (!filtered.ok()) continue;
        if (auto next = filtered.MoveValueUnsafe().record_batch()) surviving = next;
    }
    return surviving;
}

}  // namespace vgi
