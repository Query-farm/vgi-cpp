// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "arg_schema.h"

#include <charconv>
#include <cmath>
#include <unordered_map>

#include <arrow/util/key_value_metadata.h>

namespace vgi {

std::shared_ptr<arrow::DataType> arg_type_to_arrow(const std::string& name) {
    static const std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> kByName = {
        {"int8", arrow::int8()},       {"int16", arrow::int16()},
        {"int32", arrow::int32()},     {"int64", arrow::int64()},
        {"uint8", arrow::uint8()},     {"uint16", arrow::uint16()},
        {"uint32", arrow::uint32()},   {"uint64", arrow::uint64()},
        {"float32", arrow::float32()}, {"float", arrow::float32()},
        {"float64", arrow::float64()}, {"double", arrow::float64()},
        {"bool", arrow::boolean()},    {"boolean", arrow::boolean()},
        {"varchar", arrow::utf8()},    {"string", arrow::utf8()},
        {"utf8", arrow::utf8()},       {"blob", arrow::binary()},
        {"binary", arrow::binary()},
    };
    auto it = kByName.find(name);
    return it == kByName.end() ? arrow::null() : it->second;
}

namespace {

// A whole bound loses its `.0`: the engine renders this text verbatim, and a
// range of `[0.0, 10.0]` on an integer argument reads as a mistake.
std::string format_bound(double value) {
    if (std::isfinite(value) && value == std::floor(value)) {
        return std::to_string(static_cast<int64_t>(value));
    }
    char digits[32];
    const auto end = std::to_chars(digits, digits + sizeof(digits), value).ptr;
    return std::string(digits, end);
}

// Interval notation from the four bounds: square brackets for the inclusive
// ones, parentheses for the exclusive, and `-inf`/`+inf` for an open side.
std::optional<std::string> format_range(const ArgSpec& spec) {
    if (!spec.ge && !spec.le && !spec.gt && !spec.lt) return std::nullopt;
    const std::string low = spec.gt   ? "(" + format_bound(*spec.gt)
                            : spec.ge ? "[" + format_bound(*spec.ge)
                                      : "(-inf";
    const std::string high = spec.lt   ? format_bound(*spec.lt) + ")"
                             : spec.le ? format_bound(*spec.le) + "]"
                                       : "+inf)";
    return low + ", " + high;
}

}  // namespace

std::shared_ptr<arrow::Schema> build_arg_schema(const std::vector<ArgSpec>& specs) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(specs.size());

    for (const auto& spec : specs) {
        auto type = spec.arrow_type ? spec.arrow_type : arg_type_to_arrow(spec.type);
        std::vector<std::string> keys;
        std::vector<std::string> values;

        if (!spec.index.has_value()) {
            keys.push_back("vgi_arg");
            values.push_back("named");
        }

        // A polymorphic parameter is marked, not typed. Only when the caller
        // gave no concrete Arrow type: `column_typed`-style declarations must
        // keep their real type so DuckDB can tell overloads apart.
        const bool untyped = !spec.arrow_type && (spec.type == "any" || spec.type.empty());
        if (!spec.constant) {
            if (spec.type == "table") {
                keys.push_back("vgi_type");
                values.push_back("table");
            } else if (untyped) {
                keys.push_back("vgi_type");
                values.push_back("any");
                type = arrow::null();
            }
        } else if (!spec.arrow_type &&
                   (spec.type == "struct" || spec.type == "any" || spec.type.empty())) {
            keys.push_back("vgi_type");
            values.push_back("any");
            type = arrow::null();
        }

        if (spec.constant) {
            keys.push_back("vgi_const");
            values.push_back("true");
        }
        if (spec.varargs) {
            keys.push_back("vgi_varargs");
            values.push_back("true");
        }
        // Presence-only: an empty doc is an absent key, not an empty value.
        if (!spec.description.empty()) {
            keys.push_back("vgi_doc");
            values.push_back(spec.description);
        }
        // Constraint metadata, presence-only in the same way.
        if (auto range = format_range(spec)) {
            keys.push_back("vgi_range");
            values.push_back(*range);
        }
        if (spec.choices) {
            keys.push_back("vgi_choices");
            values.push_back(*spec.choices);
        }
        if (spec.pattern) {
            keys.push_back("vgi_pattern");
            values.push_back(*spec.pattern);
        }
        if (spec.default_value) {
            keys.push_back("vgi_default");
            values.push_back(*spec.default_value);
        }

        auto field = arrow::field(spec.name, type, /*nullable=*/true);
        if (!keys.empty()) {
            field = field->WithMetadata(arrow::key_value_metadata(keys, values));
        }
        fields.push_back(std::move(field));
    }

    return arrow::schema(std::move(fields));
}

std::shared_ptr<arrow::Schema> build_scalar_output_schema(
    const std::shared_ptr<arrow::DataType>& return_type) {
    if (return_type) {
        return arrow::schema({arrow::field("result", return_type, /*nullable=*/true)});
    }
    // Null-typed and *not* nullable, with the marker: a nullable null column
    // reads as "this function returns SQL NULL", which is a different claim
    // from "the type is decided at bind".
    auto field = arrow::field("result", arrow::null(), /*nullable=*/false)
                     ->WithMetadata(arrow::key_value_metadata({"vgi:any"}, {"true"}));
    return arrow::schema({field});
}

}  // namespace vgi
