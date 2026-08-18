// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/statistics.h"

#include <algorithm>
#include <memory>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/builder_union.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "wire.h"

namespace vgi {

StatValue StatValue::integer(int64_t value) {
    StatValue stat;
    stat.kind_ = Kind::Int64;
    stat.integer_ = value;
    return stat;
}

StatValue StatValue::floating(double value) {
    StatValue stat;
    stat.kind_ = Kind::Float64;
    stat.floating_ = value;
    return stat;
}

StatValue StatValue::text(std::string value) {
    StatValue stat;
    stat.kind_ = Kind::Utf8;
    stat.string_ = std::move(value);
    return stat;
}

StatValue StatValue::bytes(std::string value) {
    StatValue stat;
    stat.kind_ = Kind::Binary;
    stat.string_ = std::move(value);
    return stat;
}

namespace {

std::shared_ptr<arrow::DataType> type_of(StatValue::Kind kind) {
    switch (kind) {
        case StatValue::Kind::Float64: return arrow::float64();
        case StatValue::Kind::Utf8: return arrow::utf8();
        case StatValue::Kind::Binary: return arrow::binary();
        case StatValue::Kind::Int64: break;
    }
    return arrow::int64();
}

// The one member of a sparse union child, or a null where the row's value is
// of some other type. Sparse means every child is as long as the union itself,
// so the nulls are not optional padding — they are the layout.
std::shared_ptr<arrow::Array> build_child(const std::vector<ColumnStatistics>& statistics,
                                          bool minimum, StatValue::Kind kind) {
    const auto value_of = [&](const ColumnStatistics& stat) -> const StatValue& {
        return minimum ? stat.min : stat.max;
    };

    std::shared_ptr<arrow::Array> array;
    switch (kind) {
        case StatValue::Kind::Float64: {
            arrow::DoubleBuilder builder;
            for (const auto& stat : statistics) {
                const auto& value = value_of(stat);
                if (value.kind() == kind) {
                    (void)builder.Append(value.floating_value());
                } else {
                    (void)builder.AppendNull();
                }
            }
            (void)builder.Finish(&array);
            break;
        }
        case StatValue::Kind::Utf8: {
            arrow::StringBuilder builder;
            for (const auto& stat : statistics) {
                const auto& value = value_of(stat);
                if (value.kind() == kind) {
                    (void)builder.Append(value.string_value());
                } else {
                    (void)builder.AppendNull();
                }
            }
            (void)builder.Finish(&array);
            break;
        }
        case StatValue::Kind::Binary: {
            arrow::BinaryBuilder builder;
            for (const auto& stat : statistics) {
                const auto& value = value_of(stat);
                if (value.kind() == kind) {
                    (void)builder.Append(value.string_value());
                } else {
                    (void)builder.AppendNull();
                }
            }
            (void)builder.Finish(&array);
            break;
        }
        case StatValue::Kind::Int64: {
            arrow::Int64Builder builder;
            for (const auto& stat : statistics) {
                const auto& value = value_of(stat);
                if (value.kind() == kind) {
                    (void)builder.Append(value.integer_value());
                } else {
                    (void)builder.AppendNull();
                }
            }
            (void)builder.Finish(&array);
            break;
        }
    }
    return array;
}

std::shared_ptr<arrow::Array> build_union(const std::vector<ColumnStatistics>& statistics,
                                          bool minimum,
                                          const std::vector<StatValue::Kind>& order) {
    arrow::Int8Builder type_ids;
    (void)type_ids.Reserve(static_cast<int64_t>(statistics.size()));
    for (const auto& stat : statistics) {
        const auto& value = minimum ? stat.min : stat.max;
        const auto found = std::find(order.begin(), order.end(), value.kind());
        (void)type_ids.Append(
            found == order.end() ? 0 : static_cast<int8_t>(found - order.begin()));
    }
    std::shared_ptr<arrow::Array> ids;
    (void)type_ids.Finish(&ids);

    arrow::ArrayVector children;
    std::vector<std::string> names;
    std::vector<int8_t> codes;
    children.reserve(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        children.push_back(build_child(statistics, minimum, order[i]));
        // Named by position, matching the reference implementations: the
        // engine reads the union by type code, never by field name.
        names.push_back(std::to_string(i));
        codes.push_back(static_cast<int8_t>(i));
    }

    auto array = arrow::SparseUnionArray::Make(*ids, children, names, codes);
    if (!array.ok()) throw std::runtime_error("statistics: " + array.status().ToString());
    return array.MoveValueUnsafe();
}

}  // namespace

std::string serialize_column_statistics(const std::vector<ColumnStatistics>& statistics) {
    if (statistics.empty()) return wire::encode_schema(arrow::schema({}));

    // One child per distinct bound type, in first-seen order. `min` and `max`
    // share the layout so the two columns have the same union type, which is
    // what lets the engine decode them with one reader.
    std::vector<StatValue::Kind> order;
    const auto remember = [&](StatValue::Kind kind) {
        if (std::find(order.begin(), order.end(), kind) == order.end()) order.push_back(kind);
    };
    for (const auto& stat : statistics) {
        remember(stat.min.kind());
        remember(stat.max.kind());
    }

    arrow::StringBuilder names;
    arrow::BooleanBuilder has_null;
    arrow::BooleanBuilder has_not_null;
    arrow::Int64Builder distinct_count;
    arrow::BooleanBuilder contains_unicode;
    arrow::UInt64Builder max_string_length;
    for (const auto& stat : statistics) {
        (void)names.Append(stat.column_name);
        (void)has_null.Append(stat.has_null);
        (void)has_not_null.Append(stat.has_not_null);
        if (stat.distinct_count) {
            (void)distinct_count.Append(*stat.distinct_count);
        } else {
            (void)distinct_count.AppendNull();
        }
        if (stat.contains_unicode) {
            (void)contains_unicode.Append(*stat.contains_unicode);
        } else {
            (void)contains_unicode.AppendNull();
        }
        if (stat.max_string_length) {
            (void)max_string_length.Append(*stat.max_string_length);
        } else {
            (void)max_string_length.AppendNull();
        }
    }

    auto minimums = build_union(statistics, /*minimum=*/true, order);
    auto maximums = build_union(statistics, /*minimum=*/false, order);

    std::vector<std::shared_ptr<arrow::Array>> columns(6);
    (void)names.Finish(&columns[0]);
    (void)has_null.Finish(&columns[1]);
    (void)has_not_null.Finish(&columns[2]);
    (void)distinct_count.Finish(&columns[3]);
    (void)contains_unicode.Finish(&columns[4]);
    (void)max_string_length.Finish(&columns[5]);

    auto schema = arrow::schema({
        arrow::field("column_name", arrow::utf8(), true),
        arrow::field("min", minimums->type(), true),
        arrow::field("max", maximums->type(), true),
        arrow::field("has_null", arrow::boolean(), true),
        arrow::field("has_not_null", arrow::boolean(), true),
        arrow::field("distinct_count", arrow::int64(), true),
        arrow::field("contains_unicode", arrow::boolean(), true),
        arrow::field("max_string_length", arrow::uint64(), true),
    });
    auto batch = arrow::RecordBatch::Make(
        schema, static_cast<int64_t>(statistics.size()),
        {columns[0], minimums, maximums, columns[1], columns[2], columns[3], columns[4],
         columns[5]});
    return wire::encode_ipc(batch);
}

}  // namespace vgi
