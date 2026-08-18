// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/numeric.h"

#include <algorithm>
#include <stdexcept>

#include <arrow/compute/api.h>
#include <arrow/compute/cast.h>
#include <optional>
#include <utility>

namespace vgi {
namespace {

using arrow::Type;

// (bit width, signed) for an integer type; nullopt for anything else.
std::optional<std::pair<int, bool>> int_info(const arrow::DataType& type) {
    switch (type.id()) {
        case Type::INT8: return std::pair{8, true};
        case Type::INT16: return std::pair{16, true};
        case Type::INT32: return std::pair{32, true};
        case Type::INT64: return std::pair{64, true};
        case Type::UINT8: return std::pair{8, false};
        case Type::UINT16: return std::pair{16, false};
        case Type::UINT32: return std::pair{32, false};
        case Type::UINT64: return std::pair{64, false};
        default: return std::nullopt;
    }
}

std::shared_ptr<arrow::DataType> common_numeric(const std::shared_ptr<arrow::DataType>& a,
                                                const std::shared_ptr<arrow::DataType>& b) {
    if (a->Equals(*b)) return a;
    if (is_floating_type(*a) || is_floating_type(*b)) return arrow::float64();

    if (a->id() == Type::DECIMAL128 && b->id() == Type::DECIMAL128) {
        const auto& da = static_cast<const arrow::Decimal128Type&>(*a);
        const auto& db = static_cast<const arrow::Decimal128Type&>(*b);
        return arrow::decimal128(std::max(da.precision(), db.precision()),
                                 std::max(da.scale(), db.scale()));
    }

    const auto ia = int_info(*a);
    const auto ib = int_info(*b);
    if (ia && ib) {
        // Same signedness: the wider of the two. Mixed: a signed type wide
        // enough for both, which int64 always is for the widths above.
        if (ia->second == ib->second) return ia->first >= ib->first ? a : b;
        return arrow::int64();
    }
    return arrow::int64();
}

// Cast that reports why, since every caller here would otherwise discard the
// status and produce a null array.
std::shared_ptr<arrow::Array> cast_or_throw(const std::shared_ptr<arrow::Array>& array,
                                            const std::shared_ptr<arrow::DataType>& type,
                                            bool safe = true) {
    if (array->type()->Equals(*type)) return array;
    arrow::compute::CastOptions options(safe);
    options.to_type = type;
    auto result = arrow::compute::Cast(arrow::Datum(array), options);
    if (!result.ok()) {
        throw std::runtime_error("cast " + array->type()->ToString() + " -> " +
                                 type->ToString() + ": " + result.status().message());
    }
    return result.MoveValueUnsafe().make_array();
}

std::shared_ptr<arrow::Array> add_arrays(const std::shared_ptr<arrow::Array>& a,
                                         const std::shared_ptr<arrow::Array>& b,
                                         const char* what) {
    auto sum = arrow::compute::CallFunction("add", {a, b});
    if (!sum.ok()) {
        throw std::runtime_error(std::string(what) + ": " + sum.status().message());
    }
    return sum.MoveValueUnsafe().make_array();
}

std::shared_ptr<arrow::DataType> bound_output_type(const ProcessParams& params) {
    if (!params.output_schema || params.output_schema->num_fields() == 0) {
        throw std::runtime_error("output schema has no fields");
    }
    return params.output_schema->field(0)->type();
}

}  // namespace

bool is_integer_type(const arrow::DataType& type) { return int_info(type).has_value(); }

bool is_floating_type(const arrow::DataType& type) {
    return type.id() == Type::HALF_FLOAT || type.id() == Type::FLOAT ||
           type.id() == Type::DOUBLE;
}

bool is_decimal_type(const arrow::DataType& type) {
    return type.id() == Type::DECIMAL128 || type.id() == Type::DECIMAL256;
}

bool is_temporal_type(const arrow::DataType& type) {
    switch (type.id()) {
        case Type::DATE32:
        case Type::DATE64:
        case Type::TIME32:
        case Type::TIME64:
        case Type::TIMESTAMP:
        case Type::DURATION:
        case Type::INTERVAL_MONTHS:
        case Type::INTERVAL_DAY_TIME:
        case Type::INTERVAL_MONTH_DAY_NANO:
            return true;
        default:
            return false;
    }
}

bool is_addable_type(const arrow::DataType& type) {
    return is_integer_type(type) || is_floating_type(type) || is_decimal_type(type) ||
           is_temporal_type(type);
}

bool is_multipliable_type(const arrow::DataType& type) {
    return is_integer_type(type) || is_floating_type(type) || is_decimal_type(type);
}

std::shared_ptr<arrow::DataType> promote_for_addition(
    const std::shared_ptr<arrow::DataType>& type) {
    if (!type) return arrow::int64();
    if (is_temporal_type(*type)) return type;
    switch (type->id()) {
        case Type::HALF_FLOAT:
        case Type::FLOAT:
        case Type::DOUBLE: return arrow::float64();
        case Type::INT8: return arrow::int16();
        case Type::INT16: return arrow::int32();
        case Type::INT32:
        case Type::INT64: return arrow::int64();
        case Type::UINT8: return arrow::uint16();
        case Type::UINT16: return arrow::uint32();
        case Type::UINT32:
        case Type::UINT64: return arrow::uint64();
        case Type::DECIMAL128: {
            // One more digit of precision holds the carry; 38 is the type's
            // ceiling, so a decimal already at the limit stays there.
            const auto& d = static_cast<const arrow::Decimal128Type&>(*type);
            return arrow::decimal128(std::min(d.precision() + 1, 38), d.scale());
        }
        default: return type;
    }
}

std::shared_ptr<arrow::DataType> common_type_for_addition(
    const std::shared_ptr<arrow::DataType>& a, const std::shared_ptr<arrow::DataType>& b) {
    if (!a) return promote_for_addition(b);
    if (!b) return promote_for_addition(a);
    return promote_for_addition(common_numeric(a, b));
}

std::shared_ptr<arrow::RecordBatch> double_first(
    const ProcessParams& params, const std::shared_ptr<arrow::RecordBatch>& batch) {
    const auto type = bound_output_type(params);

    if (type->id() == Type::DECIMAL128) {
        // Widen, add, then narrow back with checking on. Adding in the capped
        // precision would either wrap or saturate; going wide first means the
        // narrowing cast is what fails, and it fails with the canonical
        // "does not fit in precision N" the fixtures assert on.
        //
        // 75 rather than decimal256's maximum of 76, because Arrow's `add`
        // gives the sum one more digit than its operands: widening to 76 makes
        // the *addition* fail with "precision out of range: 77" before the
        // narrowing cast is ever reached. 75 still clears any decimal128
        // input, whose precision caps at 38.
        const auto& capped = static_cast<const arrow::Decimal128Type&>(*type);
        const auto wide = arrow::decimal256(75, capped.scale());
        auto widened = cast_or_throw(batch->column(0), wide);
        auto summed = add_arrays(widened, widened, "double");
        // Checked, so an overflowing sum raises rather than wrapping. Arrow
        // C++ spells this the opposite way from arrow-rs, where `safe: false`
        // is the *checking* mode: here `safe` true is checked, and passing
        // false silently produced a wrong answer for 5e37 doubled.
        auto narrowed = cast_or_throw(summed, type, /*safe=*/true);
        return arrow::RecordBatch::Make(params.output_schema, narrowed->length(), {narrowed});
    }

    auto column = cast_or_throw(batch->column(0), type);
    auto summed = cast_or_throw(add_arrays(column, column, "double"), type);
    return arrow::RecordBatch::Make(params.output_schema, summed->length(), {summed});
}

std::shared_ptr<arrow::RecordBatch> add_two(
    const ProcessParams& params, const std::shared_ptr<arrow::RecordBatch>& batch) {
    const auto type = bound_output_type(params);
    auto a = cast_or_throw(batch->column(0), type);
    auto b = cast_or_throw(batch->column(1), type);
    auto summed = cast_or_throw(add_arrays(a, b, "add"), type);
    return arrow::RecordBatch::Make(params.output_schema, summed->length(), {summed});
}

}  // namespace vgi
