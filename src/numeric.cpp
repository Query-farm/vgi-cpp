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
        // Precision is *not* the max of the two: it has to hold the widest
        // integer part alongside the widest fractional one, so it is
        // max(p - s) + max(s). `DECIMAL(10,2)` and `DECIMAL(4,3)` need
        // DECIMAL(11,3), and max(p) would give DECIMAL(10,3) — one digit short
        // of representing the first type's own values.
        const int32_t scale = std::max(da.scale(), db.scale());
        const int32_t integer_digits =
            std::max(da.precision() - da.scale(), db.precision() - db.scale());
        return arrow::decimal128(std::min(integer_digits + scale, 38), scale);
    }

    const auto ia = int_info(*a);
    const auto ib = int_info(*b);
    if (ia && ib) {
        // Same signedness: the wider of the two.
        if (ia->second == ib->second) return ia->first >= ib->first ? a : b;
        // Mixed: a signed type wide enough for both. int64 covers every pair
        // here except one — uint64 does not fit, and there is no int128 to
        // promote to, so that pair goes to float64 and loses precision above
        // 2^53 rather than silently wrapping.
        const auto& unsigned_side = ia->second ? *ib : *ia;
        return unsigned_side.first >= 64 ? arrow::float64() : arrow::int64();
    }

    // Two *different* temporal types have no common numeric form: adding a
    // DATE to a TIMESTAMP is a category error, and answering int64 hands back
    // the sum of two epoch counts in different units as a plain integer.
    if (!ia && !ib) {
        throw std::invalid_argument("no common numeric type for " + a->ToString() + " and " +
                                    b->ToString());
    }
    // One side is an integer and the other is not — a decimal, most likely.
    // Widen to the non-integer side rather than dropping its scale.
    return ia ? b : a;
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

// These helpers each produce exactly one column, so the schema they are
// answering against has to have exactly one field. A wider one is not a
// harmless mismatch: `RecordBatch::Make` validates nothing, `num_columns()`
// then reports the schema's field count while the array vector is shorter, and
// iterating the columns reads off the end of it.
std::shared_ptr<arrow::DataType> bound_output_type(const ProcessParams& params,
                                                   const char* what) {
    if (!params.output_schema || params.output_schema->num_fields() != 1) {
        throw std::runtime_error(
            std::string(what) + ": expects a one-column output schema, got " +
            (params.output_schema ? std::to_string(params.output_schema->num_fields()) +
                                        " columns"
                                  : "none"));
    }
    return params.output_schema->field(0)->type();
}

// Likewise for the input: `RecordBatch::column` does not bounds-check.
//
// Returned by value, because it does not hand back a reference into the batch
// — the `shared_ptr` is materialized per call, and a reference to it dangles.
std::shared_ptr<arrow::Array> input_column(const std::shared_ptr<arrow::RecordBatch>& batch,
                                           int index, const char* what) {
    if (!batch || index >= batch->num_columns()) {
        throw std::runtime_error(std::string(what) + ": expects at least " +
                                 std::to_string(index + 1) + " input column(s), got " +
                                 std::to_string(batch ? batch->num_columns() : 0));
    }
    return batch->column(index);
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
    // int64 for an unresolved type, and deliberately: a scalar function's
    // `bind` runs before the engine has typed every argument, so a null here
    // is the ordinary "not yet" and not an error. Refusing it fails the bind of
    // any function that advertises a return type from its first argument.
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
    const auto type = bound_output_type(params, "double_first");
    const auto first = input_column(batch, 0, "double_first");

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
        auto widened = cast_or_throw(first, wide);
        auto summed = add_arrays(widened, widened, "double");
        // Checked, so an overflowing sum raises rather than wrapping. Arrow
        // C++ spells this the opposite way from arrow-rs, where `safe: false`
        // is the *checking* mode: here `safe` true is checked, and passing
        // false silently produced a wrong answer for 5e37 doubled.
        auto narrowed = cast_or_throw(summed, type, /*safe=*/true);
        return arrow::RecordBatch::Make(params.output_schema, narrowed->length(), {narrowed});
    }

    auto column = cast_or_throw(first, type);
    auto summed = cast_or_throw(add_arrays(column, column, "double"), type);
    return arrow::RecordBatch::Make(params.output_schema, summed->length(), {summed});
}

std::shared_ptr<arrow::RecordBatch> add_two(
    const ProcessParams& params, const std::shared_ptr<arrow::RecordBatch>& batch) {
    const auto type = bound_output_type(params, "add_two");
    auto a = cast_or_throw(input_column(batch, 0, "add_two"), type);
    auto b = cast_or_throw(input_column(batch, 1, "add_two"), type);
    auto summed = cast_or_throw(add_arrays(a, b, "add"), type);
    return arrow::RecordBatch::Make(params.output_schema, summed->length(), {summed});
}

}  // namespace vgi
