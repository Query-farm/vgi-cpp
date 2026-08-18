// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Result-type rules for arithmetic over Arrow types.
//
// A function like `double(x)` or `add_values(a, b)` cannot declare a fixed
// return type: it has to answer at bind with a type wide enough to hold the
// result. These rules match the canonical Python implementation, which derives
// them from pyarrow's own promotion, so a C++ worker and a Python worker
// answer the same SQL query with the same column type.

#pragma once

#include <memory>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/function.h"

namespace vgi {

bool is_integer_type(const arrow::DataType& type);
bool is_floating_type(const arrow::DataType& type);
bool is_decimal_type(const arrow::DataType& type);
bool is_temporal_type(const arrow::DataType& type);

// `_is_addable_type`: integer, floating, decimal, or temporal.
bool is_addable_type(const arrow::DataType& type);
// `_is_multipliable_type`: as addable, but temporal is excluded — multiplying
// a timestamp is not meaningful.
bool is_multipliable_type(const arrow::DataType& type);

// One argument's worth of headroom: the type that can hold `type` + `type`
// without overflowing. Temporal types are returned unchanged, since a date
// plus a date is still a date's width.
std::shared_ptr<arrow::DataType> promote_for_addition(const std::shared_ptr<arrow::DataType>& type);

// The common numeric type of `a` and `b`, then promoted for headroom.
std::shared_ptr<arrow::DataType> common_type_for_addition(
    const std::shared_ptr<arrow::DataType>& a, const std::shared_ptr<arrow::DataType>& b);

// x + x over the first input column, in the bound output type.
//
// Addition rather than multiplication by two on purpose: for decimals the two
// differ in how they overflow, and the canonical worker adds.
std::shared_ptr<arrow::RecordBatch> double_first(const ProcessParams& params,
                                                 const std::shared_ptr<arrow::RecordBatch>& batch);

// a + b over the first two input columns, in the bound output type.
std::shared_ptr<arrow::RecordBatch> add_two(const ProcessParams& params,
                                            const std::shared_ptr<arrow::RecordBatch>& batch);

namespace bounds {

// The two bounds the canonical fixtures use, named as Python spells them so a
// bind error reads the same across implementations.
inline const char* kAddableName = "_is_addable_type";
inline const char* kMultipliableName = "_is_multipliable_type";

inline ArgSpec::TypeBound addable() {
    return {kAddableName, [](const arrow::DataType& t) { return is_addable_type(t); }};
}
inline ArgSpec::TypeBound multipliable() {
    return {kMultipliableName, [](const arrow::DataType& t) { return is_multipliable_type(t); }};
}

}  // namespace bounds

}  // namespace vgi
