// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The dictionary-encoded enum strings the VGI wire carries.
//
// Every one of these is serialized as `dictionary(int16, utf8)`. The values are
// the canonical Python enums' *member names*, and case is load-bearing: the
// DuckDB extension matches several of them exactly and either silently falls
// back to a default (null_handling, which quietly breaks a scalar function that
// needs to see NULLs) or rejects the catalog outright (order_dependence,
// distinct_dependence). Lowercase is not a stylistic choice here.

#pragma once

// `partition_kind` and `order_preservation` are not here: a fixture sets them,
// so they live in the public `vgi/types.h` instead.

namespace vgi::enums {

// The worker's internal function-kind keys (listing cache keys, the
// normalized `catalog_schema_contents_functions` filter). These are NOT what
// `FunctionInfo.function_type` carries on the wire -- see `function_type_name`.
namespace function_type {
inline constexpr const char* kScalar = "scalar";
inline constexpr const char* kTable = "table";
inline constexpr const char* kTableBuffering = "table_buffering";
inline constexpr const char* kAggregate = "aggregate";
}  // namespace function_type

// `FunctionInfo.function_type` on the wire: the `FunctionType` member names.
namespace function_type_name {
inline constexpr const char* kScalar = "SCALAR";
inline constexpr const char* kTable = "TABLE";
inline constexpr const char* kTableBuffering = "TABLE_BUFFERING";
inline constexpr const char* kAggregate = "AGGREGATE";
}  // namespace function_type_name

// `MacroInfo.macro_type` on the wire: the `MacroType` member names.
namespace macro_type {
inline constexpr const char* kScalar = "SCALAR";
inline constexpr const char* kTable = "TABLE";
}  // namespace macro_type

namespace stability {
inline constexpr const char* kConsistent = "CONSISTENT";
inline constexpr const char* kVolatile = "VOLATILE";
inline constexpr const char* kConsistentWithinQuery = "CONSISTENT_WITHIN_QUERY";
}  // namespace stability

namespace null_handling {
inline constexpr const char* kDefault = "DEFAULT";
inline constexpr const char* kSpecial = "SPECIAL";
}  // namespace null_handling

namespace order_dependence {
inline constexpr const char* kOrderDependent = "ORDER_DEPENDENT";
inline constexpr const char* kNotOrderDependent = "NOT_ORDER_DEPENDENT";
}  // namespace order_dependence

namespace distinct_dependence {
inline constexpr const char* kDistinctDependent = "DISTINCT_DEPENDENT";
inline constexpr const char* kNotDistinctDependent = "NOT_DISTINCT_DEPENDENT";
}  // namespace distinct_dependence

namespace phase {
inline constexpr const char* kProcess = "PROCESS";
inline constexpr const char* kInput = "INPUT";
inline constexpr const char* kFinalize = "FINALIZE";
inline constexpr const char* kTableBuffering = "TABLE_BUFFERING";
inline constexpr const char* kTableBufferingFinalize = "TABLE_BUFFERING_FINALIZE";
}  // namespace phase

}  // namespace vgi::enums
