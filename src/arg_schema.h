// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Turning declared ArgSpecs into the Arrow schema the `arguments` wire field
// carries.
//
// The encoding is unusual and worth stating plainly: a function's parameter
// list travels as a *schema*, one field per parameter, where the field name
// and type are the parameter's and everything else — constness, varargs,
// named-ness, documentation — rides in the field's metadata. The DuckDB
// extension reads the same keys.
//
// The distinction that matters most is `vgi_const`. A const parameter is a
// bind-time constant, not a column: its value belongs in the bind's arguments
// and must not appear in the per-row input batch. Getting that wrong fails
// quietly, with columns silently off by one.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/type.h>

#include "vgi/types.h"

namespace vgi {

// Map a VGI argument type name ("varchar", "int64", "any") to Arrow.
//
// An unknown name becomes null(), which is also what "any" maps to — the
// extension reads the `vgi_type` metadata rather than the type to decide that a
// parameter is polymorphic, so the type itself only has to be a placeholder.
std::shared_ptr<arrow::DataType> arg_type_to_arrow(const std::string& name);

// The schema for `FunctionInfo.arguments`.
std::shared_ptr<arrow::Schema> build_arg_schema(const std::vector<ArgSpec>& specs);

// The schema for `FunctionInfo.output_schema` of a scalar function.
//
// A function with a fixed return type advertises it. One without advertises a
// null-typed `result` carrying the `vgi:any` marker, which is how DuckDB is
// told to defer the type until bind rather than treating it as SQL NULL.
std::shared_ptr<arrow::Schema> build_scalar_output_schema(
    const std::shared_ptr<arrow::DataType>& return_type);

}  // namespace vgi
