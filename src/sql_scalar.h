// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// The two bounded scalar SQL grammars the DuckDB extension reads, checked on
// the worker side so a catalog the extension would refuse at ATTACH fails when
// the worker starts instead:
//
//  - a CHECK constraint (vgi_constraint_contract.cpp ParseNumericCheck): one
//    comparison between two numeric operands, each a numeric literal or an
//    unqualified primitive signed numeric column;
//  - a VIRTUAL generated column (vgi_column_expression.cpp Parse/Scalar with
//    generated = true): literals, unqualified column references, CAST to a
//    Boolean / signed integer / VARCHAR type, comparisons, AND / OR / NOT,
//    IS [NOT] NULL, COALESCE and the functions + - * % abs lower upper ||.
//
// This is a recognizer, not DuckDB's parser: it accepts a subset of what the
// extension accepts and refuses anything it does not understand, so a
// refusal here can be stricter than ATTACH but an acceptance never is weaker.
#pragma once

#include <string>
#include <vector>

#include <arrow/type_fwd.h>

namespace vgi::sql {

// Throws std::invalid_argument when `sql` is not a CHECK the extension reads
// against `columns` (the served table schema; the `is_row_id` column is not
// referable).
void validate_numeric_check(const std::string& sql, const arrow::Schema& columns);

// Throws std::invalid_argument when a top-level field's VIRTUAL generated
// column definition is one the extension refuses: a non-nullable field, an
// unsupported type, SQL outside the generated grammar, a reference to a
// missing column or one of an unsupported type, a self-reference or cycle, or
// a table of only generated columns. `generated_sql[i]` is the decoded SQL of
// top-level field `i`, empty for a physical column.
void validate_generated_columns(const arrow::Schema& columns,
                                const std::vector<std::string>& generated_sql);

}  // namespace vgi::sql
