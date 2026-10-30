// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Canonical constraint records for catalogs that state their own facts.
//
// Under vgi.v3 a table's constraints travel as `TableInfo.constraints`: a list
// of IPC-encoded `ConstraintInfo` records, each the provider's attestation of
// one fact about every row the table can return. The SDK never derives them
// -- not from the legacy `CatalogTable::not_null` / `primary_key` / ... fields,
// and not from Arrow nullability. A worker builds each one explicitly with
// `constraint_record()` and puts it in `CatalogTable::constraints` (or a
// `TimeTravelVersion::constraints` snapshot).
//
// The DuckDB extension holds the record set to three rules at ATTACH, and this
// SDK checks the latter two whenever it serves a table:
//
//  (a) records are read only when ATTACH negotiated `CONSTRAINT_RECORDS_READ`
//      (and CHECK records also `SCALAR_EXPRESSION_READ`). A declarative
//      catalog whose tables carry records offers both automatically; see
//      `CatalogModel::catalog_read_profiles`.
//  (b) only ENFORCED + ALL_ROWS records are accepted.
//  (c) for every table, with records or without, an Arrow field (descending
//      through ordinary STRUCTs) is non-nullable exactly when a NOT_NULL or
//      PRIMARY_KEY record names its column path.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <arrow/type_fwd.h>

#include "vgi/types.h"

namespace vgi {

// Read profile: the catalog serves canonical ConstraintInfo records.
inline constexpr const char* CONSTRAINT_RECORDS_READ = "vgi.constraint.records.read.v1";
// Read profile: CHECK records carry DuckDB scalar SQL expressions. Selected
// only alongside CONSTRAINT_RECORDS_READ.
inline constexpr const char* SCALAR_EXPRESSION_READ = "vgi.expression.duckdb.scalar.v1";

// A column path: raw components from the top-level column down through
// STRUCT children, e.g. {"id"} or {"bbox", "xmin"}. Dots are literal.
using ColumnPath = std::vector<std::string>;

enum class ConstraintKind { NotNull, PrimaryKey, Unique, Check, ForeignKey };

// Whether the authority enforces the constraint on every write path.
enum class ConstraintEnforcement { Enforced, MetadataOnly };

// Which existing rows the constraint has been checked against. Only AllRows
// justifies an engine fact.
enum class ConstraintValidation { AllRows, FutureWrites, Unvalidated };

// The wire spelling of each enum ("NOT_NULL", "ENFORCED", "ALL_ROWS", ...).
const char* to_string(ConstraintKind kind);
const char* to_string(ConstraintEnforcement enforcement);
const char* to_string(ConstraintValidation validation);

// The key a FOREIGN_KEY record points at.
struct ConstraintReference {
    SchemaPath schema_path;
    std::string table;
    std::vector<ColumnPath> columns;
    // `id` of the referenced table's PRIMARY_KEY or UNIQUE record.
    std::string constraint_id;
};

// One constraint, as `constraint_record()` encodes it.
struct ConstraintRecordSpec {
    // Schema path and name of the owning table.
    SchemaPath schema_path;
    std::string table;
    // Stable provider-owned identity, unique within the table. Never derived
    // by the SDK.
    std::string id;
    ConstraintKind kind = ConstraintKind::NotNull;
    // Column paths; exactly one for NOT_NULL, none for CHECK.
    std::vector<ColumnPath> columns;
    // The provider's attestation. Both are required and have no default: the
    // DuckDB extension reads only ENFORCED + ALL_ROWS records, so state that
    // only when every row the table can ever return satisfies the constraint.
    std::optional<ConstraintEnforcement> enforcement;
    std::optional<ConstraintValidation> validation;
    // Optional SQL label (`CONSTRAINT <name>`).
    std::optional<std::string> declared_name;
    // CHECK only: a DuckDB scalar expression over the table's columns.
    std::optional<std::string> check_sql;
    // FOREIGN_KEY only.
    std::optional<ConstraintReference> references;
};

// Encode one canonical ConstraintInfo record for `CatalogTable::constraints`.
// Throws std::invalid_argument on an inconsistent spec.
std::string constraint_record(const ConstraintRecordSpec& spec);

// A decoded ConstraintInfo record, in `constraint_record()`'s vocabulary.
struct ConstraintRecordView {
    SchemaPath schema_path;
    std::string table;
    std::string id;
    ConstraintKind kind = ConstraintKind::NotNull;
    std::vector<ColumnPath> columns;
    ConstraintEnforcement enforcement = ConstraintEnforcement::Enforced;
    ConstraintValidation validation = ConstraintValidation::AllRows;
    std::optional<std::string> declared_name;
    std::optional<std::string> check_sql;
    // `constraint_id` is empty when the record names no referenced key.
    std::optional<ConstraintReference> references;
};

// Decode a `TableInfo.constraints` entry (the inverse of constraint_record).
// Throws std::invalid_argument on bytes that are not one ConstraintInfo.
ConstraintRecordView read_constraint_record(const std::string& bytes);

// Check a table's records the way the DuckDB extension does at ATTACH: every
// record is ENFORCED + ALL_ROWS, every NOT_NULL / PRIMARY_KEY path names a
// column, and a field -- descending through ordinary STRUCTs, skipping a
// top-level `is_row_id` column -- is non-nullable exactly when such a record
// names its path. Throws std::invalid_argument naming `table` and the column
// path. The SDK runs this on every table it serves; call it to fail earlier.
void validate_constraint_records(const std::string& table, const arrow::Schema& columns,
                                 const std::vector<std::string>& constraints);

}  // namespace vgi
