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
// The DuckDB extension holds every served table to its ATTACH contract, and
// this SDK applies the same rules before it serves one (and, for a declarative
// catalog, when the worker starts):
//
//  (a) records are read only when ATTACH negotiated `CONSTRAINT_RECORDS_READ`
//      (and CHECK records also `SCALAR_EXPRESSION_READ`). A catalog whose
//      tables carry records offers both automatically; see
//      `CatalogModel::catalog_read_profiles` and select_catalog_capabilities().
//  (b) only ENFORCED + ALL_ROWS records are accepted, and each must belong to
//      the table serving it, resolve against its columns, and stay inside the
//      extension's native subset (one PRIMARY KEY, top-level UNIQUE / KEY / FK
//      columns, same-schema NO ACTION / RESTRICT foreign keys, numeric
//      comparison CHECKs). validate_constraint_records() names what fails.
//  (c) an Arrow field (descending through ordinary STRUCTs) is non-nullable
//      exactly when a NOT_NULL or PRIMARY_KEY record names its column path.
//      The SDK makes this hold rather than asking the author to: served_columns()
//      serves every field no record names as nullable, and every field a record
//      names as non-nullable. Nullability is never read as a constraint.
#pragma once

#include <memory>
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
// DDL profile: CREATE TABLE may carry constraint definitions. Requires
// CONSTRAINT_RECORDS_READ and CONSTRAINT_ENFORCE alongside it.
inline constexpr const char* CONSTRAINT_CREATE = "vgi.constraint.create.v1";
// Write profile: the catalog enforces its constraints on every write path.
inline constexpr const char* CONSTRAINT_ENFORCE = "vgi.constraint.enforce.v1";

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

// The schema a table is served with: `columns` with every field a NOT_NULL or
// PRIMARY_KEY record names made non-nullable and every other field made
// nullable, descending through ordinary STRUCTs (not ones carrying
// `ARROW:extension:name` or `vgi.logical_type.*` metadata). A top-level
// `is_row_id` column is left as it is. Metadata is preserved. Throws
// std::invalid_argument on a record that does not decode.
std::shared_ptr<arrow::Schema> served_columns(const arrow::Schema& columns,
                                              const std::vector<std::string>& constraints);

// Check a table's records the way the DuckDB extension does at ATTACH, against
// the table as served (served_columns() is applied first): every record is
// ENFORCED + ALL_ROWS and owned by `schema_path`.`table` (a one-component
// schema path); ids are unique and declared names unique case-insensitively;
// at most one PRIMARY_KEY; NOT_NULL paths resolve through ordinary STRUCTs and
// PRIMARY_KEY / UNIQUE / FOREIGN_KEY name top-level columns, never the row-id
// column; a FOREIGN_KEY points at a named key of a table in the same schema
// with MATCH SIMPLE and NO ACTION / RESTRICT; a CHECK is a comparison of
// numeric literals and primitive signed numeric columns under the default
// evaluation context. Throws std::invalid_argument naming the table, the
// constraint id and the column path. The SDK runs this on every table it
// serves; call it to fail earlier.
void validate_constraint_records(const SchemaPath& schema_path, const std::string& table,
                                 const arrow::Schema& columns,
                                 const std::vector<std::string>& constraints);

// The same checks without the ownership one: `table` is only the name used in
// messages (e.g. "main.t").
void validate_constraint_records(const std::string& table, const arrow::Schema& columns,
                                 const std::vector<std::string>& constraints);

// Catalog capability negotiation (`CatalogAttachResult.catalog_capabilities` /
// `selected_capabilities`, `ClientCapabilities.catalog`): profile names by
// category.
struct CatalogCapabilityProfiles {
    std::vector<std::string> read, ddl, write, transaction;
    bool empty() const {
        return read.empty() && ddl.empty() && write.empty() && transaction.empty();
    }
};

// Encode a CatalogCapabilities record. Throws std::invalid_argument on a
// malformed or duplicate profile name.
std::string encode_catalog_capabilities(const CatalogCapabilityProfiles& profiles);

// Decode a CatalogCapabilities record (the inverse of the above).
CatalogCapabilityProfiles decode_catalog_capabilities(const std::string& bytes);

// The catalog offer inside a `CatalogAttachRequest.client_capabilities`
// ClientCapabilities record; empty when the client sent none.
CatalogCapabilityProfiles client_catalog_offer(
    const std::optional<std::string>& client_capabilities);

// What an attachment selects: `available` intersected with `offered`, per
// category, in `available`'s order, then pruned to what the extension accepts
// together -- SCALAR_EXPRESSION_READ, CONSTRAINT_CREATE and CONSTRAINT_ENFORCE
// only alongside CONSTRAINT_RECORDS_READ, and CONSTRAINT_CREATE only alongside
// CONSTRAINT_ENFORCE.
CatalogCapabilityProfiles select_catalog_capabilities(const CatalogCapabilityProfiles& available,
                                                      const CatalogCapabilityProfiles& offered);

}  // namespace vgi
