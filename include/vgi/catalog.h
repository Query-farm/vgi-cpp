// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/type.h>

namespace vgi {

// One source a multi-branch table reads from.
//
// A multi-branch table is one logical relation stitched from several scans —
// a hot tier and a cold tier, say, or a native Parquet reader beside a
// worker-served fallback. Each branch names its own function; `branch_filter`
// is the SQL predicate identifying which rows it owns, and the engine uses it
// to skip branches a query cannot match.
struct CatalogBranch {
    std::string function_name;
    std::string scan_arguments;
    std::optional<std::string> branch_filter;
    bool writable = false;
    // Where the branch's data actually lives, when it is another catalog's
    // table rather than a function's output.
    std::optional<std::string> source_catalog;
    std::optional<std::string> source_schema;
    std::optional<std::string> source_table;
};

// One version of a time-travelling table.
//
// A version carries its own columns, because schema evolution is the point:
// `AT (VERSION => 1)` sees the columns version 1 had, not today's.
struct TimeTravelVersion {
    int64_t version = 0;
    std::shared_ptr<arrow::Schema> columns;
    std::string scan_function;
    std::string scan_arguments;
    // The calendar year this version became current, for `AT (TIMESTAMP => …)`.
    std::optional<int> timestamp_year;
};

// A table the catalog advertises.
//
// A VGI table is not storage: it is a *name* bound to a table function that
// produces its rows. `SELECT * FROM cat.data.numbers` resolves the table, reads
// its scan function, and calls that. The columns declared here are what the
// planner types the query against before any scan runs, so they have to match
// what the scan actually emits.
struct CatalogTable {
    std::string name;
    std::shared_ptr<arrow::Schema> columns;
    // The table function that produces the rows.
    std::string scan_function;
    // IPC-serialized arguments handed to that function. Empty for a scan that
    // takes none.
    std::string scan_arguments;

    std::optional<std::string> comment;
    std::optional<int64_t> cardinality;
    std::vector<std::pair<std::string, std::string>> tags;

    // Versions, newest last. Empty means the table does not time-travel and an
    // AT clause against it is an error rather than a no-op.
    std::vector<TimeTravelVersion> time_travel;

    // WHERE-filter groups this table requires, in conjunctive normal form: an
    // AND of OR-groups of dotted column paths.
    //
    // A group is satisfied when *any* of its paths carries a filter, and every
    // group must be satisfied. A single-path group is a plain mandatory
    // filter; `{"ticker", "cik"}` means "one of these two". Empty means no
    // requirement. The engine enforces it — the worker only declares it.
    std::vector<std::vector<std::string>> required_filters;

    // Sources this table is stitched from.
    //
    // Empty means the single `scan_function` above is the whole table. A
    // non-empty list *overrides* it: the engine asks for the branches and
    // reads each in turn.
    std::vector<CatalogBranch> branches;
    // DuckDB extensions the branches need (`iceberg`, `parquet`).
    std::vector<std::string> required_extensions;

    // Whether the scan function travels inside the table record.
    //
    // Inlined, the engine skips `catalog_table_scan_function_get` entirely —
    // one fewer round trip per query. Not inlined, it asks, which is what a
    // table whose scan depends on the query (a time-travel AT clause, say)
    // needs.
    bool inline_scan = true;
};

// A view: a name bound to SQL the engine expands.
struct CatalogView {
    std::string name;
    // The SQL text. Called `definition` on the wire; named to match, since a
    // mismatch here is rejected rather than ignored.
    std::string definition;
    std::optional<std::string> comment;
};

// A DuckDB setting this catalog introduces.
//
// Declared at ATTACH, which is what makes `SET my_setting = ...` work at all:
// the engine registers each of these as a real DuckDB setting. Separately, a
// function must name a setting in `required_settings` for its *value* to be
// forwarded — declaring it here creates it, naming it there delivers it.
struct SettingSpec {
    std::string name;
    std::string description;
    std::shared_ptr<arrow::DataType> type;
};

// A secret type this catalog introduces.
//
// Declaring it is what lets a user write `CREATE SECRET ... (TYPE vgi_example,
// ...)`. Its parameter schema names the fields, and a field carrying
// `redact=true` metadata is one the engine must not print.
struct SecretTypeSpec {
    std::string name;
    std::string description;
    std::shared_ptr<arrow::Schema> parameters;
};

// A schema and everything declared in it.
struct CatalogSchema {
    std::string name = "main";
    std::vector<CatalogTable> tables;
    std::vector<CatalogView> views;
};

// The catalog a worker advertises: what `ATTACH '<name>' (TYPE vgi)` binds to.
//
// A worker serves one primary catalog and may serve secondaries alongside it,
// which is why the name is data rather than a compile-time constant — the
// example worker picks it from VGI_WORKER_CATALOG_NAME so one binary can stand
// in for several fixtures.
struct CatalogModel {
    std::string name = "main";
    std::string implementation_version;
    std::string source_url;

    // The data versions this worker can serve, and which it picks when the
    // caller does not say. Empty means the catalog is unversioned and
    // `resolved_data_version` is null.
    std::vector<std::string> supported_data_versions;
    std::optional<std::string> default_data_version;
    // The implementation versions it can serve. Empty means only
    // `implementation_version` above.
    std::vector<std::string> supported_implementation_versions;
    // The version range this catalog's data satisfies, advertised at discovery.
    std::optional<std::string> data_version_spec;
    std::optional<std::string> comment;
    std::vector<std::pair<std::string, std::string>> tags;

    // Whether this catalog answers AT clauses at all. A function-backed table
    // may read the clause itself rather than declaring versions, which is why
    // this is a catalog-level flag and not derived from the tables.
    bool supports_time_travel = false;

    // Settings this catalog introduces to the engine.
    std::vector<SettingSpec> settings;
    // Secret types this catalog introduces.
    std::vector<SecretTypeSpec> secret_types;

    // Schemas declared up front. Registering a function in a schema adds it
    // here too, so a worker with only functions never has to list them.
    //
    // Held indirectly so that `schema()` can hand out a reference that
    // survives a later registration. A vector of values would reallocate, and
    // the documented usage — take a schema, register more functions, then add
    // tables to it — would dangle.
    std::vector<std::unique_ptr<CatalogSchema>> schemas;

    CatalogModel();

    // The schema of that name, creating it if absent. The reference stays
    // valid for the life of the model.
    CatalogSchema& schema(const std::string& schema_name);
    const CatalogSchema* find_schema(const std::string& schema_name) const;

    // Every declared schema name, in declaration order.
    std::vector<std::string> schema_names() const;
};

}  // namespace vgi
