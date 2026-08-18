// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/type.h>

#include "vgi/statistics.h"

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

// A foreign key on a table, by column *name* rather than index — the wire
// carries names here, and a name survives a schema that gains a column.
struct ForeignKey {
    std::vector<std::string> columns;
    std::string referenced_table;
    std::vector<std::string> referenced_columns;
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
    // Absent means the single `scan_function` above is the whole table. Present
    // *overrides* it: the engine asks for the branches and reads each in turn —
    // including when the list is empty, which is a table that declares itself
    // multi-branch and then offers no branch, and which the engine refuses.
    // That is a different thing from a table that never mentioned branches, so
    // the two cannot share one empty vector.
    std::optional<std::vector<CatalogBranch>> branches;
    // DuckDB extensions the branches need (`iceberg`, `parquet`).
    std::vector<std::string> required_extensions;

    // Constraints, as DuckDB's catalog views report them. Columns are given
    // by index into `columns`, except the foreign keys, whose wire form is by
    // name.
    //
    // Declarative only: nothing here is enforced by the worker, and the engine
    // does not enforce it on a VGI table either. It exists so a tool reading
    // the catalog sees the same shape it would see for a local table.
    std::vector<int32_t> not_null;
    std::vector<std::vector<int32_t>> unique;
    std::vector<std::vector<int32_t>> primary_key;
    std::vector<std::string> check;
    std::vector<ForeignKey> foreign_keys;

    // Per-column bounds for the optimizer. Empty means the table declines to
    // answer, which is different from answering with no columns: the engine
    // then falls back to asking the scan function for its own.
    std::vector<ColumnStatistics> column_statistics;

    // Whether this table answers an AT clause without declaring versions.
    //
    // Needed for a function-backed table that reads the clause itself at bind:
    // without it an AT against a table with an empty `time_travel` list is
    // refused before the function ever sees it.
    bool supports_time_travel = false;

    // Whether the scan function travels inside the table record.
    //
    // Inlined, the engine skips `catalog_table_scan_function_get` entirely —
    // one fewer round trip per query, and the inlined binding is the one the
    // AT-resolved version supplies. Not inlined, the engine asks, either
    // through that method or through `catalog_table_scan_branches_get`; both
    // resolve the clause, so either setting travels correctly.
    bool inline_scan = true;
};

// A view: a name bound to SQL the engine expands.
struct CatalogView {
    std::string name;
    // The SQL text. Called `definition` on the wire; named to match, since a
    // mismatch here is rejected rather than ignored.
    std::string definition;
    std::optional<std::string> comment;
    std::vector<std::pair<std::string, std::string>> tags;
    // Aligned by name, not by position: the view's columns are whatever its
    // SQL produces, and the engine matches these against them.
    std::vector<std::pair<std::string, std::string>> column_comments;
};

// A macro: a name bound to SQL the engine substitutes at the call site.
//
// Unlike a view, a macro takes parameters, and unlike a function it never
// reaches the worker at run time — the engine expands it. Declaring one is
// therefore pure catalog metadata.
struct CatalogMacro {
    std::string name;
    // Order is load-bearing: it is the positional order of the call.
    std::vector<std::string> parameters;
    std::string definition;
    // A table macro is called in FROM, a scalar one in an expression.
    bool table_macro = false;
    std::optional<std::string> comment;
    // Defaults for trailing parameters, by name. Integers only, which is what
    // the wire carries.
    std::vector<std::pair<std::string, int64_t>> defaults;
    // Per-parameter descriptions, surfaced by `vgi_function_arguments()`
    // through the same `vgi_doc` channel a function's arguments use.
    std::vector<std::pair<std::string, std::string>> parameter_docs;
};

// An option a caller may set in the ATTACH statement.
//
// Distinct from a setting: a setting is global and mutable, an attach option
// belongs to one attachment and is fixed for its life. The engine validates
// the value against `type` before the worker ever sees it, so a wrong-typed
// option fails the ATTACH rather than the first query.
struct AttachOptionSpec {
    std::string name;
    std::string description;
    std::shared_ptr<arrow::DataType> type;
    // The value used when the caller supplies none. Null means there is none;
    // an option with a default is always satisfiable, so declaring both a
    // default and `required` is a contradiction and is refused.
    std::shared_ptr<arrow::Array> default_value;
    bool required = false;
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
    std::optional<std::string> comment;
    std::vector<std::pair<std::string, std::string>> tags;
    std::vector<CatalogTable> tables;
    std::vector<CatalogView> views;
    std::vector<CatalogMacro> macros;
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

    // Options this catalog accepts in an ATTACH statement. The declared
    // defaults are what an attachment that sets none is given.
    std::vector<AttachOptionSpec> attach_options;

    // Function names this catalog asks the engine to publish into its global,
    // catalog-independent namespace, each under `global_function_prefix`.
    //
    // A published function stays schema-resident: bind dispatch is keyed on
    // (schema, name), so the global name is an alias the engine resolves, not
    // a second registration. Every name here must also be registered.
    std::vector<std::string> global_functions;
    std::string global_function_prefix;

    // Whether this catalog answers AT clauses at all. A function-backed table
    // may read the clause itself rather than declaring versions, which is why
    // this is a catalog-level flag and not derived from the tables.
    bool supports_time_travel = false;

    // Whether the engine should open a catalog transaction around BEGIN/COMMIT.
    //
    // Off, `BindParams::transaction_opaque_data` is never set and a function
    // has no way to remember anything for the length of a statement group. On,
    // the engine calls `catalog_transaction_begin` and echoes the id it gets
    // back on every call the transaction covers.
    bool supports_transactions = false;

    // Settings this catalog introduces to the engine.
    std::vector<SettingSpec> settings;
    // Secret types this catalog introduces.
    std::vector<SecretTypeSpec> secret_types;

    // Whether a version request is an npm-style range rather than an exact
    // version.
    //
    // Off, `data_version_spec '1.0.0'` must name a supported version exactly.
    // On, `^1.0.0` / `~1.0.0` / `1` / `1.0` resolve to the highest supported
    // version they cover, and an unsatisfiable range fails the ATTACH.
    bool npm_version_resolution = false;

    // The schemas each data version exposes, keyed by resolved version.
    //
    // Non-empty, these *replace* `schemas` for the life of an attachment: a
    // catalog whose table set changes across versions is the whole reason the
    // resolved version is sealed into `attach_opaque_data` and echoed on every
    // later call.
    std::map<std::string, std::vector<CatalogSchema>> version_schemas;

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
