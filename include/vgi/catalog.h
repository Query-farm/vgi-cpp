// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/type.h>

#include "vgi/constraints.h"
#include "vgi/statistics.h"
#include "vgi/types.h"

namespace vgi {

// Translate a VIRTUAL SQL convenience value with explicit owner and default context.
// Returns canonical base64 SqlExpression IPC; does not bind or execute SQL.
std::string generated_expression_metadata(const std::string& sql, const SchemaPath& owner);

// One column path in a `required_filters` group: raw components from the
// top-level column down through STRUCT children, e.g. {"ticker"} or
// {"bbox", "xmin"}. Dots and quotes are literal.
//
// A type of its own, not a std::vector<std::string>, so that the flat spelling
// `{{"ticker", "cik"}}` -- which against nested vectors picked vector's
// iterator-pair constructor from the two string literals and compiled into
// garbage paths or std::length_error -- is a compile error. Spell a group of
// two alternative paths `{{{"ticker"}, {"cik"}}}`.
struct RequiredFilterPath {
    std::vector<std::string> components;

    RequiredFilterPath(std::initializer_list<std::string> path) : components(path) {}
    explicit RequiredFilterPath(std::vector<std::string> path) : components(std::move(path)) {}

    bool operator==(const RequiredFilterPath& other) const {
        return components == other.components;
    }
};

// An OR-group of required-filter paths: satisfied when any path is filtered.
using RequiredFilterGroup = std::vector<RequiredFilterPath>;

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
    std::optional<SchemaPath> source_schema_path;
    std::optional<std::string> source_table;
    std::optional<std::string> format_name;
    std::optional<std::vector<std::string>> format_locations;
    std::optional<std::string> format_options;
    // Schema containing a VGI function branch; absent for native functions.
    std::optional<SchemaPath> schema_path;
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
    // Canonical records for this version (vgi::constraint_record()). Absent,
    // the version serves the table-level `CatalogTable::constraints` and
    // `definition_revision`; an empty vector states that this version has
    // none. Either way the records are checked against *this* version's
    // columns when the worker starts, naming the table and version.
    std::optional<std::vector<std::string>> constraints;
    // Nonempty when present.
    std::optional<std::string> definition_revision;
};

// A foreign key on a table, by column *name* rather than index — the wire
// carries names here, and a name survives a schema that gains a column.
struct ForeignKey {
    std::vector<std::string> columns;
    std::string referenced_table;
    std::vector<std::string> referenced_columns;
    // Defaults to the owning table's schema path when absent.
    std::optional<SchemaPath> referenced_schema_path;
};

// The legacy constraint members below are deprecated so that code naming them
// is warned. The struct's own implicitly defined members (copy, move,
// destruction) touch them too; that use is not the caller's, so it is
// silenced here and only here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// A table the catalog advertises.
//
// A VGI table is not storage: it is a *name* bound to a table function that
// produces its rows. `SELECT * FROM cat.data.numbers` resolves the table, reads
// its scan function, and calls that. The columns declared here are what the
// planner types the query against before any scan runs, so they have to match
// what the scan actually emits.
struct CatalogTable {
    // Canonical ConstraintInfo IPC records, each built with
    // vgi::constraint_record() (vgi/constraints.h). The provider attests every
    // one; identities are never synthesized and nothing is derived from the
    // legacy fields below or from Arrow nullability. The worker checks every
    // record against the extension's ATTACH rules when it starts (see
    // vgi::validate_constraint_records()), and serves `columns` with
    // nullability rewritten to match: non-nullable exactly at the
    // NOT_NULL/PRIMARY_KEY paths, nullable everywhere else. Records make ATTACH
    // offer the constraint read profiles automatically. A time-travel version
    // that states no records of its own serves these.
    std::vector<std::string> constraints;
    // An opaque token naming this definition; nonempty when present.
    std::optional<std::string> definition_revision;
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
    // AND of OR-groups of raw component paths (dots and quotes are literal).
    //
    // A group is satisfied when *any* of its paths carries a filter, and every
    // group must be satisfied. A single-path group is a plain mandatory
    // filter; `{{{"ticker"}, {"cik"}}}` means "one of these two" and
    // `{{{"bbox", "xmin"}}}` requires a filter on a STRUCT member. Empty means
    // no requirement. The engine enforces it — the worker only declares it,
    // and each path must resolve against `columns` through STRUCT parents.
    std::vector<RequiredFilterGroup> required_filters;

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

    // Legacy (pre-vgi.v3) constraint descriptors. vgi.v3 has no wire form for
    // them and the SDK will not translate them into canonical records: a table
    // that sets any of these is refused when the worker starts. Declare
    // vgi::constraint_record() records in `constraints` instead.
    [[deprecated(
        "not sent under vgi.v3; declare vgi::constraint_record() records in "
        "CatalogTable::constraints")]] std::vector<int32_t> not_null;
    [[deprecated(
        "not sent under vgi.v3; declare vgi::constraint_record() records in "
        "CatalogTable::constraints")]] std::vector<std::vector<int32_t>> unique;
    [[deprecated(
        "not sent under vgi.v3; declare vgi::constraint_record() records in "
        "CatalogTable::constraints")]] std::vector<std::vector<int32_t>> primary_key;
    [[deprecated(
        "not sent under vgi.v3; declare vgi::constraint_record() records in "
        "CatalogTable::constraints")]] std::vector<std::string> check;
    [[deprecated(
        "not sent under vgi.v3; declare vgi::constraint_record() records in "
        "CatalogTable::constraints")]] std::vector<ForeignKey> foreign_keys;

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
#pragma GCC diagnostic pop

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
    std::vector<std::string> aliases;
    // DuckDB reads only "regular" views; any other value is refused when the
    // worker starts.
    std::string security = "regular";
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
    // Defaults for the simple descriptor, by name. This convenience form accepts integers;
    // build overloads with vgi::macro_overload() for other typed values and declarations.
    std::vector<std::pair<std::string, int64_t>> defaults;
    // Per-parameter descriptions, sent as each `MacroParameter.documentation`
    // and surfaced by `vgi_function_arguments()`.
    std::vector<std::pair<std::string, std::string>> parameter_docs;
    // Complete MacroOverload IPC records, each built with vgi::macro_overload()
    // in this macro's schema and kind; the simple-definition fields above must
    // then be empty.
    std::optional<std::vector<std::string>> overloads;
};

// One parameter of a macro overload.
struct MacroParameterSpec {
    std::string name;
    // Null: the parameter takes any type.
    std::shared_ptr<arrow::DataType> declared_type;
    // A one-element array holding the literal default; null for none. A
    // parameter without a default may not follow one with a default.
    std::shared_ptr<arrow::Array> default_value;
    std::optional<std::string> documentation;
};

// One overload of a macro family, as `macro_overload()` encodes it.
struct MacroOverloadSpec {
    // The owning schema and kind: must equal the CatalogMacro's.
    SchemaPath schema_path = {"main"};
    bool table_macro = false;
    std::vector<MacroParameterSpec> parameters;
    // The macro body: a scalar expression, or a query for a table macro.
    std::string definition;
};

// Encode one MacroOverload IPC record for `CatalogMacro::overloads`, under the
// duckdb.macro.v1 profile and the default evaluation context. Throws
// std::invalid_argument on an empty or NUL-bearing definition or name, a
// duplicate (case-insensitive) parameter name, a default that is not a
// one-element array, or a required parameter after a defaulted one.
std::string macro_overload(const MacroOverloadSpec& spec);

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
    // The option carries a credential (an API key, a token, a password).
    // Clients mask it, and the engine redacts it from `duckdb_databases()` and
    // never logs it. A secret option should normally have no default.
    bool secret = false;
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
    SchemaPath path = {"main"};
    std::optional<std::string> comment;
    std::vector<std::pair<std::string, std::string>> tags;
    std::vector<CatalogTable> tables;
    std::vector<CatalogView> views;
    std::vector<CatalogMacro> macros;
};

// One schema's entry in a `catalog_contents` answer: its path, its encoded
// SchemaInfo, and the encoded items of each kind.
struct SchemaContents {
    SchemaPath path;
    std::string schema;
    std::vector<std::string> tables, views, scalar_functions, aggregate_functions, table_functions,
        scalar_macros, table_macros, indexes;
};

// What a catalog answers `catalog_contents` with.
//
// `not_modified` is only valid with an `etag` equal to the call's
// `if_none_match` and no schemas -- the cheap-validator short circuit. A full
// answer whose etag equals `if_none_match` is turned into not_modified by the
// framework, and with no etag at all `if_none_match` is ignored.
struct CatalogContentsResult {
    std::vector<SchemaContents> schemas;
    std::optional<std::string> etag;
    bool not_modified = false;
};

// One `catalog_contents` call, as a handler sees it.
struct CatalogContentsCall {
    std::string catalog_name;
    // The version the catalog reports for this attachment.
    int64_t catalog_version = 0;
    std::optional<std::string> if_none_match;
    // The default composition: every schema of the attachment with all its
    // items. A cheap validator answers not_modified without calling it.
    std::function<std::vector<SchemaContents>()> contents;
};

// Overrides how a catalog answers `catalog_contents` -- to revalidate with an
// etag of its own, or (a test fixture) to fail and drive the client's
// per-schema fallback.
using CatalogContentsHandler = std::function<CatalogContentsResult(const CatalogContentsCall&)>;

// A framework-computed etag for `catalog_contents`.
enum class CatalogContentsEtag {
    None,
    // SHA-256 of the answer's contents (catalog_contents_digest): every call
    // builds the snapshot, and a matching `if_none_match` is answered
    // not_modified.
    ContentHash,
};

// The content-hash etag of a `catalog_contents` answer: lowercase hex SHA-256
// of the six collection streams the answer sends (schemas parents first), in
// vgi-python 2.2's `contents_digest` framing -- a `vgi.catalog.contents.2.2\0`
// prefix, then per collection its name, a presence byte and, when present,
// the stream's u64 little-endian length and bytes.
//
// The framing is vgi-python's, but the etag is not portable across SDKs: it
// hashes this SDK's own stream bytes, and vgi-python packs up to 1024 rows per
// batch where this SDK writes one batch per item, so the same catalog hashes
// differently in each. Only an empty catalog (every collection null) agrees.
// That is all an etag needs: it is only ever compared with one this worker
// served.
std::string catalog_contents_digest(const std::vector<SchemaContents>& schemas);

// A DDL-capable in-memory catalog (Worker::register_memory_catalog).
//
// Accepts CREATE / DROP SCHEMA, TABLE (definitions only: it holds no rows and
// accepts no writes) and VIEW. CREATE TABLE may declare column NOT NULL: ATTACH
// negotiates CONSTRAINT_RECORDS_READ, CONSTRAINT_CREATE and CONSTRAINT_ENFORCE,
// and each NOT NULL becomes an ENFORCED + ALL_ROWS canonical record -- true
// because no write path exists to store a NULL. PRIMARY KEY, UNIQUE, CHECK
// and FOREIGN KEY are refused (UNIMPLEMENTED): nothing here could enforce
// them. A non-nullable column no NOT NULL names is served nullable. Logical
// (extension) column types such as UUID are refused: stored here they would
// be read back as their storage type. Every ATTACH is private: it starts from an empty `main`
// schema and never sees another attachment's objects, even in one worker process. Its version
// starts at 1, is bumped by every DDL, and is not frozen.
//
// State lives in the worker process that served the ATTACH, which on every
// transport this SDK offers is the one serving the rest of that session.
struct MemoryCatalogOptions {
    std::string name;
    std::optional<std::string> comment;
    // Report catalog_version 0 ("unknown") always: a catalog that does not
    // track its version, which a client re-checks at every transaction start.
    bool unversioned = false;
    // Advertise `supports_catalog_contents` at ATTACH (default on).
    bool supports_catalog_contents = true;
    // Answer catalog_contents this way instead of the plain snapshot.
    CatalogContentsHandler catalog_contents_handler;
    CatalogContentsEtag catalog_contents_etag = CatalogContentsEtag::None;
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

    // Extra read-only protocol offers, merged with the automatic ones: when a
    // table this attachment serves carries `constraints`, ATTACH offers
    // CONSTRAINT_RECORDS_READ and SCALAR_EXPRESSION_READ without being asked.
    // Name them here only for tables the model cannot see (a custom
    // catalog_contents_handler; its table items are still checked and served
    // with their nullability matched to their records). With nothing to offer, capability
    // negotiation stays absent; ATTACH selects only the client's offered
    // subset, and the scalar profile only alongside constraint records. The
    // provider remains responsible for the semantic guarantees of each
    // named profile.
    std::vector<std::string> catalog_read_profiles;

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

    // Whether ATTACH advertises `supports_catalog_contents`, so the engine
    // loads the whole catalog with one `catalog_contents` call instead of
    // `catalog_schemas` plus a `catalog_schema_contents_*` call per schema and
    // kind.  On by default: a catalog here is declared up front and served
    // read-only, so the composed answer is exactly the per-schema one.  Turn
    // it off to force the per-schema path.
    bool supports_catalog_contents = true;
    // How `catalog_contents` is answered, when not the default composition
    // (version-frozen, no etag). See CatalogContentsHandler.
    CatalogContentsHandler catalog_contents_handler;
    CatalogContentsEtag catalog_contents_etag = CatalogContentsEtag::None;

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
    CatalogSchema& schema(const SchemaPath& schema_path);
    CatalogSchema& schema(const std::string& schema_name) {
        return schema(SchemaPath{schema_name});
    }
    const CatalogSchema* find_schema(const SchemaPath& schema_path) const;
    const CatalogSchema* find_schema(const std::string& schema_name) const {
        return find_schema(SchemaPath{schema_name});
    }

    // Every declared schema path, in declaration order.
    std::vector<SchemaPath> schema_paths() const;
};

}  // namespace vgi
