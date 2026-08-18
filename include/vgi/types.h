// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/type.h>

namespace vgi {

// How DuckDB may supply one argument of a function.
//
// A VGI argument is not simply "a column": the engine needs to know whether a
// value must be constant at bind time (so it can be read during binding),
// whether it may be named, and what it looks like in an error message.  That
// is what an ArgSpec carries.
struct ArgSpec {
    std::string name;
    // Position in the call.  Named-only arguments carry no index.
    std::optional<int> index;
    // DuckDB type name as written in SQL ("varchar", "bigint", "any"), not an
    // Arrow type: the engine resolves overloads in its own type system before
    // any Arrow schema exists.
    std::string type;
    std::string description;
    // A constant argument is evaluated during bind and its value handed to
    // bind(); a column argument arrives per batch in process().
    bool constant = false;
    // Named parameters only: a required one that the call site omitted is
    // refused at bind. `ArgSpec::named` clears it, since an optional keyword
    // argument is the usual case; positional parameters are the engine's to
    // supply and it always does.
    bool required = true;
    bool varargs = false;
    // Set to declare a concrete Arrow type directly, bypassing `type`. Needed
    // when two overloads differ only by width (int32 vs int64) — the VGI type
    // names are coarser than Arrow's and would collapse them.
    std::shared_ptr<arrow::DataType> arrow_type;

    static ArgSpec column(std::string name, int index, std::string type,
                          std::string description = "");
    static ArgSpec constant_arg(std::string name, int index, std::string type,
                                std::string description = "");
    static ArgSpec named(std::string name, std::string type, std::string description = "");
    // A polymorphic column: the engine resolves its type at the call site and
    // reports it to bind. Declared as `any` with a null placeholder type.
    static ArgSpec any_column(std::string name, int index, std::string description = "");
    // A table-valued argument — the input relation of a table-in-out function.
    // Marked with `vgi_type=table` rather than an Arrow type, since the shape
    // is whatever the caller passes.
    static ArgSpec table(std::string name, int index, std::string description = "");

    // A named predicate the resolved type must satisfy, checked at bind.
    //
    // Without it a function like `double(x)` accepts `double('text')` and
    // fails somewhere inside process() with a cast error that names neither
    // the function nor the argument.
    struct TypeBound {
        std::string name;
        bool (*accepts)(const arrow::DataType&) = nullptr;
    };
    std::optional<TypeBound> type_bound;

    // Per-argument constraint metadata, surfaced by
    // `vgi_function_arguments()` for discovery. All optional.
    //
    // A discovery surface *and* a contract: the engine renders the bounds as
    // interval notation to show a caller what a sensible value looks like, and
    // a constant argument that violates one is refused at bind with an error
    // naming the function and the argument.
    std::optional<double> ge;
    std::optional<double> le;
    std::optional<double> gt;
    std::optional<double> lt;
    // A closed set of allowed values, as a JSON array.
    std::optional<std::string> choices;
    // A regex the value must match — an open set, where `choices` is closed.
    std::optional<std::string> pattern;
    // The argument's default, as a JSON scalar.
    std::optional<std::string> default_value;

    ArgSpec& with_range(std::optional<double> low, std::optional<double> high);
    ArgSpec& with_bound(TypeBound bound);
    // Mark the parameter variadic: it stands for one or more arguments of its
    // type rather than exactly one.
    ArgSpec& with_varargs();

    // A constant argument whose Arrow type is given directly, for the cases
    // the VGI type names cannot express — a struct, or a specific integer
    // width the coarser names would collapse.
    static ArgSpec constant_typed(std::string name, int index,
                                  std::shared_ptr<arrow::DataType> type,
                                  std::string description = "");
    // A column argument with an exact Arrow type. Distinct from the named
    // forms above because overload resolution happens on the exact type: two
    // overloads that differ only by integer width need this, not "int64".
    static ArgSpec column_typed(std::string name, int index, std::shared_ptr<arrow::DataType> type,
                                std::string description = "");
};

// A worked example, surfaced by the engine's function-documentation views.
struct FunctionExample {
    std::string sql;
    std::string description;
    std::optional<std::string> expected_output;
};

// How a function treats NULL arguments.
enum class NullHandling {
    // The engine short-circuits: a row with any NULL argument yields NULL
    // without the function being called.
    Default,
    // The function is called for every row and decides for itself. Needed by
    // anything that gives NULL a meaning, e.g. coalesce-like behaviour.
    Special,
};

// Severity of a message a function sends back to the client.
//
// In-band, because a worker has nowhere else to put it: stdout is the Arrow
// channel and stderr is swallowed by the engine.
enum class LogLevel { Debug, Info, Warning, Error };

// Whether the engine may reuse a result.
enum class Stability {
    // Same arguments, same answer, always. The engine may fold or cache.
    Consistent,
    // Re-evaluated per row; nothing may be reused. Random and clock-reading
    // functions need this or the engine will call once and repeat the answer.
    Volatile,
    // Fixed for the duration of one query, free to differ between queries.
    ConsistentWithinQuery,
};

// A secret the engine should resolve before calling.
struct SecretLookup {
    std::string secret_type;
    // A scope narrows which secret matches (an S3 prefix, a host). Empty means
    // any secret of this type.
    std::optional<std::string> scope;
    // A specific secret by name, when the caller knows it.
    std::optional<std::string> secret_name;
};

// The wire spellings of the two enum-valued strings in FunctionMetadata.
//
// Public because a function sets them: they are the only place a fixture would
// otherwise have to spell a protocol constant by hand, and the engine matches
// them exactly — a lowercase value is silently ignored.
namespace partition_kinds {
inline constexpr const char* kNotPartitioned = "NOT_PARTITIONED";
inline constexpr const char* kSingleValuePartitions = "SINGLE_VALUE_PARTITIONS";
inline constexpr const char* kOverlappingPartitions = "OVERLAPPING_PARTITIONS";
inline constexpr const char* kDisjointPartitions = "DISJOINT_PARTITIONS";
}  // namespace partition_kinds

namespace order_preservations {
inline constexpr const char* kPreservesOrder = "PRESERVES_ORDER";
inline constexpr const char* kNoOrderGuarantee = "NO_ORDER_GUARANTEE";
inline constexpr const char* kFixedOrder = "FIXED_ORDER";
}  // namespace order_preservations

// Everything the engine shows a user about a function, plus the return type
// when it is fixed.  A function whose return type depends on its arguments
// leaves `return_type` empty and answers during bind instead.
struct FunctionMetadata {
    std::string description;
    std::shared_ptr<arrow::DataType> return_type;  // null = decided at bind
    std::vector<std::pair<std::string, std::string>> tags;
    std::vector<FunctionExample> examples;
    std::vector<std::string> categories;
    Stability stability = Stability::Consistent;
    NullHandling null_handling = NullHandling::Default;
    // Secrets this function needs resolved before it runs.
    //
    // Answered from bind: when a function asks and the engine has not resolved
    // them yet, bind returns the lookups and the engine re-binds with the
    // values in place. Two round trips, which is why the list is empty for
    // functions that need none.
    std::vector<SecretLookup> required_secrets;
    // Whether the function's *input rows come from its arguments* rather than
    // from a table-valued argument.
    //
    // This is what makes a "blended" function callable both as a plain scan —
    // `f(52, 13)` — and correlated against a table — `FROM t, f(t.x, t.y)`.
    // Without it neither call site resolves, because the engine is looking for
    // a table argument the function does not declare.
    bool input_from_args = false;
    // How this function's output is partitioned. Declaring
    // SINGLE_VALUE_PARTITIONS says each batch holds exactly one distinct value
    // of the partition column, which is what lets the engine cache and skip
    // per partition.
    std::string partition_kind;
    // Whether the engine may narrow this scan's columns before calling.
    // Declaring it is what makes the bound output schema carry only the
    // columns the query needs; without it the function is always asked for
    // all of them.
    bool projection_pushdown = false;
    // Whether the engine may push WHERE predicates into this scan.
    bool filter_pushdown = false;
    // Whether the framework should apply those predicates for you.
    //
    // Off by default: a function that declares filter_pushdown usually wants
    // to *use* the filters — to skip a partition or narrow a remote request —
    // and having them silently applied afterwards as well is only wasted work.
    // On, the framework filters each emitted batch, which is what a fixture
    // that merely advertises the capability wants.
    bool auto_apply_filters = false;
    // Whether the engine may rewrite a scan of this function into a
    // late-materialization plan: fetch the row ids first, then fetch only the
    // surviving rows' columns.
    //
    // Only sound for a function whose rows have a stable, unique row id —
    // without that the second fetch cannot find them again.
    bool late_materialization = false;
    // Whether the engine must stamp each input batch with its index.
    //
    // A buffering sink that reassembles the input's order needs it; without
    // declaring it the index is not sent, and asking for it later finds
    // nothing.
    bool requires_input_batch_index = false;
    // Whether the order rows arrive in matters to a buffering sink, and
    // whether the order its finalize emits them in matters to the engine.
    bool sink_order_dependent = false;
    bool source_order_dependent = false;
    // Whether this function tags its batches with `vgi_batch_index`.
    //
    // Declaring it is what lets the engine hand the scan to several workers
    // and still restore the original order, by sorting on the tag rather than
    // on arrival. A function that declares it and omits the tag is rejected.
    bool supports_batch_index = false;
    // What order the engine may assume of this function's rows. Empty leaves
    // the engine's own default in place; the values are in `src/enums.h`.
    std::string order_preservation;
    // Whether this function's answer depends on the order its input arrives
    // in, and whether it depends on duplicates being present. Declaring
    // either constrains what the optimizer may reorder or deduplicate above
    // the call. Off is the safe default *for the optimizer's freedom*, so a
    // function that does care has to say so.
    bool order_dependent = false;
    bool distinct_dependent = false;
    // Whether the engine may push a TABLESAMPLE SYSTEM clause into this scan.
    //
    // Declaring it hands the function the sampling rate and lets it discard
    // rows at the source; DuckDB then drops its own sampling operator, so a
    // function that declares this and ignores the hint returns every row.
    bool sampling_pushdown = false;
    // DuckDB settings this function reads. Declaring them is what makes the
    // engine forward their values; a setting not declared here never arrives,
    // however it was set.
    std::vector<std::string> required_settings;
};

}  // namespace vgi
