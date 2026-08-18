// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

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
    static ArgSpec named(std::string name, std::string type,
                         std::string description = "");
    // A polymorphic column: the engine resolves its type at the call site and
    // reports it to bind. Declared as `any` with a null placeholder type.
    static ArgSpec any_column(std::string name, int index,
                              std::string description = "");
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
    static ArgSpec column_typed(std::string name, int index,
                                std::shared_ptr<arrow::DataType> type,
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
};

}  // namespace vgi
