// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>
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
};

// Everything the engine shows a user about a function, plus the return type
// when it is fixed.  A function whose return type depends on its arguments
// leaves `return_type` empty and answers during bind instead.
struct FunctionMetadata {
    std::string description;
    std::shared_ptr<arrow::DataType> return_type;  // null = decided at bind
    std::vector<std::string> tags;
    bool volatile_ = false;
};

}  // namespace vgi
