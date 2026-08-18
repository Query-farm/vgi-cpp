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
#include <arrow/record_batch.h>
#include <arrow/type.h>

namespace vgi {

// A call's argument values, as they arrive on the wire.
//
// DuckDB ships arguments as a one-row batch with a single `args` struct
// column, whose child fields are named `positional_<i>` and `named_<name>`.
// Both forms are flattened here so a function can ask for what it declared
// rather than reconstructing the layout.
//
// Every value is a one-row array, not a scalar, because an argument's type is
// not known until bind and Arrow is the only common vocabulary for it. The
// typed accessors are conveniences over that; `positional()` gives the array
// when a function needs the type or a value the accessors do not cover.
class Arguments {
public:
    Arguments() = default;

    // Parse the IPC-serialized `arguments` blob. Empty input yields no
    // arguments, which is what a zero-parameter function sees.
    static Arguments parse(const std::string& ipc_bytes);

    // The array for positional argument `index`, or null if the call did not
    // supply one.
    std::shared_ptr<arrow::Array> positional(size_t index) const;
    std::shared_ptr<arrow::Array> named(const std::string& name) const;

    size_t positional_count() const noexcept { return positional_.size(); }

    // Whether the call supplied `index`/`name` and left it SQL NULL.
    //
    // The typed accessors collapse "absent" and "present but null" into
    // nullopt, which is what a function with a default wants. A bind-time
    // check needs them apart: a missing argument takes the default, and an
    // explicit NULL is a caller error.
    bool positional_is_null(size_t index) const;
    bool named_is_null(const std::string& name) const;

    // The declared type of positional argument `index`, which is what a bind
    // needs to decide a result type. Null when absent.
    std::shared_ptr<arrow::DataType> positional_type(size_t index) const;

    // Typed reads of a constant argument. Absent, null, or a type that cannot
    // be converted yields nullopt rather than throwing — a function usually
    // has a sensible default and the alternative is a throw per call site.
    std::optional<int64_t> const_int64(size_t index) const;
    std::optional<double> const_double(size_t index) const;
    std::optional<std::string> const_string(size_t index) const;
    std::optional<bool> const_bool(size_t index) const;

    std::optional<int64_t> named_int64(const std::string& name) const;
    std::optional<std::string> named_string(const std::string& name) const;
    std::optional<double> named_double(const std::string& name) const;

private:
    std::vector<std::shared_ptr<arrow::Array>> positional_;
    std::vector<std::shared_ptr<arrow::Field>> positional_fields_;
    std::map<std::string, std::shared_ptr<arrow::Array>> named_;
};

// Serialize scan arguments for a catalog table's scan function.
//
// The flat form the `arguments` field carries: positional values become
// columns named `arg_0`, `arg_1`, …, and named ones keep their own name. A
// scan that takes none serializes an empty *schema*, not an empty batch — the
// consumer distinguishes them.
std::string serialize_scan_arguments(
    const std::vector<std::shared_ptr<arrow::Array>>& positional,
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& named = {});

}  // namespace vgi
