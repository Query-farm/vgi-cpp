// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vgi {

// One end of a column's value range.
//
// Four types rather than one because DuckDB seeds a range from the value's own
// type; a bound handed over as text would not prune a numeric filter.
class StatValue {
public:
    enum class Kind { Int64, Float64, Utf8, Binary };

    static StatValue integer(int64_t value);
    static StatValue floating(double value);
    static StatValue text(std::string value);
    // Raw bytes — WKB for a GEOMETRY column, which the engine reinterprets
    // from the BLOB it arrives as.
    static StatValue bytes(std::string value);

    Kind kind() const noexcept { return kind_; }
    int64_t integer_value() const noexcept { return integer_; }
    double floating_value() const noexcept { return floating_; }
    const std::string& string_value() const noexcept { return string_; }

private:
    Kind kind_ = Kind::Int64;
    int64_t integer_ = 0;
    double floating_ = 0;
    std::string string_;
};

// What the optimizer is told about one column.
//
// Bounds are a promise: a scan that reports [0, 99] and then emits 100 has
// already had the row pruned away by a filter that trusted the promise, so a
// function that cannot bound a column must not report it at all.
struct ColumnStatistics {
    std::string column_name;
    StatValue min;
    StatValue max;
    bool has_null = false;
    bool has_not_null = true;
    std::optional<int64_t> distinct_count;
    std::optional<bool> contains_unicode;
    std::optional<uint64_t> max_string_length;
};

// The IPC bytes the engine reads statistics from.
//
// An empty list serializes to a bare empty schema rather than to a zero-row
// batch, which is how "no statistics" is spelled on this wire.
std::string serialize_column_statistics(const std::vector<ColumnStatistics>& statistics);

}  // namespace vgi
