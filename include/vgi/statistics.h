// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
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
// already had the row pruned away by a filter that trusted the promise. So
// they are optional, and every default here is the one that promises nothing:
// an unset bound travels as null, and a column is assumed to hold nulls until
// someone says otherwise. Reporting a bound you cannot stand behind — or
// leaving a default in place that happens to say [0, 0] and never null — is
// how a scan comes back with no rows and no error.
struct ColumnStatistics {
    std::string column_name;
    std::optional<StatValue> min;
    std::optional<StatValue> max;
    bool has_null = true;
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
