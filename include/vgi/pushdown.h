// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/record_batch.h>

namespace vgi {

// One pushed-down predicate, as the worker sees it.
//
// A structural view rather than an evaluable one: a function usually wants to
// know *that* a column is constrained (to skip a partition, narrow a request,
// or report what it was told) rather than to evaluate the predicate itself,
// which `PushdownFilters::apply` already does.
struct Filter {
    enum class Kind { Constant, In, IsNull, IsNotNull, JoinKeys, Other };

    Kind kind = Kind::Other;
    std::string column_name;
    // For Constant: one of eq/ne/lt/le/gt/ge.
    std::string op;
};

// Integer bounds on a column implied by the comparison filters on it.
struct ColumnBounds {
    std::optional<int64_t> min;
    std::optional<int64_t> max;
};

// The predicates the engine pushed into this scan.
//
// The wire form is a one-row batch whose first column is a JSON filter tree
// and whose remaining columns are the constant *values* the tree references by
// index. Values ride as Arrow columns rather than inside the JSON so they keep
// their exact type — a decimal or a timestamp survives, where a JSON number
// would not.
class PushdownFilters {
public:
    // Parse the IPC filter blob. Empty input yields no filters, which is what
    // a scan with nothing pushed into it sees.
    //
    // `join_key_batches` are the side batches a `join_keys` filter refers to —
    // DuckDB turns an `IN (…)` list, and a semi-join's build side, into a
    // filter that names a column in them rather than carrying the values
    // inline. Without them such a filter has no values at all.
    static PushdownFilters parse(const std::string& ipc_bytes,
                                 const std::vector<std::string>& join_key_batches = {});

    bool empty() const noexcept { return filters_.empty(); }

    // Every filter, flattened out of any and/or nesting.
    const std::vector<Filter>& filters() const noexcept { return filters_; }
    // Those naming `column`.
    std::vector<Filter> column_filters(const std::string& column) const;
    // Whether any predicate constrains `column`.
    bool has_filter_for_column(const std::string& column) const;
    // Every column any predicate names, deduplicated and sorted.
    std::vector<std::string> filtered_columns() const;

    // Integer bounds implied for `column`, if any comparison constrains it.
    ColumnBounds column_bounds(const std::string& column) const;

    // The discrete set of values `column` can take, or null when the
    // predicates do not enumerate one.
    //
    // This is the partition-pruning accessor: a scan that owns one file per
    // value answers the whole query by opening only these. Null means "cannot
    // enumerate", never "no rows" — a caller that treated it as an empty set
    // would drop every row.
    //
    // `=` and `IN` enumerate. A disjunction does too, but only when *every*
    // branch pins this same column to discrete values: one branch's values
    // would be an unsafe subset, since a pruning caller would then skip the
    // other branches' rows.
    std::shared_ptr<arrow::Array> column_values(const std::string& column) const;

    // Apply every filter to `batch`, returning the surviving rows.
    //
    // Best effort by design: a filter this cannot evaluate is skipped rather
    // than failing the scan, because pushdown is an optimization and the
    // engine re-checks the predicate itself. Dropping a row it should have
    // kept would be a wrong answer; keeping one it could have dropped is only
    // slower.
    std::shared_ptr<arrow::RecordBatch> apply(
        const std::shared_ptr<arrow::RecordBatch>& batch) const;

    // The filters as the reference implementations' `repr`, `(none)` when
    // there are none: `PushdownFilters([ConstantFilter(n < 5000)])`.
    //
    // Separate from `format`, which renders the predicate as SQL. This one
    // names the filter *kinds*, which is what a diagnostic fixture asserts on
    // when what it is checking is which shape the engine pushed.
    std::string format_repr() const;

    // The filters as SQL-ish text, `(none)` when there are none.
    //
    // For diagnostics and for fixtures that report what they were told. The
    // rendering matches the Python reference exactly — strings single-quoted,
    // booleans `True`/`False`, an IN list of more than 20 values collapsed to
    // a count — because the tests compare the string.
    std::string format() const;

    // The parsed filter tree. Public only so the implementation's free
    // helpers can name it; it is not part of the SDK's surface.
    struct Spec;

private:
    std::shared_ptr<arrow::Array> values_for(const Spec& spec) const;
    // Top-level specs naming `column`, plus the children of a top-level `and`
    // over it — the shape DuckDB pushes an `IN (…)` in when it also derives
    // range bounds. One level only, matching the reference implementation.
    std::vector<std::shared_ptr<Spec>> column_specs(const std::string& column) const;
    std::shared_ptr<arrow::Array> or_column_values(const Spec& spec,
                                                   const std::string& column) const;

    std::vector<Filter> filters_;
    std::vector<std::shared_ptr<Spec>> specs_;
    std::vector<std::shared_ptr<arrow::Array>> values_;
    std::map<std::string, std::shared_ptr<arrow::Array>> join_keys_;
};

}  // namespace vgi
