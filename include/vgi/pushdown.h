// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
    static PushdownFilters parse(const std::string& ipc_bytes);

    bool empty() const noexcept { return filters_.empty(); }

    // Every filter, flattened out of any and/or nesting.
    const std::vector<Filter>& filters() const noexcept { return filters_; }
    // Those naming `column`.
    std::vector<Filter> column_filters(const std::string& column) const;

    // Integer bounds implied for `column`, if any comparison constrains it.
    ColumnBounds column_bounds(const std::string& column) const;

    // Apply every filter to `batch`, returning the surviving rows.
    //
    // Best effort by design: a filter this cannot evaluate is skipped rather
    // than failing the scan, because pushdown is an optimization and the
    // engine re-checks the predicate itself. Dropping a row it should have
    // kept would be a wrong answer; keeping one it could have dropped is only
    // slower.
    std::shared_ptr<arrow::RecordBatch> apply(
        const std::shared_ptr<arrow::RecordBatch>& batch) const;

    // The parsed filter tree. Public only so the implementation's free
    // helpers can name it; it is not part of the SDK's surface.
    struct Spec;

private:
    std::vector<Filter> filters_;
    std::vector<std::shared_ptr<Spec>> specs_;
    std::vector<std::shared_ptr<arrow::Array>> values_;
};

}  // namespace vgi
