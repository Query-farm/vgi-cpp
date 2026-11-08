// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

// Protocol 3.2 filtered scan planning (vgi-python docs/protocol/scan-planning.md).
//
// A table declaring `CatalogTable::scan_planning = kScanPlanningFiltered` has
// its scan redirect resolved per scan: the engine defers
// `catalog_table_scan_branches_get` until it has collected the query's
// predicates on the table and sends them as `pushdown_filters`. The table's
// `CatalogTable::plan_scan_branches` hook receives them as a
// ScanPlanningFilters and may return fewer branches, or fewer files per
// branch. Every predicate is advisory: the engine re-applies all of them to
// the rows the branches produce, so a worker may ignore any predicate, but it
// must never drop a branch or file that could hold a matching row.

#include <memory>
#include <string>
#include <vector>

#include <arrow/type.h>

#include "vgi/pushdown.h"
#include "vgi/statistics.h"

namespace vgi {

// `TableInfo.scan_planning` values.
//
// "bind" (the default) is 3.1 behaviour: the engine resolves the scan redirect
// at bind time, with no predicates. "filtered" resolves it per scan with the
// query's predicates; such a table must not inline its scan
// (`CatalogTable::inline_scan = false`), and may answer with zero branches,
// meaning the scan produces no rows.
inline constexpr const char* kScanPlanningBind = "bind";
inline constexpr const char* kScanPlanningFiltered = "filtered";

// The query predicates a filtered `catalog_table_scan_branches_get` carries.
//
// `raw()` is the wire value: an Arrow IPC stream holding one VGI Filter
// Encoding v2 snapshot batch whose column references index the table's
// declared columns. `decode()` parses it against those columns once, on first
// use, so a planning hook that never looks at the predicates never pays for
// parsing them.
class ScanPlanningFilters {
public:
    ScanPlanningFilters(std::string raw, std::shared_ptr<arrow::Schema> table_columns);

    const std::string& raw() const noexcept { return raw_; }
    // The schema the column references index: the table's declared columns,
    // as the AT-resolved version declares them.
    const std::shared_ptr<arrow::Schema>& table_columns() const noexcept { return table_columns_; }

    // The predicates, decoded against `table_columns()` and cached. Throws
    // std::invalid_argument when the payload is not a valid snapshot.
    const PushdownFilters& decode() const;

    // False only when the predicates provably exclude every row of a candidate
    // (a file, a partition, a branch) that `statistics` describes the way
    // `CatalogBranch::column_statistics` does. See PushdownFilters::may_match.
    bool may_match(const std::vector<ColumnStatistics>& statistics) const;

private:
    std::string raw_;
    std::shared_ptr<arrow::Schema> table_columns_;
    mutable std::shared_ptr<const PushdownFilters> decoded_;
};

}  // namespace vgi
