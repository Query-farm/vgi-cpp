# Filtered scan planning (protocol 3.2)

A multi-branch (or redirected) table normally answers
`catalog_table_scan_branches_get` at bind time, before the query's `WHERE`
exists, so it has to return every file on every scan. Protocol 3.2 lets such a
table see the query's predicates first, and lets the engine prune the branches
it returns. The normative spec is vgi-python's
[`docs/protocol/scan-planning.md`](https://github.com/Query-farm/vgi-python/blob/main/docs/protocol/scan-planning.md).

Every mechanism is pruning only: the engine re-applies every predicate to the
rows the remaining branches produce, so ignoring the predicates, or returning
more than needed, stays correct. Omitting a branch or file that holds a
matching row does not.

## Planning with the query's predicates

```cpp
#include <vgi/worker.h>

vgi::CatalogTable events;
events.name = "events";
events.columns = arrow::schema({arrow::field("event_date", arrow::date32()),
                                arrow::field("n", arrow::int64())});
events.scan_planning = vgi::kScanPlanningFiltered;  // default kScanPlanningBind
events.inline_scan = false;                          // required for "filtered"
events.plan_scan_branches = [](const vgi::ScanPlanningRequest& request) {
    std::vector<vgi::CatalogBranch> branches;
    for (const auto& file : manifest()) {
        // file.stats: std::vector<vgi::ColumnStatistics> (min/max/has_null/...)
        if (request.pushdown_filters && !request.pushdown_filters->may_match(file.stats))
            continue;
        branches.push_back(read_parquet_branch(file));
    }
    return branches;  // empty is legal for a filtered table: no rows
};
```

- `scan_planning = kScanPlanningFiltered` makes the engine call
  `catalog_table_scan_branches_get` once per scan, after collecting that
  scan's predicates, and send them as `pushdown_filters`. A filtered table
  must set `inline_scan = false`; the worker refuses to start otherwise.
- `plan_scan_branches` answers every `catalog_table_scan_branches_get` for the
  table. `request.pushdown_filters` is a `vgi::ScanPlanningFilters` (absent when
  no predicate was collected): `raw()` is the Filter Encoding v2 snapshot,
  `decode()` parses it once against the table's declared columns into the
  usual `vgi::PushdownFilters`, and `may_match(statistics)` returns false only
  when the predicates provably exclude every row of a candidate described by
  min/max/null statistics. `request.declared_branches()` returns what the table
  would have served without the hook.
- Without a hook, the declared `branches` (or the single scan function) are
  served and the predicates ignored, exactly as in 3.1.

## Branch statistics

`CatalogBranch::row_count` (an estimate, never trusted as exact) and
`CatalogBranch::column_statistics` travel on each branch, for both `"bind"` and
`"filtered"` tables. The engine drops a branch whose statistics prove that no
row can satisfy the query, so statistics must be conservative claims about
every row: leave a column out rather than guess. A DATE or TIMESTAMP bound is
sent as its ISO text (`vgi::StatValue::text("2026-10-03")`), which the engine
casts to the column's type, and which `may_match` casts the same way.

The example worker's `data.planned_days`, `planned_days_stats` and
`planned_days_bf` tables exercise worker-side pruning, statistics pruning and
`branch_filter` contradiction respectively.
