// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/type.h>

namespace vgi {

// A table the catalog advertises.
//
// A VGI table is not storage: it is a *name* bound to a table function that
// produces its rows. `SELECT * FROM cat.data.numbers` resolves the table, reads
// its scan function, and calls that. The columns declared here are what the
// planner types the query against before any scan runs, so they have to match
// what the scan actually emits.
struct CatalogTable {
    std::string name;
    std::shared_ptr<arrow::Schema> columns;
    // The table function that produces the rows.
    std::string scan_function;
    // IPC-serialized arguments handed to that function. Empty for a scan that
    // takes none.
    std::string scan_arguments;

    std::optional<std::string> comment;
    std::optional<int64_t> cardinality;
    std::vector<std::pair<std::string, std::string>> tags;

    // Whether the scan function travels inside the table record.
    //
    // Inlined, the engine skips `catalog_table_scan_function_get` entirely —
    // one fewer round trip per query. Not inlined, it asks, which is what a
    // table whose scan depends on the query (a time-travel AT clause, say)
    // needs.
    bool inline_scan = true;
};

// A view: a name bound to SQL the engine expands.
struct CatalogView {
    std::string name;
    // The SQL text. Called `definition` on the wire; named to match, since a
    // mismatch here is rejected rather than ignored.
    std::string definition;
    std::optional<std::string> comment;
};

// A schema and everything declared in it.
struct CatalogSchema {
    std::string name = "main";
    std::vector<CatalogTable> tables;
    std::vector<CatalogView> views;
};

// The catalog a worker advertises: what `ATTACH '<name>' (TYPE vgi)` binds to.
//
// A worker serves one primary catalog and may serve secondaries alongside it,
// which is why the name is data rather than a compile-time constant — the
// example worker picks it from VGI_WORKER_CATALOG_NAME so one binary can stand
// in for several fixtures.
struct CatalogModel {
    std::string name = "main";
    std::string implementation_version;
    std::string source_url;

    // Schemas declared up front. Registering a function in a schema adds it
    // here too, so a worker with only functions never has to list them.
    //
    // Held indirectly so that `schema()` can hand out a reference that
    // survives a later registration. A vector of values would reallocate, and
    // the documented usage — take a schema, register more functions, then add
    // tables to it — would dangle.
    std::vector<std::unique_ptr<CatalogSchema>> schemas;

    CatalogModel();

    // The schema of that name, creating it if absent. The reference stays
    // valid for the life of the model.
    CatalogSchema& schema(const std::string& schema_name);
    const CatalogSchema* find_schema(const std::string& schema_name) const;

    // Every declared schema name, in declaration order.
    std::vector<std::string> schema_names() const;
};

}  // namespace vgi
