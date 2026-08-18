// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace vgi {

// The catalog a worker advertises: what `ATTACH '<name>' (TYPE vgi)` binds to.
//
// A worker serves one primary catalog and may serve secondaries alongside it
// (the MetaWorker model), which is why the name is data rather than a
// compile-time constant — the example worker picks it from
// VGI_WORKER_CATALOG_NAME so one binary can stand in for several fixtures.
struct CatalogModel {
    std::string name = "main";
    std::string implementation_version;
    // Schemas within the catalog.  "main" always exists; a worker that
    // declares none gets it implicitly.
    std::vector<std::string> schemas{"main"};
    std::string source_url;
};

}  // namespace vgi
