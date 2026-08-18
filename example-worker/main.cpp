// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The VGI example worker: the fixture set the integration suite in
// ~/Development/vgi runs against.
//
// One binary serves whichever catalog VGI_WORKER_CATALOG_NAME names, mirroring
// vgi-rust's example worker so the same wrapper scripts drive either.

#include <cstdlib>
#include <string>

#include <vgi/worker.h>

#include "registry.h"

int main(int argc, char** argv) {
    vgi::Worker worker;

    vgi::CatalogModel catalog;
    const char* name = std::getenv("VGI_WORKER_CATALOG_NAME");
    catalog.name = (name && *name) ? name : "example";
    worker.set_catalog(std::move(catalog));

    example::register_arithmetic(worker);
    example::register_strings(worker);
    example::register_seeded(worker);
    example::register_geo(worker);
    example::register_types(worker);
    example::register_same_name(worker);
    example::register_table_functions(worker);
    example::register_table_in_out(worker);
    example::register_aggregates(worker);
    example::register_buffering(worker);
    example::register_cache(worker);
    example::register_more_tables(worker);
    example::register_sum_all_columns(worker);
    example::register_settings_fixtures(worker);
    example::register_settings_tables(worker);
    example::register_secret_fixtures(worker);
    example::register_series(worker);
    example::register_filter_fixtures(worker);
    example::declare_catalog(worker);

    // Serves until the engine disconnects; does not return.
    worker.run(argc, argv);
}
