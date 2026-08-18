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
    // `example` is the composite fixture: it serves the side catalogs beside
    // its own, because several tests attach two of them from one binary. A
    // wrapper that names one of the other fixtures serves that one alone, and
    // its discovery answer has to list exactly it.
    const bool composite = catalog.name == "example";
    const std::string catalog_name = catalog.name;
    if (catalog.name == "versioned") {
        // The versioned fixture: three data versions, one implementation, and
        // an advertised range. ATTACH resolves against these.
        catalog.implementation_version = "1.0.0";
        catalog.data_version_spec = ">=1.0.0,<2.0.0";
        catalog.supported_data_versions = {"1.0.0", "1.1.0", "1.2.0"};
        catalog.default_data_version = "1.2.0";
        catalog.comment =
            "Example catalog demonstrating data_version_spec validation and cookie stickiness";
    } else if (catalog.name == "versioned_tables") {
        example::declare_versioned_tables(catalog);
    }
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
    example::register_format_fixtures(worker);
    example::register_settings_tables(worker);
    example::register_secret_fixtures(worker);
    example::register_series(worker);
    example::register_filter_fixtures(worker);
    example::register_logging_fixtures(worker);
    example::register_global_probes(worker);
    example::register_copy_to(worker);
    example::register_copy_from(worker);
    example::register_cached(worker);
    example::register_more_aggregates(worker);
    example::register_generators(worker);
    example::register_secret_scalars(worker);
    example::register_secret_table_in_out(worker);
    example::register_same_name_exchange(worker);
    example::register_substream_finalize(worker);
    example::register_unnest_tensor_rows(worker);
    example::register_cancellable_inout(worker);
    example::register_large_state(worker);
    example::register_source_shapes(worker);
    example::register_blended(worker);
    example::register_versioned(worker);
    example::register_static_scans(worker);
    example::register_window_aggregates(worker);
    example::register_percentile(worker);
    example::register_same_name_aggregates(worker);
    example::register_nest_tensor(worker);
    example::register_rff(worker);
    example::register_more_cache(worker);
    example::register_cache_partitions(worker);
    example::register_partitioned(worker);
    example::register_partition_broken(worker);
    example::register_cached_scalars(worker);
    example::register_transaction_storage(worker);
    example::declare_catalog(worker);

    // A side catalog is registered when this binary is the composite fixture,
    // and also when the wrapper named that catalog outright — in which case it
    // *is* the primary and there is nothing else to serve.
    if (composite || catalog_name == "versioned_tables") {
        example::register_versioned_tables_scans(worker);
    }
    if (composite || catalog_name == "attach_options") {
        example::register_attach_options(worker);
    }
    if (composite) {
        example::register_accumulate(worker);
        example::register_projection_repro(worker);
        example::register_extra_catalogs(worker);
    }

    // Serves until the engine disconnects; does not return.
    worker.run(argc, argv);
}
