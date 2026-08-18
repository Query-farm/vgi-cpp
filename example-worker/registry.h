// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Registration entry points, one per fixture group.
//
// Grouped rather than registered from main() so each file owns the functions
// it defines and main() stays a list of groups — the same shape as the Rust
// example worker's module-per-group `register` functions.

#pragma once

#include <vector>

#include <vgi/catalog.h>

namespace vgi {
class Worker;
}

namespace example {

void register_arithmetic(vgi::Worker& worker);
void register_strings(vgi::Worker& worker);
void register_seeded(vgi::Worker& worker);
void register_geo(vgi::Worker& worker);
void register_types(vgi::Worker& worker);
void register_same_name(vgi::Worker& worker);
void register_table_functions(vgi::Worker& worker);
void register_table_in_out(vgi::Worker& worker);
void register_aggregates(vgi::Worker& worker);
void register_buffering(vgi::Worker& worker);
void register_cache(vgi::Worker& worker);
void register_more_tables(vgi::Worker& worker);
void register_sum_all_columns(vgi::Worker& worker);
void register_settings_fixtures(vgi::Worker& worker);
void register_format_fixtures(vgi::Worker& worker);
void register_settings_tables(vgi::Worker& worker);
void register_secret_fixtures(vgi::Worker& worker);
void register_series(vgi::Worker& worker);
void register_filter_fixtures(vgi::Worker& worker);
void register_logging_fixtures(vgi::Worker& worker);
void register_global_probes(vgi::Worker& worker);
void register_copy_to(vgi::Worker& worker);
void register_copy_from(vgi::Worker& worker);
void register_cached(vgi::Worker& worker);
void register_more_aggregates(vgi::Worker& worker);
void register_generators(vgi::Worker& worker);
void register_secret_scalars(vgi::Worker& worker);
void register_secret_table_in_out(vgi::Worker& worker);
void register_same_name_exchange(vgi::Worker& worker);
void register_substream_finalize(vgi::Worker& worker);
void register_unnest_tensor_rows(vgi::Worker& worker);
void register_cancellable_inout(vgi::Worker& worker);
void register_large_state(vgi::Worker& worker);
void register_source_shapes(vgi::Worker& worker);
void register_blended(vgi::Worker& worker);
void register_versioned(vgi::Worker& worker);
void register_static_scans(vgi::Worker& worker);
void register_versioned_tables_scans(vgi::Worker& worker);
void register_window_aggregates(vgi::Worker& worker);
void register_percentile(vgi::Worker& worker);
void register_same_name_aggregates(vgi::Worker& worker);
void register_nest_tensor(vgi::Worker& worker);
void register_rff(vgi::Worker& worker);
void register_more_cache(vgi::Worker& worker);
void register_cache_partitions(vgi::Worker& worker);
void register_partitioned(vgi::Worker& worker);
void register_partition_broken(vgi::Worker& worker);
void register_accumulate(vgi::Worker& worker);
void register_cached_scalars(vgi::Worker& worker);
void register_projection_repro(vgi::Worker& worker);
void register_transaction_storage(vgi::Worker& worker);
std::vector<vgi::CatalogTable> rff_tables();
std::vector<vgi::TimeTravelVersion> versioned_data_versions();
// Declares the catalog's tables and views. Runs after the function
// registrations, since a table names the function that scans it.
void declare_catalog(vgi::Worker& worker);
// Declares the catalogs beyond `example` this one binary also serves.
void register_extra_catalogs(vgi::Worker& worker);
// Shapes the `versioned_tables` catalog model, before any function registers.
void declare_versioned_tables(vgi::CatalogModel& catalog);

}  // namespace example
