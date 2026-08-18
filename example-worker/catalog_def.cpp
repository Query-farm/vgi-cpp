// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The declarative example catalog: schemas, tables and views.
//
// A VGI table is a name bound to a table function. Declaring one here is what
// turns `SELECT * FROM ex.data.cacheable_numbers` into a call to the
// `cacheable_numbers` function — the table has no storage of its own.

#include <string>
#include <vector>

#include <arrow/array/builder_primitive.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <vgi/worker.h>

#include "registry.h"

namespace example {
namespace {

std::shared_ptr<arrow::Schema> columns(
    std::vector<std::pair<std::string, std::shared_ptr<arrow::DataType>>> fields) {
    std::vector<std::shared_ptr<arrow::Field>> built;
    built.reserve(fields.size());
    for (auto& [name, type] : fields) {
        built.push_back(arrow::field(name, std::move(type), /*nullable=*/true));
    }
    return arrow::schema(std::move(built));
}

vgi::CatalogTable backed_by(std::string name, std::string scan_function,
                            std::shared_ptr<arrow::Schema> schema, std::string comment = {}) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.scan_function = std::move(scan_function);
    table.columns = std::move(schema);
    table.comment = std::move(comment);
    return table;
}

// A one-element array holding `value`, for a scan argument.
std::shared_ptr<arrow::Array> int64_arg(int64_t value) {
    arrow::Int64Builder builder;
    (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// A branch that scans `sequence(count)`.
vgi::CatalogBranch sequence_branch(int64_t count,
                                   std::optional<std::string> branch_filter = std::nullopt) {
    vgi::CatalogBranch branch;
    branch.function_name = "sequence";
    branch.scan_arguments = vgi::serialize_scan_arguments({int64_arg(count)});
    branch.branch_filter = std::move(branch_filter);
    return branch;
}

vgi::CatalogTable multi_branch(std::string name, std::vector<vgi::CatalogBranch> branches) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.columns = columns({{"n", arrow::int64()}});
    // No single scan function: the branches are the sources, and declaring one
    // here as well would be a second, contradictory answer.
    table.branches = std::move(branches);
    table.inline_scan = false;
    return table;
}

}  // namespace

void declare_catalog(vgi::Worker& worker) {
    // Settings the catalog introduces. `SET greeting = 'Bonjour'` only works
    // because they are declared here; a function then names the ones it reads
    // in `required_settings` to have the values forwarded.
    worker.catalog().settings = {
        {"vgi_verbose_mode", "Enable verbose output", arrow::boolean()},
        {"greeting", "Greeting prefix", arrow::utf8()},
        {"multiplier", "Integer multiplier", arrow::int64()},
        {"threshold", "Floating-point threshold", arrow::float64()},
        {"config", "Free-form configuration string", arrow::utf8()},
    };

    // A field marked `redact` is one the engine must keep out of logs and
    // error text; declaring the schema is what carries that marker across.
    const auto redacted = arrow::key_value_metadata({"redact"}, {"true"});
    worker.catalog().secret_types = {
        {"vgi_example", "Example VGI secret for testing",
         arrow::schema({
             arrow::field("secret_string", arrow::utf8(), true)->WithMetadata(redacted),
             arrow::field("api_key", arrow::utf8(), true)->WithMetadata(redacted),
             arrow::field("port", arrow::int32(), true),
             arrow::field("use_ssl", arrow::boolean(), true),
             arrow::field("timeout", arrow::float64(), true),
         })},
    };

    auto& data = worker.catalog().schema("data");

    data.tables.push_back(
        backed_by("numbers", "sequence", columns({{"n", arrow::int64()}})));
    data.tables.push_back(backed_by("cacheable_numbers", "cacheable_numbers",
                                    columns({{"n", arrow::int64()}}),
                                    "Cacheable 10-row result advertising vgi.cache.ttl"));
    data.tables.push_back(
        backed_by("cache_nonce", "cache_nonce", columns({{"nonce", arrow::int64()}}),
                  "One-row cacheable result whose value changes per real invocation"));
    data.tables.push_back(
        backed_by("cache_no_store", "cache_no_store", columns({{"n", arrow::int64()}}),
                  "Advertises vgi.cache.no_store — must never be cached"));
    data.tables.push_back(
        backed_by("cache_scoped_txn", "cache_scoped_txn", columns({{"n", arrow::int64()}}),
                  "Advertises vgi.cache.scope=transaction"));
    // Uncommented on purpose: `cache_bench` is a scaling fixture called
    // directly, and `table/comments.test` pins the commented set exactly.
    data.tables.push_back(
        backed_by("cache_bench", "cache_bench", columns({{"v", arrow::int64()}})));
    data.tables.push_back(backed_by("cache_multicol", "cache_multicol",
                                    columns({{"a", arrow::int64()},
                                             {"b", arrow::int64()},
                                             {"c", arrow::int64()}}),
                                    "Multi-column cacheable result (projection-coverage reuse)"));
    data.tables.push_back(
        backed_by("cache_big", "cache_big", columns({{"n", arrow::int64()}}),
                  "Large multi-batch cacheable result (advertises vgi.cache.ttl)"));
    data.tables.push_back(
        backed_by("cache_revalidatable", "cache_revalidatable",
                  columns({{"nonce", arrow::int64()}}),
                  "Always-revalidate result (304 not_modified reuses stored bytes)"));
    data.tables.push_back(
        backed_by("cache_whoami", "cache_whoami", columns({{"who", arrow::utf8()}}),
                  "Cacheable result echoing the caller's auth principal (identity-scoped)"));
    data.tables.push_back(backed_by("ten_thousand_table", "ten_thousand_table",
                                    columns({{"n", arrow::int64()}})));
    data.tables.push_back(
        backed_by("cache_parallel", "cache_parallel", columns({{"v", arrow::int64()}})));

    // Multi-branch tables: one logical relation stitched from several scans.
    data.tables.push_back(
        multi_branch("multi_branch_numbers", {sequence_branch(50), sequence_branch(50)}));
    data.tables.push_back(multi_branch("multi_branch_filtered_numbers",
                                       {sequence_branch(100, "n < 50"),
                                        sequence_branch(100, "n >= 50")}));
    data.tables.push_back(multi_branch("multi_branch_empty", {}));

    data.tables.push_back(
        backed_by("cache_poison", "cache_poison", columns({{"n", arrow::int64()}}),
                  "Cacheable first batch then a mid-stream error (never-partial check)"));
    data.tables.push_back(
        backed_by("cache_filtered", "cache_filtered", columns({{"n", arrow::int64()}}),
                  "Cacheable sequence with static filter pushdown (filter_bytes keying)"));
    data.tables.push_back(
        backed_by("cache_ordered", "cache_ordered", columns({{"n", arrow::int64()}}),
                  "Multi-worker order-sensitive cacheable result (batch_index; parallel "
                  "capture, ordered serve)"));
    data.tables.push_back(
        backed_by("cache_external_fail", "cache_external_fail", columns({{"n", arrow::int64()}}),
                  "Cacheable first batch then an unresolvable external-location pointer"));
    for (const char* name : {"cache_partitioned", "cache_partition_scope",
                             "cache_partition_parallel"}) {
        data.tables.push_back(backed_by(name, name, columns({{"n", arrow::int64()}})));
    }

    // Time travel is the point: each version resolves to its own scan
    // arguments, so an `AT` clause keys a distinct cache entry.
    {
        vgi::CatalogTable versioned;
        versioned.name = "cache_versioned";
        versioned.columns = columns({{"v", arrow::int64()}});
        versioned.scan_function = "cache_versioned_scan";
        versioned.scan_arguments = vgi::serialize_scan_arguments({int64_arg(3)});
        versioned.comment = "Version-specific cacheable rows (AT-keyed cache isolation)";
        for (int64_t version = 1; version <= 3; ++version) {
            vgi::TimeTravelVersion entry;
            entry.version = version;
            entry.columns = versioned.columns;
            entry.scan_function = "cache_versioned_scan";
            entry.scan_arguments = vgi::serialize_scan_arguments({int64_arg(version)});
            versioned.time_travel.push_back(std::move(entry));
        }
        data.tables.push_back(std::move(versioned));
    }
    // A table whose scan is `sequence` with a fixed argument, and two views
    // over it. Declared here rather than as functions because that is what
    // they are to the engine — a name it resolves, not a call it makes.
    // Declares no statistics of its own: the optimizer has to fall back to
    // the scan function's, which is exactly what this table is here to pin.
    {
        auto funny = backed_by("funny_numbers", "sequence", columns({{"n", arrow::int64()}}),
                               "123456 integers; stats served by the sequence function, "
                               "not the table");
        funny.scan_arguments = vgi::serialize_scan_arguments({int64_arg(123456)});
        funny.cardinality = 123456;
        data.tables.push_back(std::move(funny));
    }

    {
        auto large = backed_by("large_sequence", "sequence", columns({{"n", arrow::int64()}}));
        large.scan_arguments = vgi::serialize_scan_arguments({int64_arg(1000000)});
        large.cardinality = 1000000;
        data.tables.push_back(std::move(large));
    }

    {
        // Inlined, and that is what makes AT work here: `catalog_table_get`
        // already carries the AT clause, so the record it returns can name the
        // version's own scan arguments. Left un-inlined the engine never asked
        // for the scan separately, and every AT ran version 3.
        vgi::CatalogTable versioned;
        versioned.name = "versioned_data";
        versioned.time_travel = versioned_data_versions();
        versioned.columns = versioned.time_travel.back().columns;
        versioned.scan_function = "versioned_data_scan";
        versioned.scan_arguments = versioned.time_travel.back().scan_arguments;
        versioned.inline_scan = true;
        versioned.comment =
            "Versioned data table demonstrating time travel with schema evolution";
        data.tables.push_back(std::move(versioned));
    }

    for (auto& table : rff_tables()) data.tables.push_back(std::move(table));

    data.views.push_back({"first_ten", "SELECT * FROM sequence(10)", std::nullopt});
    data.views.push_back(
        {"even_numbers", "SELECT * FROM sequence(100) WHERE n % 2 = 0", std::nullopt});

    data.tables.push_back(
        backed_by("cache_projection", "cache_projection",
                  columns({{"a", arrow::int64()},
                           {"b", arrow::int64()},
                           {"c", arrow::int64()}}),
                  "Projection-pushdown cacheable result (SELECT a vs b are distinct keys)"));
}

}  // namespace example
