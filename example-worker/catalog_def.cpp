// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The declarative example catalog: schemas, tables and views.
//
// A VGI table is a name bound to a table function. Declaring one here is what
// turns `SELECT * FROM ex.data.cacheable_numbers` into a call to the
// `cacheable_numbers` function — the table has no storage of its own.

#include <string>
#include <vector>

#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <vgi/worker.h>

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
                            std::shared_ptr<arrow::Schema> schema) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.scan_function = std::move(scan_function);
    table.columns = std::move(schema);
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
    data.tables.push_back(
        backed_by("cacheable_numbers", "cacheable_numbers", columns({{"n", arrow::int64()}})));
    data.tables.push_back(
        backed_by("cache_nonce", "cache_nonce", columns({{"nonce", arrow::int64()}})));
    data.tables.push_back(
        backed_by("cache_no_store", "cache_no_store", columns({{"n", arrow::int64()}})));
    data.tables.push_back(
        backed_by("cache_scoped_txn", "cache_scoped_txn", columns({{"n", arrow::int64()}})));
    data.tables.push_back(
        backed_by("cache_bench", "cache_bench", columns({{"n", arrow::int64()}})));
    data.tables.push_back(backed_by("cache_multicol", "cache_multicol",
                                    columns({{"a", arrow::int64()},
                                             {"b", arrow::int64()},
                                             {"c", arrow::int64()}})));
    data.tables.push_back(
        backed_by("cache_big", "cache_big", columns({{"n", arrow::int64()}})));
    data.tables.push_back(backed_by("cache_revalidatable", "cache_revalidatable",
                                    columns({{"nonce", arrow::int64()}})));
    data.tables.push_back(backed_by("ten_thousand_table", "ten_thousand_table",
                                    columns({{"n", arrow::int64()}})));
}

}  // namespace example
