// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The catalogs that exercise the catalog_contents RPC (the whole catalog in
// one call). They mirror vgi-python's vgi/_test_fixtures/catalog_contents.py --
// the cross-SDK contract -- and vgi-go's examples/catalog_contents, and are
// driven by vgi/test/sql/integration/catalog/catalog_contents*.test.
//
// The same static two-schema catalog is served under three names, differing
// only in how they answer catalog_contents:
//
//   contents_probe   advertises supports_catalog_contents and serves it (the
//                    default: version-frozen, no etag) -- loaded in one RPC.
//   contents_broken  advertises it, but catalog_contents fails: the client must
//                    fall back to catalog_schemas + the per-schema RPCs.
//   contents_legacy  does not advertise it, like an older worker: the client
//                    must never call catalog_contents.
//
// Three DDL-capable in-memory catalogs (version not frozen) advertise it too.
// Every ATTACH gets its own empty "main" schema, so tests sharing a worker
// never see each other's objects:
//
//   contents_memory  reports catalog_version 0 ("unknown") and no etag: the
//                    client's version-0 rule.
//   contents_reval   a cheap validator: the etag is "gen-<n>", n = the catalog
//                    version, bumped by every DDL; a matching if_none_match is
//                    answered not_modified without building anything.
//   contents_hash    no etag of its own, but the framework content hash: the
//                    snapshot is built on every call and its SHA-256 is the etag.
//
// The static catalog holds every kind the client seeds from catalog_contents
// (tables, a view, scalar / aggregate / table functions, scalar and table
// macros), split over "main" and "extra". Its functions are the example
// worker's double / vgi_sum / sequence, homed in each catalog, so they are
// reachable only through that catalog's own ATTACH.

#include <memory>
#include <stdexcept>
#include <string>

#include <arrow/array/builder_primitive.h>
#include <arrow/type.h>

#include <vgi/worker.h>

#include "registry.h"

namespace example {
namespace {

constexpr const char* kBrokenMessage = "contents_broken: catalog_contents deliberately fails";

std::shared_ptr<arrow::Array> int64_value(int64_t value) {
    arrow::Int64Builder builder;
    (void)builder.Append(value);
    return builder.Finish().ValueOrDie();
}

// A table backed by `sequence(count)`: column `n`, 0..count-1.
vgi::CatalogTable sequence_table(std::string name, int64_t count, std::string comment) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.scan_function = "sequence";
    table.columns = arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
    table.scan_arguments = vgi::serialize_scan_arguments({int64_value(count)});
    table.comment = std::move(comment);
    return table;
}

// The two-schema static catalog, served under `name`.
vgi::CatalogModel& static_catalog(vgi::Worker& worker, const std::string& name) {
    auto& catalog = worker.catalog(name);
    catalog.comment = "catalog_contents test catalog (" + name + ")";

    worker.register_scalar_in(name, "main", make_double());
    worker.register_aggregate_in(name, "main", make_vgi_sum());
    worker.register_table_in(name, "main", make_sequence());

    auto& main = catalog.schema("main");
    main.comment = "Every object kind";
    main.tables.push_back(sequence_table("ten", 10, "Integers 0..9"));
    vgi::CatalogView answer;
    answer.name = "answer";
    answer.definition = "SELECT 42 AS answer";
    answer.comment = "One row";
    main.views.push_back(std::move(answer));
    vgi::CatalogMacro triple;
    triple.name = "contents_triple";
    triple.parameters = {"x"};
    triple.definition = "x * 3";
    triple.comment = "Triple a value";
    main.macros.push_back(std::move(triple));
    vgi::CatalogMacro range;
    range.name = "contents_range";
    range.parameters = {"n"};
    range.definition = "SELECT * FROM range(n)";
    range.table_macro = true;
    range.comment = "Table macro over range(n)";
    main.macros.push_back(std::move(range));

    auto& extra = catalog.schema("extra");
    extra.comment = "A second schema, tables only";
    extra.tables.push_back(sequence_table("five", 5, "Integers 0..4"));
    return catalog;
}

// contents_reval's validator: "gen-<version>", answered not_modified before
// building anything when it matches.
vgi::CatalogContentsResult generation_etag(const vgi::CatalogContentsCall& call) {
    vgi::CatalogContentsResult result;
    result.etag = "gen-" + std::to_string(call.catalog_version);
    if (call.if_none_match && *call.if_none_match == *result.etag) {
        result.not_modified = true;
        return result;
    }
    result.schemas = call.contents();
    return result;
}

vgi::MemoryCatalogOptions memory_options(const std::string& name) {
    vgi::MemoryCatalogOptions options;
    options.name = name;
    options.comment = "catalog_contents test catalog (" + name + ")";
    return options;
}

}  // namespace

void register_catalog_contents(vgi::Worker& worker) {
    static_catalog(worker, "contents_probe");
    static_catalog(worker, "contents_broken").catalog_contents_handler =
        [](const vgi::CatalogContentsCall&) -> vgi::CatalogContentsResult {
        throw std::runtime_error(kBrokenMessage);
    };
    static_catalog(worker, "contents_legacy").supports_catalog_contents = false;

    auto memory = memory_options("contents_memory");
    memory.unversioned = true;
    worker.register_memory_catalog(std::move(memory));

    auto reval = memory_options("contents_reval");
    reval.catalog_contents_handler = generation_etag;
    worker.register_memory_catalog(std::move(reval));

    auto hash = memory_options("contents_hash");
    hash.catalog_contents_etag = vgi::CatalogContentsEtag::ContentHash;
    worker.register_memory_catalog(std::move(hash));
}

}  // namespace example
