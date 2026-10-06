// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// A DDL-capable in-memory catalog: see MemoryCatalogOptions in vgi/catalog.h.
// The C++ counterpart of vgi-go's MemoryCatalog (and vgi-python's
// InMemoryCatalog as the contents_* fixtures use it).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <arrow/record_batch.h>

#include "vgi/catalog.h"

namespace vgi {

class MemoryCatalog {
public:
    explicit MemoryCatalog(MemoryCatalogOptions options);

    const std::string& name() const noexcept { return options_.name; }

    // Whether `attach_opaque_data` is one of this catalog's attachments.
    // Attachments are `<name>\0\0\0<token>\0` -- the five-field shape the
    // dispatcher's own seals take, with the catalog first -- so the router
    // can tell whose a request is from its first field alone.
    static std::optional<std::string> catalog_of(const std::string& attach_opaque_data);

    // Serve `method` with `params` (the call's parameter batch). Returns the
    // payload for a method that answers one, nullopt for a void method, and
    // throws for a method a memory catalog does not serve.
    std::optional<std::shared_ptr<arrow::RecordBatch>> serve(
        const std::string& method, const std::shared_ptr<arrow::RecordBatch>& params);

    // The CatalogInfo item `catalog_catalogs` lists for this catalog.
    std::string catalog_info() const;

private:
    struct Object {
        std::string name;
        std::string item;  // encoded TableInfo / ViewInfo
    };
    struct Schema {
        SchemaPath path;
        std::optional<std::string> comment;
        std::map<std::string, Object> tables;  // keyed by lower-cased name
        std::map<std::string, Object> views;
    };
    struct State {
        int64_t version = 1;
        std::map<std::string, Schema> schemas;  // keyed by path, NUL-joined
    };

    State& state(const std::string& attach);
    int64_t reported_version(const State& state) const;
    std::vector<const Schema*> sorted_schemas(const State& state) const;
    std::string schema_item(const std::string& attach, const Schema& schema) const;
    std::vector<SchemaContents> snapshot(const std::string& attach);

    std::shared_ptr<arrow::RecordBatch> attach(const std::shared_ptr<arrow::RecordBatch>& params);
    std::shared_ptr<arrow::RecordBatch> contents(const std::string& attach,
                                                 const std::shared_ptr<arrow::RecordBatch>& params);
    void schema_create(const std::string& attach,
                       const std::shared_ptr<arrow::RecordBatch>& params);
    void schema_drop(const std::string& attach, const std::shared_ptr<arrow::RecordBatch>& params);
    void put_object(const std::string& attach, const SchemaPath& path, const std::string& name,
                    const char* kind, const std::string& on_conflict, bool view, std::string item);
    void drop_object(const std::string& attach, const SchemaPath& path, const std::string& name,
                     const char* kind, bool ignore_not_found, bool view);

    MemoryCatalogOptions options_;
    std::mutex mutex_;
    std::map<std::string, State> attaches_;
};

}  // namespace vgi
