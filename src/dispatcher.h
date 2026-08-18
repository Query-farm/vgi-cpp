// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <vgi_rpc/server.h>

#include "vgi/catalog.h"
#include "vgi/function.h"

namespace vgi {

// Owns the function registries and the catalog identity, and turns them into
// the RPC methods a VGI engine calls.
//
// Split from Worker because the HTTP transport needs the registries after the
// server has been built, and because the registries outlive any one
// connection while a Worker is a one-shot builder.
class Dispatcher {
public:
    void set_catalog(CatalogModel catalog) { catalog_ = std::move(catalog); }
    const CatalogModel& catalog() const noexcept { return catalog_; }

    void register_scalar(std::shared_ptr<ScalarFunction> fn);

    const std::vector<std::shared_ptr<ScalarFunction>>& scalars() const noexcept {
        return scalars_;
    }

    // Register every VGI method on `builder`.
    void install(vgi_rpc::ServerBuilder& builder);

    using UnaryHandler = vgi_rpc::Result (Dispatcher::*)(const vgi_rpc::Request&);
    using VoidHandler = void (Dispatcher::*)(const vgi_rpc::Request&);

    // ── Handlers ──────────────────────────────────────────────────────────
    //
    // One member per protocol method, grouped into translation units by area
    // (catalog.cpp, function.cpp, …) rather than a single file: the surface is
    // 70 methods, and a handler is much easier to review beside its siblings
    // than in a 5,000-line switch.

    vgi_rpc::Result catalog_attach(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schemas(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_get(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_functions(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_tables(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_views(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_macros(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_indexes(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_copy_from_formats(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_version(const vgi_rpc::Request& request);
    void catalog_detach(const vgi_rpc::Request& request);

private:
    // Serialize one Info dataclass as the IPC bytes a list<binary> column
    // carries.
    static std::string encode_function_info(const ScalarFunction& fn,
                                            const std::string& schema_name);

    CatalogModel catalog_;
    std::vector<std::shared_ptr<ScalarFunction>> scalars_;
    std::unordered_map<std::string, size_t> scalar_by_name_;
};

}  // namespace vgi
