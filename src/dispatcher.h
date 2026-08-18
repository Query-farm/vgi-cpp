// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <vgi_rpc/server.h>
#include <vgi_rpc/stream.h>

#include "vgi/catalog.h"
#include "vgi/function.h"

namespace vgi {

// The `init` stream's header schema (`GlobalInitResponse`). Defined in
// function_dispatch.cpp; registration needs it, and so does the handler.
const std::shared_ptr<arrow::Schema>& global_init_response_schema();

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

    vgi_rpc::Result bind(const vgi_rpc::Request& request);
    vgi_rpc::Stream init(const vgi_rpc::Request& request);

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

    // Every registration under `name`, in registration order.
    //
    // A name is not a key: `type_info` is registered five times, once per
    // argument type, and the engine chooses between them at the call site.
    // Resolution therefore happens at bind, against the types the engine
    // resolved, not at lookup.
    std::vector<std::shared_ptr<ScalarFunction>> scalars_named(const std::string& name) const;

    // The overload of `name` that matches `params`, or a clear error.
    //
    // Failing here is a user-visible error: the engine advertised this
    // function from our own discovery answer, so being unable to resolve it
    // means the two disagree about what was advertised.
    std::shared_ptr<ScalarFunction> resolve_scalar(const std::string& name,
                                                   const BindParams& params) const;
    static void check_type_bounds(const ScalarFunction& fn, const BindParams& params);
    BindParams read_bind_request(const std::shared_ptr<arrow::RecordBatch>& bind_call) const;

    CatalogModel catalog_;
    std::vector<std::shared_ptr<ScalarFunction>> scalars_;
    // name -> indices into scalars_, in registration order.
    std::unordered_map<std::string, std::vector<size_t>> scalar_by_name_;
};

}  // namespace vgi
