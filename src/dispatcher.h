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

private:
    CatalogModel catalog_;
    std::vector<std::shared_ptr<ScalarFunction>> scalars_;
    std::unordered_map<std::string, size_t> scalar_by_name_;
};

}  // namespace vgi
