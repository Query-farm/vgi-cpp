// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "dispatcher.h"

#include <stdexcept>

namespace vgi {

void Dispatcher::register_scalar(std::shared_ptr<ScalarFunction> fn) {
    if (!fn) throw std::invalid_argument("register_scalar: null function");
    const auto name = fn->name();
    // A duplicate is a programming error, not a runtime condition: the engine
    // resolves by name, so a second registration would silently shadow the
    // first at some later, much less obvious point.
    if (scalar_by_name_.count(name)) {
        throw std::invalid_argument("scalar function already registered: " + name);
    }
    scalar_by_name_.emplace(name, scalars_.size());
    scalars_.push_back(std::move(fn));
}

void Dispatcher::install(vgi_rpc::ServerBuilder&) {
    // Nothing yet.  The VGI methods land here — see docs/roadmap.md; the first
    // milestone is catalog_attach / catalog_schemas /
    // catalog_schema_contents_functions / bind / init plus the scalar call
    // path, which together are the narrowest slice the extension can ATTACH.
}

}  // namespace vgi
