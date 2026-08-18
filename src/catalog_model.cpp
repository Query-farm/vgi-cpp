// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/catalog.h"

#include <algorithm>

namespace vgi {

CatalogModel::CatalogModel() {
    // `main` always exists; a worker that declares nothing still has one.
    schemas.push_back(std::make_unique<CatalogSchema>());
}

CatalogSchema& CatalogModel::schema(const std::string& schema_name) {
    for (auto& s : schemas) {
        if (s->name == schema_name) return *s;
    }
    // Named, not brace-positional: this list has grown twice, and a positional
    // one silently stops initializing whatever was appended.
    auto created = std::make_unique<CatalogSchema>();
    created->name = schema_name;
    schemas.push_back(std::move(created));
    return *schemas.back();
}

const CatalogSchema* CatalogModel::find_schema(const std::string& schema_name) const {
    for (const auto& s : schemas) {
        if (s->name == schema_name) return s.get();
    }
    return nullptr;
}

std::vector<std::string> CatalogModel::schema_names() const {
    std::vector<std::string> names;
    names.reserve(schemas.size());
    for (const auto& s : schemas) names.push_back(s->name);
    return names;
}

}  // namespace vgi
