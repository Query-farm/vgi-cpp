// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/catalog.h"

#include <algorithm>

namespace vgi {

CatalogSchema& CatalogModel::schema(const std::string& schema_name) {
    auto it = std::find_if(schemas.begin(), schemas.end(),
                           [&](const CatalogSchema& s) { return s.name == schema_name; });
    if (it != schemas.end()) return *it;
    schemas.push_back(CatalogSchema{schema_name, {}, {}});
    return schemas.back();
}

const CatalogSchema* CatalogModel::find_schema(const std::string& schema_name) const {
    auto it = std::find_if(schemas.begin(), schemas.end(),
                           [&](const CatalogSchema& s) { return s.name == schema_name; });
    return it == schemas.end() ? nullptr : &*it;
}

std::vector<std::string> CatalogModel::schema_names() const {
    std::vector<std::string> names;
    names.reserve(schemas.size());
    for (const auto& s : schemas) names.push_back(s.name);
    return names;
}

}  // namespace vgi
