// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "vgi/catalog.h"

#include <algorithm>

namespace vgi {

CatalogModel::CatalogModel() {
    // `main` always exists; a worker that declares nothing still has one.
    schemas.push_back(std::make_unique<CatalogSchema>());
}

CatalogSchema& CatalogModel::schema(const SchemaPath& schema_path) {
    for (auto& s : schemas) {
        if (s->path == schema_path) return *s;
    }
    // Named, not brace-positional: this list has grown twice, and a positional
    // one silently stops initializing whatever was appended.
    auto created = std::make_unique<CatalogSchema>();
    created->path = schema_path;
    schemas.push_back(std::move(created));
    return *schemas.back();
}

const CatalogSchema* CatalogModel::find_schema(const SchemaPath& schema_path) const {
    for (const auto& s : schemas) {
        if (s->path == schema_path) return s.get();
    }
    return nullptr;
}

std::vector<SchemaPath> CatalogModel::schema_paths() const {
    std::vector<SchemaPath> paths;
    paths.reserve(schemas.size());
    for (const auto& s : schemas) paths.push_back(s->path);
    return paths;
}

}  // namespace vgi
