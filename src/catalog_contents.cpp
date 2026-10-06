// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "catalog_contents.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

#include <vgi_rpc/crypto.h>

#include "methods.h"
#include "wire.h"

namespace vgi {

namespace {

std::string path_key(const SchemaPath& path) {
    std::string key;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i) key.push_back('\0');
        key += path[i];
    }
    return key;
}

std::string path_display(const SchemaPath& path) {
    std::string out;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i) out.push_back('.');
        out += path[i];
    }
    return out;
}

void check_paths(const std::vector<SchemaContents>& schemas, bool verify_items) {
    std::set<std::string> keys;
    for (const auto& schema : schemas) {
        if (schema.path.empty()) {
            throw std::invalid_argument("catalog_contents returned a schema with an empty path");
        }
        if (!keys.insert(path_key(schema.path)).second) {
            throw std::invalid_argument("catalog_contents returned duplicate schema path " +
                                        path_display(schema.path));
        }
        if (verify_items) {
            const auto info = wire::decode_ipc(schema.schema);
            if (!info || wire::get_schema_path(info, "path") != schema.path) {
                throw std::invalid_argument("catalog_contents schema path " +
                                            path_display(schema.path) +
                                            " does not match its SchemaInfo.path");
            }
        }
    }
    for (const auto& schema : schemas) {
        if (schema.path.size() < 2) continue;
        const SchemaPath parent(schema.path.begin(), schema.path.end() - 1);
        if (!keys.count(path_key(parent))) {
            throw std::invalid_argument("catalog_contents returned schema " +
                                        path_display(schema.path) + " without its parent " +
                                        path_display(parent));
        }
    }
}

CatalogContentsResult not_modified(const std::string& etag) {
    CatalogContentsResult result;
    result.etag = etag;
    result.not_modified = true;
    return result;
}

}  // namespace

std::string catalog_contents_digest(const std::vector<SchemaContents>& schemas) {
    vgi_rpc::crypto::Sha256 hash;
    const auto length = [&](uint64_t n) {
        uint8_t bytes[8];
        for (int i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(n >> (8 * i));
        hash.update(bytes, sizeof bytes);
    };
    const auto chunk = [&](const std::string& data) {
        length(data.size());
        hash.update(data);
    };
    const auto chunks = [&](const std::vector<std::string>& values) {
        length(values.size());
        for (const auto& value : values) chunk(value);
    };
    length(schemas.size());
    for (const auto& schema : schemas) {
        length(schema.path.size());
        for (const auto& part : schema.path) chunk(part);
        chunk(schema.schema);
        for (const auto* kind : {&schema.tables, &schema.views, &schema.scalar_functions,
                                 &schema.aggregate_functions, &schema.table_functions,
                                 &schema.scalar_macros, &schema.table_macros, &schema.indexes}) {
            chunks(*kind);
        }
    }
    return hash.hex_digest();
}

CatalogContentsResult answer_catalog_contents(const CatalogContentsHandler& handler,
                                              CatalogContentsEtag etag_mode,
                                              const CatalogContentsCall& call) {
    const auto& if_none_match = call.if_none_match;
    CatalogContentsResult result;
    if (handler) {
        result = handler(call);
    } else {
        result.schemas = call.contents();
    }

    if (result.not_modified) {
        if (!result.etag || !if_none_match || *result.etag != *if_none_match) {
            throw std::invalid_argument(
                "catalog_contents returned not_modified, but only a catalog whose etag equals "
                "if_none_match may (and it must return that etag)");
        }
        if (!result.schemas.empty()) {
            throw std::invalid_argument(
                "catalog_contents returned not_modified with schemas; it must return none");
        }
        return not_modified(*result.etag);
    }

    check_paths(result.schemas, static_cast<bool>(handler));
    // The same parent-before-child order catalog_schemas guarantees.
    std::stable_sort(result.schemas.begin(), result.schemas.end(),
                     [](const SchemaContents& a, const SchemaContents& b) {
                         return a.path.size() < b.path.size();
                     });

    if (!result.etag && etag_mode == CatalogContentsEtag::ContentHash) {
        result.etag = catalog_contents_digest(result.schemas);
    }
    if (result.etag && if_none_match && *result.etag == *if_none_match) {
        return not_modified(*result.etag);
    }
    return result;
}

std::shared_ptr<arrow::RecordBatch> encode_catalog_contents(int64_t catalog_version,
                                                            const CatalogContentsResult& result) {
    auto builder = wire::ResultBuilder(payload_schema_of("catalog_contents"));
    builder.set_int64("catalog_version", catalog_version)
        .set_optional_string("etag", result.etag)
        .set_bool("not_modified", result.not_modified)
        .set_schema_contents("schemas", result.schemas);
    return builder.finish();
}

}  // namespace vgi
