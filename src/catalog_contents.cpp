// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "catalog_contents.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

#include <arrow/array/builder_binary.h>
#include <arrow/record_batch.h>
#include <vgi_rpc/crypto.h>

#include "methods.h"
#include "vgi/constraints.h"
#include "wire.h"

namespace vgi {

namespace {

// The CatalogContentsResponse collection fields, in CatalogCollections order.
constexpr const char* kCollectionNames[] = {"schemas",   "tables", "views",
                                            "functions", "macros", "indexes"};

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

// The same parent-before-child order catalog_schemas guarantees.
void sort_parents_first(std::vector<SchemaContents>& schemas) {
    std::stable_sort(schemas.begin(), schemas.end(),
                     [](const SchemaContents& a, const SchemaContents& b) {
                         return a.path.size() < b.path.size();
                     });
}

// A table a custom handler encoded itself, held to the same contract as one
// the SDK encodes: its records checked against the extension's ATTACH rules,
// and its columns served with nullability rewritten to match them.
std::string serve_table_item(const std::string& item) {
    auto info = wire::decode_ipc(item);
    if (!info || info->num_rows() != 1)
        throw std::invalid_argument("catalog_contents returned a malformed TableInfo");
    const auto schema_path = wire::get_schema_path(info, "schema_path");
    const auto name = wire::get_string(info, "name");
    const auto columns_bytes = wire::get_binary(info, "columns");
    if (columns_bytes.empty()) return item;
    const auto constraints = wire::get_binary_list(info, "constraints");
    const auto columns = wire::decode_schema(columns_bytes);
    validate_constraint_records(schema_path, name, *columns, constraints);
    const auto served = served_columns(*columns, constraints);
    if (served->Equals(*columns, /*check_metadata=*/true)) return item;
    const auto index = info->schema()->GetFieldIndex("columns");
    arrow::BinaryBuilder builder;
    const auto encoded = wire::encode_schema(served);
    if (!builder.Append(encoded).ok()) throw std::runtime_error("cannot rebuild TableInfo.columns");
    auto replaced =
        info->SetColumn(index, info->schema()->field(index), builder.Finish().ValueOrDie());
    if (!replaced.ok()) throw std::runtime_error(replaced.status().ToString());
    return wire::encode_ipc(*replaced);
}

CatalogContentsResult not_modified(const std::string& etag) {
    CatalogContentsResult result;
    result.etag = etag;
    result.not_modified = true;
    return result;
}

}  // namespace

CatalogCollections encode_collections(const std::vector<SchemaContents>& schemas) {
    // Functions of every subtype share one stream, both macro kinds another.
    std::vector<std::string> infos;
    std::vector<const std::vector<std::string>*> tables, views, functions, macros, indexes;
    for (const auto& schema : schemas) {
        infos.push_back(schema.schema);
        tables.push_back(&schema.tables);
        views.push_back(&schema.views);
        functions.insert(functions.end(), {&schema.scalar_functions, &schema.aggregate_functions,
                                           &schema.table_functions});
        macros.insert(macros.end(), {&schema.scalar_macros, &schema.table_macros});
        indexes.push_back(&schema.indexes);
    }
    return {wire::concat_ipc({&infos}),  wire::concat_ipc(tables), wire::concat_ipc(views),
            wire::concat_ipc(functions), wire::concat_ipc(macros), wire::concat_ipc(indexes)};
}

std::string collections_digest(const CatalogCollections& collections) {
    // vgi-python's contents_codec.contents_digest framing (protocol 2.2): a
    // domain prefix, then per collection its name, a presence byte and, when
    // present, the stream's little-endian u64 length and bytes.
    vgi_rpc::crypto::Sha256 hash;
    hash.update("vgi.catalog.contents.2.2");
    hash.update_byte(0);
    for (size_t i = 0; i < collections.size(); ++i) {
        hash.update(kCollectionNames[i]);
        hash.update_byte(0);
        const auto& data = collections[i];
        hash.update_byte(data ? 1 : 0);
        if (!data) continue;
        uint8_t bytes[8];
        for (int b = 0; b < 8; ++b)
            bytes[b] = static_cast<uint8_t>(uint64_t(data->size()) >> (8 * b));
        hash.update(bytes, sizeof bytes);
        hash.update(*data);
    }
    return hash.hex_digest();
}

std::string catalog_contents_digest(const std::vector<SchemaContents>& schemas) {
    auto ordered = schemas;
    sort_parents_first(ordered);
    return collections_digest(encode_collections(ordered));
}

CatalogContentsAnswer answer_catalog_contents(const CatalogContentsHandler& handler,
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
        return {not_modified(*result.etag), {}};
    }

    check_paths(result.schemas, static_cast<bool>(handler));
    if (handler) {
        for (auto& schema : result.schemas)
            for (auto& table : schema.tables) table = serve_table_item(table);
    }
    sort_parents_first(result.schemas);

    // A catalog's own etag answers not_modified before anything is encoded.
    if (result.etag && if_none_match && *result.etag == *if_none_match) {
        return {not_modified(*result.etag), {}};
    }
    auto collections = encode_collections(result.schemas);
    if (!result.etag && etag_mode == CatalogContentsEtag::ContentHash) {
        result.etag = collections_digest(collections);
        if (if_none_match && *result.etag == *if_none_match) {
            return {not_modified(*result.etag), {}};
        }
    }
    return {std::move(result), std::move(collections)};
}

std::shared_ptr<arrow::RecordBatch> encode_catalog_contents(int64_t catalog_version,
                                                            const CatalogContentsAnswer& answer) {
    auto builder = wire::ResultBuilder(payload_schema_of("catalog_contents"));
    builder.set_int64("catalog_version", catalog_version)
        .set_optional_string("etag", answer.result.etag)
        .set_bool("not_modified", answer.result.not_modified);
    for (size_t i = 0; i < answer.collections.size(); ++i) {
        if (const auto& stream = answer.collections[i]) {
            builder.set_binary(kCollectionNames[i], *stream);
        } else {
            builder.set_null(kCollectionNames[i]);
        }
    }
    return builder.finish();
}

}  // namespace vgi
