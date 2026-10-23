// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The catalog-independent half of `catalog_contents`: the protocol's
// answer/etag rules and the wire encoding, shared by every catalog kind a
// worker serves (declared catalogs and memory catalogs), so they revalidate
// alike. Mirrors vgi-go's answerCatalogContents.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <arrow/record_batch.h>

#include "vgi/catalog.h"

namespace vgi {

// The six CatalogContentsResponse collections, in wire order (schemas, tables,
// views, functions, macros, indexes): each one IPC stream of that kind's info
// rows across every schema, or null when the kind is empty.
using CatalogCollections = std::array<std::optional<std::string>, 6>;

// The collections of `schemas`, which must already be in serving order.
CatalogCollections encode_collections(const std::vector<SchemaContents>& schemas);

// The content-hash etag of these collections (see catalog_contents_digest).
std::string collections_digest(const CatalogCollections& collections);

// A checked answer with its collections encoded once, so a content-hash etag
// and the response hash and send the same bytes. A not_modified answer has
// every collection null.
struct CatalogContentsAnswer {
    CatalogContentsResult result;
    CatalogCollections collections;
};

// Ask `handler` (or, with none, the call's default composition) for the
// snapshot and apply the rules:
//  - not_modified is valid only with an etag equal to if_none_match and no
//    schemas; anything else is an error naming the rule broken.
//  - schema paths must be non-empty and unique with every parent present; a
//    handler's paths must equal the SchemaInfo.path of their items.
//  - schemas are served parents first.
//  - with no etag of its own, `etag_mode` may supply one (content hash).
//  - a full answer whose etag equals if_none_match becomes not_modified.
//  - with no etag, if_none_match is ignored.
CatalogContentsAnswer answer_catalog_contents(const CatalogContentsHandler& handler,
                                              CatalogContentsEtag etag_mode,
                                              const CatalogContentsCall& call);

// The CatalogContentsResponse payload for `answer`.
std::shared_ptr<arrow::RecordBatch> encode_catalog_contents(int64_t catalog_version,
                                                            const CatalogContentsAnswer& answer);

}  // namespace vgi
