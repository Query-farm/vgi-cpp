// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/cache_control.h"

namespace vgi {
namespace {

// The wire keys, spelled exactly as the canonical Python implementation does.
// A misspelled key is silently ignored by the engine rather than rejected, so
// they are constants here instead of literals at each use.
constexpr const char* kTtl = "vgi.cache.ttl";
constexpr const char* kExpires = "vgi.cache.expires";
constexpr const char* kNoStore = "vgi.cache.no_store";
constexpr const char* kScope = "vgi.cache.scope";
constexpr const char* kEtag = "vgi.cache.etag";
constexpr const char* kLastModified = "vgi.cache.last_modified";
constexpr const char* kRevalidatable = "vgi.cache.revalidatable";
constexpr const char* kStaleWhileRevalidate = "vgi.cache.stale_while_revalidate";
constexpr const char* kStaleIfError = "vgi.cache.stale_if_error";
constexpr const char* kNotModified = "vgi.cache.not_modified";
constexpr const char* kPartitionScope = "vgi.cache.partition_scope";
constexpr const char* kPerValue = "vgi.cache.per_value";

}  // namespace

std::map<std::string, std::string> CacheControl::to_metadata() const {
    std::map<std::string, std::string> metadata;

    if (ttl_seconds) {
        // Clamped: a negative lifetime is a mistake, and rendering it would
        // make the engine treat the result as already expired rather than
        // uncacheable, which is a different and more confusing outcome.
        metadata[kTtl] = std::to_string(*ttl_seconds < 0 ? 0 : *ttl_seconds);
    }
    if (expires) metadata[kExpires] = *expires;
    if (no_store) metadata[kNoStore] = "1";
    metadata[kScope] = scope;
    if (etag) metadata[kEtag] = *etag;
    if (last_modified) metadata[kLastModified] = *last_modified;
    if (revalidatable) metadata[kRevalidatable] = "1";
    if (stale_while_revalidate) {
        metadata[kStaleWhileRevalidate] = std::to_string(*stale_while_revalidate);
    }
    if (stale_if_error) metadata[kStaleIfError] = std::to_string(*stale_if_error);
    if (not_modified) metadata[kNotModified] = "1";
    if (partition_scope) metadata[kPartitionScope] = "1";
    if (per_value) metadata[kPerValue] = "1";

    return metadata;
}

}  // namespace vgi
