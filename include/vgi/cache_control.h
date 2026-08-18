// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace vgi {

// A result-cache advertisement, rendered onto a batch's wire metadata as
// `vgi.cache.*` keys.
//
// The engine caches nothing unless a batch says it may. Everything here is
// advice the engine is free to ignore; nothing here is required for
// correctness, and getting it wrong shows up as stale answers rather than
// errors, which is why the semantics are spelled out per field.
struct CacheControl {
    // Freshness in whole seconds from when the full result is received.
    // Relative rather than absolute, so it is immune to clock skew between
    // worker and engine — which is why it wins over `expires`.
    std::optional<int64_t> ttl_seconds;
    // Absolute RFC 3339 UTC deadline. Lifetime is `expires - now` at receipt.
    std::optional<std::string> expires;
    // Reuse scope: "catalog" or "transaction".
    std::string scope = "catalog";
    // Never cache. Overrides every freshness key above.
    bool no_store = false;

    // Strong validator for conditional revalidation, an opaque quoted string.
    std::optional<std::string> etag;
    // Weaker RFC 3339 UTC validator; the fallback when there is no ETag.
    std::optional<std::string> last_modified;
    // Whether the worker can check freshness without recomputing the result.
    // Gates whether the engine ever bothers sending a conditional request.
    bool revalidatable = false;

    // Grace windows, in seconds: serve stale while revalidating, and serve
    // stale if the revalidation call fails.
    std::optional<int64_t> stale_while_revalidate;
    std::optional<int64_t> stale_if_error;

    // The 304 equivalent — set on a zero-row batch answering a conditional
    // request to say the engine's stored payload is still good.
    bool not_modified = false;

    // Additionally cache the result split by partition value. Only meaningful
    // for a SINGLE_VALUE_PARTITIONS table function.
    bool partition_scope = false;
    // Additionally memoize per distinct input value. Only meaningful for an
    // exchange-mode map.
    //
    // Off by default and worth leaving off: a per-value serve costs a cache
    // probe, a decode and an assembly step per distinct value, which only pays
    // back when that is cheaper than calling the function. For arithmetic it
    // is far slower than simply answering.
    bool per_value = false;

    // The `vgi.cache.*` metadata this advertises.
    std::map<std::string, std::string> to_metadata() const;
};

}  // namespace vgi
