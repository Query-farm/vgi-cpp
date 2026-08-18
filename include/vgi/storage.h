// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vgi {

// State that outlives one worker process.
//
// This is not a convenience. The engine parallelizes a buffering sink across
// several worker *processes* and finalizes in yet another — four distinct pids
// for a three-row query, in the case that motivated this — so a function that
// accumulates in memory sees an empty accumulator at finalize and silently
// returns nothing. Anything a function must remember between calls belongs
// here.
//
// Three shapes, because the fixtures need three:
//
//   * **kv** — overwrite-by-key, for a single evolving value.
//   * **log** — append-only with monotonic ids, for accumulating batches that
//     are later replayed in order.
//   * **queue** — FIFO with an atomic claim, so several workers can share out
//     work without coordinating.
//
// Every method is safe across processes. `scope` is the execution id: clearing
// a scope discards everything one execution accumulated.
class FunctionStorage {
public:
    virtual ~FunctionStorage() = default;

    virtual std::optional<std::string> kv_get(const std::string& scope,
                                              const std::string& key) = 0;
    virtual void kv_put(const std::string& scope, const std::string& key,
                        const std::string& value) = 0;
    virtual void kv_del(const std::string& scope, const std::string& key) = 0;

    // Append `value`, returning its monotonic id within (scope, ns, key).
    virtual int64_t append(const std::string& scope, const std::string& ns,
                           const std::string& key, const std::string& value) = 0;
    // Entries with id > after_id, in id order, at most `limit` of them.
    virtual std::vector<std::pair<int64_t, std::string>> scan(const std::string& scope,
                                                              const std::string& ns,
                                                              const std::string& key,
                                                              int64_t after_id,
                                                              size_t limit) = 0;

    virtual void queue_push(const std::string& scope,
                            const std::vector<std::string>& items) = 0;
    virtual std::optional<std::string> queue_pop(const std::string& scope) = 0;

    // Discard everything under `scope`.
    virtual void clear(const std::string& scope) = 0;
};

// The process-wide store.
//
// Filesystem-backed under `$TMPDIR/vgi-cpp-worker-<uid>/`, namespaced by uid so
// that on a host with a shared /tmp two users neither read each other's state
// nor lock each other out of the first creator's 0700 directory.
std::shared_ptr<FunctionStorage> default_storage();

}  // namespace vgi
