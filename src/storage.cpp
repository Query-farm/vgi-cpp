// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/storage.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <stdexcept>
#include <cstring>

namespace vgi {
namespace {

namespace fs = std::filesystem;

// Hex, so an arbitrary byte string becomes a safe path component. Scope and
// key are opaque bytes chosen by the engine and may contain anything.
std::string hex(const std::string& bytes) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char c : bytes) {
        out.push_back(kDigits[c >> 4]);
        out.push_back(kDigits[c & 0x0F]);
    }
    return out;
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// Write through a temporary and rename, so a reader never observes a
// half-written value. Rename is atomic within a filesystem.
//
// Write failures are raised, not swallowed: on a full disk the write fails
// silently and the rename then installs a truncated value over a good one,
// which is indistinguishable from a legitimately empty value at every reader.
void write_file_atomically(const fs::path& path, const std::string& value) {
    const auto temporary = fs::path(path.string() + ".tmp." + std::to_string(::getpid()));
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out.write(value.data(), static_cast<std::streamsize>(value.size()));
        out.flush();
        if (!out) {
            std::error_code ec;
            fs::remove(temporary, ec);
            throw std::runtime_error("vgi storage: cannot write " + temporary.string());
        }
    }
    std::error_code ec;
    fs::rename(temporary, path, ec);
    if (ec) {
        fs::remove(temporary, ec);
        throw std::runtime_error("vgi storage: cannot publish " + path.string());
    }
}

class FilesystemStorage : public FunctionStorage {
public:
    FilesystemStorage() {
        base_ = fs::temp_directory_path() /
                ("vgi-cpp-worker-" + std::to_string(static_cast<long>(::getuid())));
        std::error_code ec;
        fs::create_directories(base_, ec);
        // 0700: buffered state can hold query data, and $TMPDIR may be shared.
        fs::permissions(base_, fs::perms::owner_all, fs::perm_options::replace, ec);
        sweep_stale();
    }

    std::optional<std::string> kv_get(const std::string& scope, const std::string& key) override {
        const auto path = kv_path(scope, key);
        std::error_code ec;
        if (!fs::exists(path, ec)) return std::nullopt;
        return read_file(path);
    }

    void kv_put(const std::string& scope, const std::string& key,
                const std::string& value) override {
        const auto path = kv_path(scope, key);
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
        write_file_atomically(path, value);
    }

    void kv_del(const std::string& scope, const std::string& key) override {
        std::error_code ec;
        fs::remove(kv_path(scope, key), ec);
    }

    int64_t append(const std::string& scope, const std::string& ns, const std::string& key,
                   const std::string& value) override {
        const auto dir = log_dir(scope, ns, key);
        std::error_code ec;
        fs::create_directories(dir, ec);

        // Two separate races, and they need different answers.
        //
        // *Claiming* the id: checking existence and then writing is a race,
        // and not a theoretical one — the engine runs several sink processes
        // at once, so two of them pick the same "highest + 1" and the second
        // overwrites the first's rows. An exclusive create is the only thing
        // here atomic across processes, so a claim is an `O_CREAT|O_EXCL` on a
        // *reservation* name.
        //
        // *Publishing* the entry: a claim marker is visible from the instant
        // of open, so writing the payload into the claimed file would let a
        // concurrent scan read an empty entry — which a reader cannot
        // distinguish from end-of-log, and so silently truncates the replay.
        // The payload therefore goes to a temporary and is renamed into the
        // `.entry` name only once complete.
        return claim_from(dir, highest_id(dir) + 1, value);
    }

    std::vector<std::pair<int64_t, std::string>> scan(const std::string& scope,
                                                      const std::string& ns, const std::string& key,
                                                      int64_t after_id, size_t limit) override {
        // Only `.entry` files are results; a `.claim` marker means an append
        // is in flight and its payload is not there yet.
        std::vector<std::pair<int64_t, std::string>> entries;
        const auto dir = log_dir(scope, ns, key);
        std::error_code ec;
        if (!fs::exists(dir, ec)) return entries;

        std::vector<std::pair<int64_t, fs::path>> found;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            const auto name = entry.path().filename().string();
            if (name.size() < 6 || name.compare(name.size() - 6, 6, ".entry") != 0) continue;
            const int64_t id = std::strtoll(name.c_str(), nullptr, 10);
            if (id > after_id) found.emplace_back(id, entry.path());
        }
        // Directory order is not id order on any filesystem worth relying on.
        std::sort(found.begin(), found.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [id, path] : found) {
            if (entries.size() >= limit) break;
            entries.emplace_back(id, read_file(path));
        }
        return entries;
    }

    void queue_push(const std::string& scope, const std::vector<std::string>& items) override {
        if (items.empty()) return;
        const auto dir = log_dir(scope, kQueueNamespace, "");
        std::error_code ec;
        fs::create_directories(dir, ec);

        // One directory walk for the whole push. Calling `append` per item
        // rescanned the directory each time, which made pushing N items
        // quadratic — a 5,000-chunk scan spent minutes here before emitting a
        // row.
        int64_t next = highest_id(dir) + 1;
        for (const auto& item : items) next = claim_from(dir, next, item) + 1;
    }

    std::optional<std::string> queue_pop(const std::string& scope) override {
        const auto dir = log_dir(scope, kQueueNamespace, "");
        std::error_code ec;
        if (!fs::exists(dir, ec)) return std::nullopt;

        // Probed forward from where this process last succeeded rather than by
        // listing and sorting the directory, which cost O(n log n) per pop and
        // so O(n² log n) to drain a queue.
        //
        // Skipping everything below the cursor is safe: ids are handed out in
        // ascending order, and an id this process passed over was one another
        // popper had already claimed.
        auto& cursor = queue_cursors_[scope];
        for (;;) {
            // The lowest id this pass skipped because its push had not landed
            // yet. The cursor must not move past it: an id passed over is
            // never probed again, and the item on it would be stranded.
            std::optional<int64_t> pending;
            for (int64_t id = cursor + 1; id <= queue_high(scope, dir); ++id) {
                const auto path = dir / (pad(id) + ".entry");
                // Claim by renaming: exactly one racing popper's rename
                // succeeds, which is what makes this safe across processes
                // without a lock.
                const auto claimed =
                    fs::path(path.string() + ".claimed." + std::to_string(::getpid()));
                fs::rename(path, claimed, ec);
                if (ec) {
                    ec.clear();
                    // No entry: either another popper took it — in which case
                    // the reservation is long gone — or a push reserved the id
                    // and is still writing it.
                    if (fs::exists(dir / (pad(id) + ".claim"), ec)) {
                        if (!pending) pending = id;
                    }
                    ec.clear();
                    continue;
                }
                cursor = pending ? *pending - 1 : id;
                auto value = read_file(claimed);
                fs::remove(claimed, ec);
                return value;
            }
            if (pending) {
                // Everything above is drained but a push is mid-flight below.
                // Rewind so the next call sees it.
                cursor = *pending - 1;
                return std::nullopt;
            }
            // Nothing left below the known high-water mark; re-read it once in
            // case another process pushed while we were draining.
            const int64_t known = queue_highs_[scope];
            queue_highs_[scope] = highest_id(dir);
            if (queue_highs_[scope] <= known) return std::nullopt;
        }
    }

    void clear(const std::string& scope) override {
        // An empty scope hexes to an empty path component, and `base_ / ""` is
        // `base_` itself — clearing one execution would take out every other
        // one running in this worker. There is no legitimate empty scope.
        if (scope.empty()) return;
        std::error_code ec;
        fs::remove_all(base_ / hex(scope), ec);
        queue_cursors_.erase(scope);
        queue_highs_.erase(scope);
    }

private:
    // Reclaim scopes no process has touched in a day.
    //
    // Only the buffering and aggregate destructors clear a scope, and only
    // when the query reaches them: a cancelled scan, a killed worker or a
    // producer scope — which has no destructor call at all — leaves its
    // directory behind forever. In a pooled worker on a shared $TMPDIR that
    // accumulates without bound.
    //
    // A day, and by mtime, because an *active* scope keeps writing: every
    // `kv_put` and every queue claim touches the directory. A live collection
    // therefore never looks stale, however long the query runs.
    void sweep_stale() {
        std::error_code ec;
        const auto cutoff = fs::file_time_type::clock::now() - std::chrono::hours(24);
        for (const auto& entry : fs::directory_iterator(base_, ec)) {
            if (ec) return;
            std::error_code entry_ec;
            if (!entry.is_directory(entry_ec) || entry_ec) continue;
            const auto written = fs::last_write_time(entry.path(), entry_ec);
            if (entry_ec || written >= cutoff) continue;
            fs::remove_all(entry.path(), entry_ec);
        }
    }

    fs::path kv_path(const std::string& scope, const std::string& key) const {
        return base_ / hex(scope) / ("kv_" + hex(key));
    }

    fs::path log_dir(const std::string& scope, const std::string& ns,
                     const std::string& key) const {
        return base_ / hex(scope) / ("log_" + hex(ns) + "__" + hex(key));
    }

    // The highest id this process knows to have been pushed, refreshed only
    // when the forward probe runs out.
    int64_t queue_high(const std::string& scope, const fs::path& dir) {
        auto it = queue_highs_.find(scope);
        if (it == queue_highs_.end()) it = queue_highs_.emplace(scope, highest_id(dir)).first;
        return it->second;
    }

    // Claim the first free id at or above `from`, then publish `value` under
    // it. Two separate races, and they need different answers.
    //
    // *Claiming* the id: checking existence and then writing is a race, and
    // not a theoretical one — the engine runs several sink processes at once,
    // so two of them pick the same "highest + 1" and the second overwrites the
    // first's rows. An exclusive create is the only thing here atomic across
    // processes, so a claim is an `O_CREAT|O_EXCL` on a *reservation* name.
    //
    // *Publishing* the entry: a claim marker is visible from the instant of
    // open, so writing the payload into the claimed file would let a
    // concurrent scan read an empty entry — which a reader cannot distinguish
    // from end-of-log, and so silently truncates the replay. The payload
    // therefore goes to a temporary and is renamed into the `.entry` name only
    // once complete.
    static int64_t claim_from(const fs::path& dir, int64_t from, const std::string& value) {
        const int64_t limit = from + kMaxClaimAttempts;
        for (int64_t id = from; id < limit; ++id) {
            const auto reservation = dir / (pad(id) + ".claim");
            const int fd = ::open(reservation.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
            if (fd < 0) {
                if (errno == EEXIST) continue;  // another worker took this id
                throw std::runtime_error("vgi storage: cannot claim " + reservation.string() +
                                         ": " + std::strerror(errno));
            }
            ::close(fd);
            write_file_atomically(dir / (pad(id) + ".entry"), value);
            // Dropped once the payload is there, so the marker means exactly
            // one thing to a reader: this id is reserved and its entry has not
            // landed yet. A popper uses that to tell "someone already took it"
            // from "it is still on its way".
            std::error_code ec;
            fs::remove(reservation, ec);
            return id;
        }
        throw std::runtime_error("vgi storage: could not claim a log id under " + dir.string());
    }

    // Named rather than spelled inline so the queue's log directory cannot
    // drift from the one `queue_push` and `queue_pop` each compute.
    static constexpr const char* kQueueNamespace = "__queue";

    std::map<std::string, int64_t> queue_cursors_;
    std::map<std::string, int64_t> queue_highs_;

    // How many ids to try before giving up. Fixed at the start of the loop
    // rather than recomputed: a moving bound derived from `highest_id` climbs
    // as other processes append, so it would not be a bound at all.
    static constexpr int64_t kMaxClaimAttempts = 4096;

    // Fixed width so lexical order matches numeric order, which is what lets
    // queue_pop sort paths as strings.
    static std::string pad(int64_t id) {
        std::string digits = std::to_string(id);
        return std::string(20 - std::min<size_t>(20, digits.size()), '0') + digits;
    }

    // Walks the directory, so callers must not put it in a loop condition —
    // doing so made `append` quadratic in the number of entries.
    static int64_t highest_id(const fs::path& dir) {
        int64_t highest = 0;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            const auto name = entry.path().filename().string();
            highest = std::max(highest, std::strtoll(name.c_str(), nullptr, 10));
        }
        return highest;
    }

    fs::path base_;
};

}  // namespace

std::string transaction_scope(const std::string& transaction_opaque_data) {
    return "txn:" + transaction_opaque_data;
}

std::shared_ptr<FunctionStorage> default_storage() {
    static const auto storage = std::make_shared<FilesystemStorage>();
    return storage;
}

}  // namespace vgi
