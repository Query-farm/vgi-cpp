// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Sealed framework payload v1: authenticated encryption of the framework's own
// context written to worker storage (the same format in every VGI SDK; the
// reference is vgi-python `vgi/_storage_seal.py`).
//
// The framework persists some context so that any worker process can serve
// any RPC of an execution: a buffered table function's bind call (its
// `secrets`, the resolved secret values and the attach and transaction
// values), the bound output schema, aggregate constant arguments, the
// streaming-aggregate session, and secret ATTACH options held server-side on
// OS-owned transports. None of that may sit in storage in plaintext, and
// nothing read back may be trusted unauthenticated.
//
//     K    = HKDF-SHA256(ikm, salt = "", info = "vgi.storage.payload.v1", L = 32)
//     AAD  = "vgi.storage.payload.v1" || u32be(len) shard_key || u32be(len) scope_id
//            || u32be(len) namespace || u32be(len) key
//     wire = 0x01 || nonce(12) || AES-256-GCM ciphertext || tag(16)
//
// `ikm` is `VGI_SIGNING_KEY` exactly as configured (its UTF-8 bytes, never
// hex-decoded or hashed). A worker with no signing key uses the host-local key
// file every VGI worker process of the user on the host shares (the one secret
// ATTACH options use): one query's RPCs are spread over several processes
// sharing one store, so a per-process key would not do.
//
// Any failure to open -- unknown version, short, tampered, wrong key, moved to
// another slot, or an unsealed value from an older release -- throws
// `StoragePayloadError`. There is no plaintext fallback.
//
// What user functions write themselves (process() appends, aggregate group
// states, window partitions, finalize output, queue items) is their data and is
// stored as written.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace vgi {
class FunctionStorage;
}

namespace vgi::storage_seal {

inline constexpr const char* kInfo = "vgi.storage.payload.v1";
inline constexpr uint8_t kVersion = 0x01;
inline constexpr size_t kKeyBytes = 32;
inline constexpr size_t kNonceBytes = 12;
inline constexpr size_t kTagBytes = 16;

// Framework namespaces (vgi-python `FrameworkNS`).
inline constexpr const char* kBufferingInitNs = "_vgi/buffering_init";
inline constexpr const char* kAggregateStateNs = "_vgi/aggregate_state";
inline constexpr const char* kStreamingSessionNs = "_vgi/streaming_session";
inline constexpr const char* kAttachSecretOptionsNs = "_vgi/attach_secret_options";

class StoragePayloadError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The slot a payload is bound to. This SDK's store is not sharded, so its
// framework payloads use the empty shard key.
struct Slot {
    std::string shard_key;
    std::string scope_id;
    std::string ns;
    std::string key;
};

// vgi-python `BoundStorage.pack_int_key`: 8-byte signed little-endian.
std::string pack_int_key(int64_t value);

std::array<uint8_t, kKeyBytes> derive_key(const std::string& ikm);
std::string aad(const Slot& slot);

// Under an explicit `ikm`. `nonce` fixes the nonce **for test vectors only**.
std::string seal_with(const std::string& payload, const std::string& ikm, const Slot& slot,
                      const std::optional<std::array<uint8_t, kNonceBytes>>& nonce = std::nullopt);
std::string open_with(const std::string& blob, const std::string& ikm, const Slot& slot);

// The process's key material: `VGI_SIGNING_KEY` when set and non-empty, else
// the host-local key file. Throws StoragePayloadError when neither is usable.
std::string resolve_ikm();

// `$VGI_ATTACH_SECRET_KEY_FILE`; else `$XDG_RUNTIME_DIR/vgi/attach-secret-options.key`
// when that is a directory; else `<temp>/vgi-<user>/attach-secret-options.key`.
std::string host_local_key_path();

// Seal / open under `resolve_ikm()`.
std::string seal(const std::string& payload, const Slot& slot);
std::string open(const std::string& blob, const Slot& slot);

// Seal `payload` to `slot` and store it at kv (`scope`, `kv_key`).
void sealed_put(FunctionStorage& storage, const std::string& scope, const std::string& kv_key,
                const Slot& slot, const std::string& payload);
// Read and open; nullopt when the kv entry is absent.
std::optional<std::string> sealed_get(FunctionStorage& storage, const std::string& scope,
                                      const std::string& kv_key, const Slot& slot);

}  // namespace vgi::storage_seal
