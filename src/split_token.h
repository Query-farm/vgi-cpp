// © Copyright 2025, 2026 Query Farm LLC - https://query.farm

// The split-token envelope: the framework's wrapper around a worker's payload.
//
// Layout, matching the reference implementation byte for byte so a token stays
// legible across SDKs:
//
//     offset  size  field
//     0       1     format_version      currently 1
//     1       1     flags               bit0 = payload_sealed; 1-7 reserved, MUST be 0
//     2       2     anchor_len          u16 LE
//     4       16    bind_fingerprint    truncated SHA-256 of the bind identity
//     20      var   consistency_anchor
//     20+n    var   payload
//
// The fingerprint is minted *and* verified by the same worker, so it needs
// self-consistency only — it does not have to agree with any client, and no
// cross-SDK fixture covers it. That is why it can hash C++ spellings of the
// bind fields rather than reproducing the reference's Python `repr`.
//
// Nothing here seals: this SDK's transports carry no signing key, and the
// reference's own header explains why the header must stay plaintext where
// DuckDB runs. If a key ever arrives, the keyed/keyless decision has to come
// from the worker's key state and never from the token's own `flags` byte —
// trusting that byte is `alg:none`.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace vgi::split_token {

inline constexpr uint8_t kFormatVersion = 1;
inline constexpr uint8_t kFlagPayloadSealed = 0x01;
inline constexpr size_t kFingerprintLen = 16;
inline constexpr size_t kHeaderLen = 4 + kFingerprintLen;

// The 16-byte binding check for a bind call: which function, in which schema,
// with which arguments and settings.
std::string bind_fingerprint(const std::string& schema_name, const std::string& function_name,
                             const std::string& arguments, const std::string& settings);

// Stamp a payload into a token.
std::string build(const std::string& payload, const std::string& fingerprint,
                  const std::string& anchor);

// Verify a token and return the payload, or nothing when it is malformed, was
// minted for a different bind, or names a snapshot that has moved on.
std::optional<std::string> open(const std::string& token, const std::string& expected_fingerprint,
                                const std::string& current_anchor);

// The consistency anchor for a catalog version: int64, little-endian, and an
// absent version is zero — the same spelling the reference uses.
std::string anchor_for(std::optional<int64_t> catalog_version);

}  // namespace vgi::split_token
