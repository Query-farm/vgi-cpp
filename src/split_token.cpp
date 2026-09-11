// © Copyright 2025, 2026 Query Farm LLC - https://query.farm

#include "split_token.h"

#include <cstring>
#include <stdexcept>

#include <vgi_rpc/crypto.h>

namespace vgi::split_token {

namespace {

// Prefixed and NUL-delimited so no two different field sets can hash the same:
// without the delimiters, a function named "a" in schema "bc" and one named
// "ab" in schema "c" would be the same bytes.
constexpr const char* kAadPrefix = "vgi.split_token.v1";

void feed(vgi_rpc::crypto::Sha256& h, const char* label, const std::string& value) {
    h.update(label);
    h.update_byte(0);
    h.update(value);
    h.update_byte(0);
}

void put_u16_le(std::string& out, uint16_t v) {
    out.push_back(static_cast<char>(v & 0xFF));
    out.push_back(static_cast<char>((v >> 8) & 0xFF));
}

}  // namespace

std::string bind_fingerprint(const SchemaPath& schema_path, const std::string& function_name,
                             const std::string& arguments, const std::string& settings) {
    vgi_rpc::crypto::Sha256 h;
    h.update(std::string(kAadPrefix));
    h.update_byte(0);
    for (const auto& component : schema_path) feed(h, "schema_path", component);
    feed(h, "function_name", function_name);
    feed(h, "arguments", arguments);
    feed(h, "settings", settings);
    const auto digest = h.digest();
    return std::string(reinterpret_cast<const char*>(digest.data()), kFingerprintLen);
}

std::string anchor_for(std::optional<int64_t> catalog_version) {
    const auto v = static_cast<uint64_t>(catalog_version.value_or(0));
    std::string out;
    out.reserve(8);
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
    return out;
}

std::string build(const std::string& payload, const std::string& fingerprint,
                  const std::string& anchor) {
    if (fingerprint.size() != kFingerprintLen) {
        throw std::invalid_argument("split token: fingerprint must be 16 bytes");
    }
    if (anchor.size() > 0xFFFF) {
        throw std::invalid_argument("split token: consistency anchor exceeds u16");
    }
    std::string out;
    out.reserve(kHeaderLen + anchor.size() + payload.size());
    out.push_back(static_cast<char>(kFormatVersion));
    out.push_back(0);  // no key on these transports, so nothing is sealed
    put_u16_le(out, static_cast<uint16_t>(anchor.size()));
    out.append(fingerprint);
    out.append(anchor);
    out.append(payload);
    return out;
}

OpenResult open(const std::string& token, const std::string& expected_fingerprint,
                const std::string& current_anchor) {
    const OpenResult invalid{std::nullopt, OpenError::Invalid};
    if (token.size() < kHeaderLen) return invalid;
    if (static_cast<uint8_t>(token[0]) != kFormatVersion) return invalid;

    const auto flags = static_cast<uint8_t>(token[1]);
    // Every bit is reserved here, `payload_sealed` included: this SDK holds no
    // key, so a token claiming to be sealed is one we cannot open, and a token
    // setting a reserved bit is from a future this build does not speak.
    if (flags != 0) return invalid;

    const auto anchor_len = static_cast<size_t>(static_cast<uint8_t>(token[2])) |
                            (static_cast<size_t>(static_cast<uint8_t>(token[3])) << 8);
    const size_t end_of_anchor = kHeaderLen + anchor_len;
    if (token.size() < end_of_anchor) return invalid;

    const auto fingerprint = token.substr(4, kFingerprintLen);
    if (!vgi_rpc::crypto::constant_time_equal(fingerprint, expected_fingerprint)) {
        return invalid;
    }
    // Checked after the bind check and kept distinct in the caller's error
    // text: "this snapshot moved" is a different situation for a client from
    // "this token is not yours", and only one of them means re-plan.
    if (anchor_len != current_anchor.size() ||
        token.compare(kHeaderLen, anchor_len, current_anchor) != 0) {
        return {std::nullopt, OpenError::SnapshotExpired};
    }

    return {token.substr(end_of_anchor), OpenError::None};
}

}  // namespace vgi::split_token
