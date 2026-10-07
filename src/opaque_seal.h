// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Sealing `attach_opaque_data` and `transaction_opaque_data`
// (vgi-python docs/protocol/vgi-opaque-data-sealing.md, normative).
//
// The client stores these values and sends them back on every later call, so
// on an authenticating transport (HTTP) they are sealed with XChaCha20-Poly1305
// under the deployment's signing key -- the envelope
// `version || nonce || ciphertext || tag`, the one attach tickets use -- with
// the caller's identity in the AAD, and each transaction additionally bound to
// its parent attach envelope:
//
//     attach AAD      = "vgi.attach_opaque_data.v1" 0x00 || identity
//     transaction AAD = "vgi.transaction_opaque_data.v1" 0x00 || identity
//                       || 0x00 || attach_envelope
//     identity        = 0x01 domain 0x00 principal  |  0x00 "anonymous"
//
// The values are opened once, at the RPC boundary, before any handler runs;
// handlers only ever see plaintext. Anything that fails to open -- another
// caller, another attach, a flipped byte, a malformed or plaintext value --
// is the one error `<field> not recognized` (INVALID_ARGUMENT, kind
// `opaque_data_not_recognized`, no details). There is no
// fallback: with a key, nothing skips the open.
//
// Without a key (stdio, unix: OS-owned transports) the values pass through
// unsealed, and the attach value then never carries a secret attach option
// (see Dispatcher::seal_attachment).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <arrow/record_batch.h>
#include <vgi_rpc/call_context.h>
#include <vgi_rpc/identity.h>
#include <vgi_rpc/request.h>

namespace vgi::opaque {

using Key = std::array<uint8_t, 32>;

inline constexpr uint8_t kAttachEnvelopeVersion = 0x02;
inline constexpr uint8_t kTransactionEnvelopeVersion = 0x02;

// `0x01 domain 0x00 principal` for an authenticated caller, else
// `0x00 "anonymous"`.
std::string identity_tail(const vgi_rpc::AuthContext& auth);
std::string attach_aad(const vgi_rpc::AuthContext& auth);
std::string transaction_aad(const vgi_rpc::AuthContext& auth, const std::string& attach_envelope);

std::string seal_attach(const Key& key, const std::string& plaintext,
                        const vgi_rpc::AuthContext& auth);
// Throws the uniform rejection on any failure.
std::string open_attach(const Key& key, const std::string& envelope,
                        const vgi_rpc::AuthContext& auth);
std::string seal_transaction(const Key& key, const std::string& plaintext,
                             const vgi_rpc::AuthContext& auth, const std::string& attach_envelope);
std::string open_transaction(const Key& key, const std::string& envelope,
                             const vgi_rpc::AuthContext& auth, const std::string& attach_envelope);

// The uniform error: `"<field> not recognized"`, INVALID_ARGUMENT, kind
// `opaque_data_not_recognized`, no details. A probing
// caller learns nothing about which check failed.
[[noreturn]] void reject(const std::string& field);

// The first 12 hex characters of SHA-256 over `value`'s lowercase hex text
// (vgi_rpc.sentry.short_hash): what a log line, a trace or an error may carry
// instead of an opaque value.
std::string short_hash(const std::string& value);

// One RPC call, as the boundary saw it. Installed for the length of the
// handler; handlers read it through `current()`.
class Call {
public:
    Call(const std::optional<Key>& key, const vgi_rpc::AuthContext& auth);
    ~Call();
    Call(const Call&) = delete;
    Call& operator=(const Call&) = delete;

    // Open every opaque value the request carries -- at the top level, inside
    // its `request` record and inside a nested `bind_call` -- and return the
    // request with plaintext in their place. Without a key, unchanged.
    vgi_rpc::Request open(const vgi_rpc::Request& request);

    bool sealing() const noexcept { return key_.has_value(); }
    // The attach value exactly as the caller sent it (sealed on HTTP).
    const std::optional<std::string>& attach_as_sent() const noexcept { return attach_as_sent_; }

    // Seal outgoing values for this caller. Without a key, unchanged.
    std::string seal_attach(const std::string& plaintext) const;
    std::string seal_transaction(const std::string& plaintext) const;

private:
    std::shared_ptr<arrow::RecordBatch> open_batch(const std::shared_ptr<arrow::RecordBatch>& batch,
                                                   const std::optional<std::string>& outer_attach);

    std::optional<Key> key_;
    vgi_rpc::AuthContext auth_;
    std::optional<std::string> attach_as_sent_;
    Call* previous_;
};

// The call being served on this thread, or nullptr outside a handler.
const Call* current();

// What a response may echo back as `attach_opaque_data` (a SchemaInfo's
// handle): the value the caller sent, never a freshly built plaintext.
std::string echo_attach(const std::string& plaintext);

// Seal an outgoing attach / transaction value for the current call.
std::string seal_attach_out(const std::string& plaintext);
std::string seal_transaction_out(const std::string& plaintext);

// 32 hex characters from the OS CSPRNG, for ids that live inside an opaque
// value.
std::string random_id();

}  // namespace vgi::opaque
