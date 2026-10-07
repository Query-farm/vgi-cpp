// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Attach tickets: a user's ATTACH, sealed so a runner can replay it later as
// that user (vgi-python docs/protocol/vgi-attach-tickets.md, the normative
// spec; byte-exact vectors in vgi/_test_fixtures/attach_ticket_vectors.json).
//
// A ticket is the *what* half of an unattended session; a sealed grant
// (`vgi_rpc.Identity.v1` `issue_grant`) is the *who*. While the user is
// attached and logged in, a client asks the worker to seal the options it
// attached with -- secret ones included -- into a ticket only this worker can
// open (`vgi.attach_tickets.v1` `seal_attach`). Later a runner holding the
// user's grant attaches with the single option `vgi_attach_ticket`, and the
// framework restores the sealed attach before any catalog code runs.
//
//     ticket   = "vgia1." base64url_nopad( 0x01 || nonce(24) || XChaCha20-Poly1305 )
//     key      = VGI_SIGNING_KEY (32 bytes as-is, otherwise its SHA-256)
//     aad      = "vgi.attach_ticket.v1" 0x00 || UTF-8(principal)
//
// The AAD binds the principal only: a ticket is sealed under the user's login
// domain and opened under domain `grant`. A ticket carries no authority -- it
// opens only for a caller with the same principal.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/record_batch.h>
#include <vgi_rpc/errors.h>
#include <vgi_rpc/identity.h>

namespace vgi {

// Token prefix; the format version is in it, so an incompatible format is a
// different prefix rather than something half-parsed.
inline constexpr const char* kAttachTicketPrefix = "vgia1.";
// The reserved ATTACH option a runner presents a ticket in. No catalog may
// declare an attach option with this name, compared case-insensitively.
inline constexpr const char* kAttachTicketOption = "vgi_attach_ticket";
// The protocol hosting `seal_attach`, and its version.
inline constexpr const char* kAttachTicketsProtocolName = "vgi.attach_tickets.v1";
inline constexpr const char* kAttachTicketsProtocolVersion = "1.0.0";
// The envelope's version byte, fixed by this format.
inline constexpr uint8_t kAttachTicketEnvelopeVersion = 0x01;
// Largest options record (serialized Arrow IPC bytes) a ticket carries.
inline constexpr size_t kAttachTicketMaxOptionsBytes = 16 * 1024;
// Longest ticket text considered at all.
inline constexpr size_t kAttachTicketMaxChars = 32 * 1024;
// Allowance for clocks disagreeing between the sealing and redeeming worker.
inline constexpr int64_t kAttachTicketClockSkewSeconds = 60;

using AttachTicketKey = std::array<uint8_t, 32>;

// `VGI_SIGNING_KEY` bytes normalized exactly as the HTTP token key is: a
// 32-byte value is used as-is; any other length is replaced by its SHA-256.
AttachTicketKey attach_ticket_key(const std::string& raw);

// What a ticket carries.
struct AttachTicketClaims {
    int64_t issued_at = 0;
    // 0 means no expiry; otherwise after `issued_at`.
    int64_t expires_at = 0;
    // 32 lowercase hex; a correlation handle, not a secret.
    std::string ticket_id;
    std::string catalog_name;
    // "" means none.
    std::string data_version_spec;
    std::string implementation_version;
    // Arrow IPC of the one-row options record, exactly as
    // `CatalogAttachRequest.options` carries it; empty for none.
    std::string options_ipc;
};

// `"vgi.attach_ticket.v1" 0x00 || UTF-8(principal)`.
std::string attach_ticket_aad(const std::string& principal);

// The fixed little-endian payload (spec §2.1). Throws `std::invalid_argument`
// for a field too long to encode or options over 16 KiB.
std::string encode_attach_ticket_payload(const AttachTicketClaims& claims);

// Seal `claims` for `principal`. `nonce` fixes the nonce **for vectors only**;
// production code leaves it empty and a fresh random one is drawn. Throws
// `std::invalid_argument` for an empty principal or catalog, a malformed
// ticket id, an empty lifetime, or a ticket that would exceed 32768 chars.
std::string mint_attach_ticket(const AttachTicketKey& key, const std::string& principal,
                               const AttachTicketClaims& claims,
                               const std::optional<std::array<uint8_t, 24>>& nonce = std::nullopt);

// `attach_ticket_invalid` / INVALID_ARGUMENT, with a `BadRequest` on
// `vgi_attach_ticket`. One type for every cause, so a forged ticket and another
// user's are indistinguishable. The message never contains the ticket.
class AttachTicketInvalidError : public vgi_rpc::StatusError {
public:
    explicit AttachTicketInvalidError(const std::string& detail = "attach ticket not accepted");
};

// `attach_ticket_expired` / FAILED_PRECONDITION, with a `PreconditionFailure`
// of type `ATTACH_TICKET`. Only ever raised for an authentic ticket presented
// by its own principal: the lifetime is inside the ciphertext.
class AttachTicketExpiredError : public vgi_rpc::StatusError {
public:
    explicit AttachTicketExpiredError(const std::string& detail = "attach ticket has expired");
};

// Verify `token` for the calling `principal` (empty = anonymous, which never
// opens a ticket) and return its claims. Order, normative: prefix, length,
// canonical base64url, caller, AEAD open, strict parse, lifetime (60 s skew).
// `now` overrides the clock, in Unix seconds.
AttachTicketClaims open_attach_ticket(const AttachTicketKey& key, const std::string& token,
                                      const std::string& principal,
                                      std::optional<double> now = std::nullopt);

// The caller's principal, or "" when anonymous.
std::string attach_ticket_principal(const vgi_rpc::AuthContext& auth);

// What `catalog_attach` does with `vgi_attach_ticket` (spec §6).
//
// `request` is a decoded `CatalogAttachRequest`. Returns null when its options
// carry no ticket (the request is untouched); otherwise the request the user
// originally made -- the sealed catalog name, options and version specs, with
// this request's `client_capabilities`. Throws `vgi_rpc::StatusError`
// (`invalid_request`) for another option beside the ticket, checked before the
// ticket is opened; `AttachTicketInvalidError` for a ticket that does not open
// for `principal` under `key` (or no key at all); `AttachTicketExpiredError`.
std::shared_ptr<arrow::RecordBatch> redeem_attach_ticket(
    const std::shared_ptr<arrow::RecordBatch>& request, const std::optional<AttachTicketKey>& key,
    const std::string& principal, std::optional<double> now = std::nullopt);

// Whether `name` is the reserved option name, compared case-insensitively.
bool is_reserved_attach_option(const std::string& name);

}  // namespace vgi
