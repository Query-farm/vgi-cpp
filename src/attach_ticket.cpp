// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "vgi/attach_ticket.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/type.h>
#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/crypto.h>

#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "wire.h"

namespace vgi {

namespace {

namespace crypto = vgi_rpc::crypto;

constexpr char kAadDomain[] = "vgi.attach_ticket.v1";
constexpr size_t kMaxText = 0xFFFF;
constexpr size_t kNonceBytes = crypto::kAeadNonceBytes;
constexpr size_t kTagBytes = crypto::kAeadTagBytes;

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool is_ticket_id(const std::string& id) {
    return id.size() == 32 && std::all_of(id.begin(), id.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

void put_u16(std::string& out, size_t value) {
    out.push_back(static_cast<char>(value & 0xFF));
    out.push_back(static_cast<char>((value >> 8) & 0xFF));
}

void put_u32(std::string& out, size_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

void put_i64(std::string& out, int64_t value) {
    const auto bits = static_cast<uint64_t>(value);
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
}

void put_text(std::string& out, const std::string& value, const char* field) {
    if (value.size() > kMaxText) {
        throw std::invalid_argument(std::string(field) + " is longer than 65535 bytes");
    }
    put_u16(out, value.size());
    out += value;
}

// Strict UTF-8: no overlongs, no surrogates, nothing past U+10FFFF.
bool valid_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        size_t extra = 0;
        uint32_t cp = 0;
        if (c < 0x80) {
            ++i;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            extra = 1;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            extra = 2;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            extra = 3;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (i + extra >= s.size()) return false;
        for (size_t k = 1; k <= extra; ++k) {
            const auto cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) ||
            (extra == 3 && cp < 0x10000) || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        i += extra + 1;
    }
    return true;
}

// The parse is strict: exact lengths, strict UTF-8, the field rules, and no
// trailing bytes. Every failure is the one `attach_ticket_invalid`.
class Reader {
public:
    explicit Reader(const std::string& payload) : payload_(payload) {}

    std::string take(size_t n) {
        if (n > payload_.size() - pos_) {
            throw AttachTicketInvalidError("attach ticket payload is truncated");
        }
        auto chunk = payload_.substr(pos_, n);
        pos_ += n;
        return chunk;
    }
    uint64_t take_le(size_t n) {
        const auto bytes = take(n);
        uint64_t value = 0;
        for (size_t i = 0; i < n; ++i) {
            value |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
        }
        return value;
    }
    std::string text() {
        const auto length = static_cast<size_t>(take_le(2));
        auto value = take(length);
        if (!valid_utf8(value))
            throw AttachTicketInvalidError("attach ticket payload is not UTF-8");
        return value;
    }
    bool done() const { return pos_ == payload_.size(); }

private:
    const std::string& payload_;
    size_t pos_ = 0;
};

AttachTicketClaims decode_payload(const std::string& payload) {
    Reader in(payload);
    AttachTicketClaims claims;
    claims.issued_at = static_cast<int64_t>(in.take_le(8));
    claims.expires_at = static_cast<int64_t>(in.take_le(8));
    claims.ticket_id = in.text();
    claims.catalog_name = in.text();
    claims.data_version_spec = in.text();
    claims.implementation_version = in.text();
    const auto options_len = static_cast<size_t>(in.take_le(4));
    if (options_len > kAttachTicketMaxOptionsBytes) {
        throw AttachTicketInvalidError("attach ticket options exceed 16 KiB");
    }
    claims.options_ipc = in.take(options_len);
    if (!in.done()) throw AttachTicketInvalidError("attach ticket payload has trailing bytes");
    if (!is_ticket_id(claims.ticket_id)) {
        throw AttachTicketInvalidError("attach ticket id is not 32 lowercase hex");
    }
    if (claims.catalog_name.empty())
        throw AttachTicketInvalidError("attach ticket names no catalog");
    if (claims.expires_at != 0 && claims.expires_at <= claims.issued_at) {
        throw AttachTicketInvalidError("attach ticket lifetime is empty");
    }
    return claims;
}

// Unpadded base64url, and only its canonical spelling: the alphabet, no
// padding, a length that is not 1 mod 4, and zero trailing bits (re-encoding
// must give back the same text).
std::string b64url_strict(const std::string& text) {
    const bool alphabet = !text.empty() && std::all_of(text.begin(), text.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
    });
    if (!alphabet || text.size() % 4 == 1) {
        throw AttachTicketInvalidError("attach ticket is not unpadded base64url");
    }
    auto raw = crypto::base64url_decode(text);
    if (!raw || crypto::base64url_encode(*raw) != text) {
        throw AttachTicketInvalidError("attach ticket is not canonical base64url");
    }
    return std::move(*raw);
}

double wall_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::optional<std::string> string_value(const std::shared_ptr<arrow::Array>& column) {
    if (!column || column->length() < 1 || column->IsNull(0)) return std::nullopt;
    switch (column->type_id()) {
        case arrow::Type::STRING:
            return std::static_pointer_cast<arrow::StringArray>(column)->GetString(0);
        case arrow::Type::LARGE_STRING:
            return std::static_pointer_cast<arrow::LargeStringArray>(column)->GetString(0);
        case arrow::Type::STRING_VIEW:
            return std::string(
                std::static_pointer_cast<arrow::StringViewArray>(column)->GetView(0));
        default: return std::nullopt;
    }
}

[[noreturn]] void invalid_request(const std::string& message,
                                  std::vector<vgi_rpc::FieldViolation> violations) {
    throw vgi_rpc::StatusError(message, vgi_rpc::Code::INVALID_ARGUMENT, "invalid_request",
                               {vgi_rpc::BadRequest{std::move(violations)}});
}

}  // namespace

AttachTicketInvalidError::AttachTicketInvalidError(const std::string& detail)
    : vgi_rpc::StatusError(detail, vgi_rpc::Code::INVALID_ARGUMENT, "attach_ticket_invalid",
                           {vgi_rpc::BadRequest{{{kAttachTicketOption, detail}}}}) {}

AttachTicketExpiredError::AttachTicketExpiredError(const std::string& detail)
    : vgi_rpc::StatusError(
          detail, vgi_rpc::Code::FAILED_PRECONDITION, "attach_ticket_expired",
          {vgi_rpc::PreconditionFailure{{{"ATTACH_TICKET", kAttachTicketOption, detail}}}}) {}

AttachTicketKey attach_ticket_key(const std::string& raw) {
    AttachTicketKey key{};
    if (raw.size() == key.size()) {
        std::memcpy(key.data(), raw.data(), key.size());
        return key;
    }
    crypto::Sha256 hash;
    hash.update(raw);
    return hash.digest();
}

std::string attach_ticket_aad(const std::string& principal) {
    // The terminating NUL of the domain literal is part of the AAD.
    return std::string(kAadDomain, sizeof(kAadDomain)) + principal;
}

std::string encode_attach_ticket_payload(const AttachTicketClaims& claims) {
    if (claims.options_ipc.size() > kAttachTicketMaxOptionsBytes) {
        throw std::invalid_argument("options are " + std::to_string(claims.options_ipc.size()) +
                                    " bytes; a ticket carries at most 16384");
    }
    std::string out;
    put_i64(out, claims.issued_at);
    put_i64(out, claims.expires_at);
    put_text(out, claims.ticket_id, "ticket_id");
    put_text(out, claims.catalog_name, "catalog_name");
    put_text(out, claims.data_version_spec, "data_version_spec");
    put_text(out, claims.implementation_version, "implementation_version");
    put_u32(out, claims.options_ipc.size());
    out += claims.options_ipc;
    return out;
}

std::string mint_attach_ticket(const AttachTicketKey& key, const std::string& principal,
                               const AttachTicketClaims& claims,
                               const std::optional<std::array<uint8_t, 24>>& nonce) {
    if (principal.empty()) throw std::invalid_argument("a ticket needs a principal");
    if (claims.catalog_name.empty()) throw std::invalid_argument("a ticket needs a catalog name");
    if (!is_ticket_id(claims.ticket_id)) {
        throw std::invalid_argument("ticket_id must be 32 lowercase hex");
    }
    if (claims.expires_at != 0 && claims.expires_at <= claims.issued_at) {
        throw std::invalid_argument("expires_at must be 0 or after issued_at");
    }
    const auto payload = encode_attach_ticket_payload(claims);
    const auto aad = attach_ticket_aad(principal);
    std::string envelope(1, static_cast<char>(kAttachTicketEnvelopeVersion));
    envelope += nonce ? crypto::aead_seal_with_nonce(key, payload, aad, *nonce)
                      : crypto::aead_seal(key, payload, aad);
    auto token = std::string(kAttachTicketPrefix) + crypto::base64url_encode(envelope);
    if (token.size() > kAttachTicketMaxChars) {
        throw std::invalid_argument("the ticket would be " + std::to_string(token.size()) +
                                    " characters; at most 32768 are accepted");
    }
    return token;
}

AttachTicketClaims open_attach_ticket(const AttachTicketKey& key, const std::string& token,
                                      const std::string& principal, std::optional<double> now) {
    const std::string prefix(kAttachTicketPrefix);
    if (token.compare(0, prefix.size(), prefix) != 0) {
        throw AttachTicketInvalidError("not an attach ticket");
    }
    if (token.size() > kAttachTicketMaxChars) {
        throw AttachTicketInvalidError("attach ticket is too long");
    }
    const auto envelope = b64url_strict(token.substr(prefix.size()));
    if (principal.empty()) {
        throw AttachTicketInvalidError("an anonymous caller cannot redeem an attach ticket");
    }
    if (envelope.size() < 1 + kNonceBytes + kTagBytes ||
        static_cast<uint8_t>(envelope[0]) != kAttachTicketEnvelopeVersion) {
        throw AttachTicketInvalidError("attach ticket failed verification");
    }
    // The AAD is the principal check: there is no separate comparison.
    const auto payload = crypto::aead_open(key, envelope.substr(1), attach_ticket_aad(principal));
    if (!payload) throw AttachTicketInvalidError("attach ticket failed verification");
    auto claims = decode_payload(*payload);
    const double current = now ? *now : wall_seconds();
    if (static_cast<double>(claims.issued_at) > current + kAttachTicketClockSkewSeconds) {
        throw AttachTicketExpiredError("attach ticket is not yet valid");
    }
    if (claims.expires_at != 0 &&
        current >= static_cast<double>(claims.expires_at) + kAttachTicketClockSkewSeconds) {
        throw AttachTicketExpiredError("attach ticket has expired");
    }
    return claims;
}

std::string attach_ticket_principal(const vgi_rpc::AuthContext& auth) {
    if (!auth.authenticated || !auth.principal) return {};
    return *auth.principal;
}

bool is_reserved_attach_option(const std::string& name) {
    return lower(name) == kAttachTicketOption;
}

std::shared_ptr<arrow::RecordBatch> redeem_attach_ticket(
    const std::shared_ptr<arrow::RecordBatch>& request, const std::optional<AttachTicketKey>& key,
    const std::string& principal, std::optional<double> now) {
    if (!request) return nullptr;
    const auto encoded = wire::get_optional_binary(request, "options");
    if (!encoded || encoded->empty()) return nullptr;
    const auto options = wire::decode_ipc(*encoded);
    if (!options || options->num_rows() < 1) return nullptr;

    // The first spelling of the ticket key is the ticket; anything else --
    // another option, or a second spelling of the same key -- is refused
    // before the ticket is opened. The sealed options are authoritative, so
    // there is nothing to merge.
    int ticket_column = -1;
    for (int i = 0; i < options->num_columns(); ++i) {
        if (is_reserved_attach_option(options->schema()->field(i)->name())) {
            ticket_column = i;
            break;
        }
    }
    if (ticket_column < 0) return nullptr;
    std::vector<vgi_rpc::FieldViolation> others;
    for (int i = 0; i < options->num_columns(); ++i) {
        if (i == ticket_column) continue;
        others.push_back({"options." + options->schema()->field(i)->name(),
                          std::string("not allowed alongside ") + kAttachTicketOption});
    }
    if (!others.empty()) {
        invalid_request(std::string(kAttachTicketOption) + " must be the only attach option",
                        std::move(others));
    }

    const auto token = string_value(options->column(ticket_column));
    if (!token)
        throw AttachTicketInvalidError(std::string(kAttachTicketOption) + " must be a string");
    if (!key) throw AttachTicketInvalidError("this worker does not redeem attach tickets");
    const auto claims = open_attach_ticket(*key, *token, principal, now);

    std::shared_ptr<arrow::Array> restored_options;
    if (claims.options_ipc.empty()) {
        restored_options = vgi_rpc::unwrap(arrow::MakeArrayOfNull(arrow::binary(), 1));
    } else {
        try {
            (void)wire::decode_ipc(claims.options_ipc);
        } catch (const std::exception&) {
            throw AttachTicketInvalidError("attach ticket options are not an Arrow IPC record");
        }
        arrow::BinaryBuilder builder;
        VGI_RPC_THROW_NOT_OK(builder.Append(claims.options_ipc));
        restored_options = vgi_rpc::unwrap(builder.Finish());
    }
    const auto text_or_null = [](const std::string& value) {
        arrow::StringBuilder builder;
        if (value.empty()) {
            VGI_RPC_THROW_NOT_OK(builder.AppendNull());
        } else {
            VGI_RPC_THROW_NOT_OK(builder.Append(value));
        }
        return vgi_rpc::unwrap(builder.Finish());
    };
    arrow::StringBuilder name_builder;
    VGI_RPC_THROW_NOT_OK(name_builder.Append(claims.catalog_name));
    std::shared_ptr<arrow::Array> name = vgi_rpc::unwrap(name_builder.Finish());
    std::shared_ptr<arrow::Array> data_version = text_or_null(claims.data_version_spec);
    std::shared_ptr<arrow::Array> implementation = text_or_null(claims.implementation_version);

    // Every incoming column is kept -- `client_capabilities` above all -- and
    // the four the ticket seals are replaced. The incoming name is ignored.
    const auto schema = request->schema();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (int i = 0; i < schema->num_fields(); ++i) {
        const auto& field = schema->field(i);
        const auto& field_name = field->name();
        std::shared_ptr<arrow::Array> column;
        if (field_name == "name") {
            column = name;
        } else if (field_name == "options") {
            column = restored_options;
        } else if (field_name == "data_version_spec") {
            column = data_version;
        } else if (field_name == "implementation_version") {
            column = implementation;
        } else {
            column = request->column(i)->Slice(0, 1);
        }
        fields.push_back(field);
        columns.push_back(std::move(column));
    }
    auto restored = arrow::RecordBatch::Make(arrow::schema(fields), 1, std::move(columns));
    VGI_RPC_THROW_NOT_OK(restored->ValidateFull());
    return restored;
}

}  // namespace vgi
