// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "attach_ticket_service.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/type.h>
#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/errors.h>
#include <vgi_rpc/grants.h>

#include "wire.h"

namespace vgi {

namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::shared_ptr<arrow::Schema> request_params() {
    static const auto schema =
        arrow::schema({arrow::field("request", arrow::binary(), /*nullable=*/false)});
    return schema;
}

std::shared_ptr<arrow::Schema> result_envelope() {
    static const auto schema =
        arrow::schema({arrow::field("result", arrow::binary(), /*nullable=*/false)});
    return schema;
}

// `AttachTicket`, as vgi-python's dataclass serializes it.
std::shared_ptr<arrow::Schema> attach_ticket_schema() {
    static const auto schema =
        arrow::schema({arrow::field("ticket", arrow::utf8(), /*nullable=*/false),
                       arrow::field("expires_at", arrow::float64(), /*nullable=*/false)});
    return schema;
}

[[noreturn]] void action_denied(const std::string& message) {
    throw vgi_rpc::StatusError(message, vgi_rpc::Code::PERMISSION_DENIED, "action_denied",
                               {vgi_rpc::ErrorInfo{{{"action", "seal_attach"}}}});
}

std::string random_ticket_id() {
    const auto bytes = vgi_rpc::crypto::random_bytes(16);
    return vgi_rpc::crypto::hex_encode(bytes.data(), bytes.size());
}

vgi_rpc::Result seal_attach(const DeclaredAttachOptions& declared, const AttachTicketKey& key,
                            std::optional<int64_t> ceiling, const vgi_rpc::Request& req,
                            vgi_rpc::CallContext& ctx) {
    // 1. The caller. The principal sealed is always the caller's; there is no
    //    subject parameter. No freshness requirement: a ticket is no authority.
    const auto principal = attach_ticket_principal(ctx.auth());
    if (principal.empty()) action_denied("an anonymous caller cannot seal an attach ticket");

    const auto request = wire::get_ipc(req.batch(), "request");
    if (!request) throw std::invalid_argument("seal_attach: empty request");
    const auto catalog_name = wire::get_string(request, "catalog_name");
    const auto encoded = wire::get_optional_binary(request, "options");
    const auto data_version_spec = wire::get_string(request, "data_version_spec");
    const auto implementation_version = wire::get_string(request, "implementation_version");
    const auto ttl = wire::get_int64(request, "ttl_seconds");

    // 3. Validation: every violation, reported together; never attaches.
    std::vector<vgi_rpc::FieldViolation> violations;
    if (ttl < 0)
        violations.push_back({"ttl_seconds", "must be 0 (as long as allowed) or positive"});

    std::shared_ptr<arrow::RecordBatch> options;
    if (encoded && !encoded->empty()) options = wire::decode_ipc(*encoded);
    std::vector<std::string> names;
    if (options) {
        if (options->num_rows() > 1) {
            violations.push_back({"options", "must be a one-row record"});
        } else if (options->num_rows() == 1) {
            for (const auto& field : options->schema()->fields()) names.push_back(field->name());
        }
    }

    const auto specs = declared(catalog_name);
    if (!specs) {
        violations.push_back({"catalog_name", "no catalog named '" + catalog_name + "'"});
    } else {
        std::set<std::string> known;
        for (const auto& spec : *specs) known.insert(lower(spec.name));
        for (const auto& name : names) {
            if (is_reserved_attach_option(name)) {
                violations.push_back({"options." + name, "a ticket cannot seal another ticket"});
            } else if (!known.count(lower(name))) {
                violations.push_back(
                    {"options." + name, "not an attach option this catalog declares"});
            }
        }
        std::set<std::string> supplied;
        for (const auto& name : names) supplied.insert(lower(name));
        for (const auto& spec : *specs) {
            if (spec.required && !supplied.count(lower(spec.name))) {
                violations.push_back({"options." + spec.name, "required"});
            }
        }
    }

    // The bytes the caller sent: exactly `CatalogAttachRequest.options`.
    const std::string options_ipc = !names.empty() ? *encoded : std::string{};
    if (options_ipc.size() > kAttachTicketMaxOptionsBytes) {
        violations.push_back({"options", std::to_string(options_ipc.size()) +
                                             " bytes; a ticket carries at most 16384"});
    }
    if (!violations.empty()) {
        throw vgi_rpc::StatusError("seal_attach request is invalid",
                                   vgi_rpc::Code::INVALID_ARGUMENT, "invalid_request",
                                   {vgi_rpc::BadRequest{std::move(violations)}});
    }

    // 4. Lifetime: 0 asks for the ceiling; otherwise the request, capped at it.
    AttachTicketClaims claims;
    claims.issued_at = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    std::optional<int64_t> lifetime;
    if (ttl == 0) {
        lifetime = ceiling;
    } else {
        lifetime = ceiling ? std::min(ttl, *ceiling) : ttl;
    }
    claims.expires_at = lifetime ? claims.issued_at + *lifetime : 0;
    claims.ticket_id = random_ticket_id();
    claims.catalog_name = catalog_name;
    claims.data_version_spec = data_version_spec;
    claims.implementation_version = implementation_version;
    claims.options_ipc = options_ipc;

    // 5. Seal.
    std::string token;
    try {
        token = mint_attach_ticket(key, principal, claims);
    } catch (const std::invalid_argument& error) {
        throw vgi_rpc::StatusError("seal_attach request is invalid",
                                   vgi_rpc::Code::INVALID_ARGUMENT, "invalid_request",
                                   {vgi_rpc::BadRequest{{{"request", error.what()}}}});
    }

    arrow::StringBuilder ticket_builder;
    VGI_RPC_THROW_NOT_OK(ticket_builder.Append(token));
    std::shared_ptr<arrow::Array> ticket = vgi_rpc::unwrap(ticket_builder.Finish());
    arrow::DoubleBuilder expires_builder;
    VGI_RPC_THROW_NOT_OK(expires_builder.Append(claims.expires_at == 0
                                                    ? std::numeric_limits<double>::infinity()
                                                    : static_cast<double>(claims.expires_at)));
    std::shared_ptr<arrow::Array> expires = vgi_rpc::unwrap(expires_builder.Finish());
    const auto payload = arrow::RecordBatch::Make(attach_ticket_schema(), 1, {ticket, expires});

    arrow::BinaryBuilder result_builder;
    VGI_RPC_THROW_NOT_OK(result_builder.Append(wire::encode_ipc(payload)));
    std::shared_ptr<arrow::Array> result = vgi_rpc::unwrap(result_builder.Finish());
    return vgi_rpc::Result::value(result_envelope(), {result});
}

}  // namespace

vgi_rpc::ProtocolBuilder attach_tickets_protocol(DeclaredAttachOptions declared,
                                                 AttachTicketKey key,
                                                 std::optional<int64_t> max_ttl_seconds) {
    vgi_rpc::ProtocolBuilder protocol(kAttachTicketsProtocolName, kAttachTicketsProtocolVersion);
    protocol.add_unary(
        "seal_attach", request_params(), result_envelope(),
        [declared = std::move(declared), key, max_ttl_seconds](const vgi_rpc::Request& req,
                                                               vgi_rpc::CallContext& ctx) {
            return seal_attach(declared, key, max_ttl_seconds, req, ctx);
        },
        "Seal the caller's attach of request.catalog_name into a ticket.");
    return protocol;
}

std::optional<int64_t> attach_ticket_max_ttl(std::optional<int64_t> grant_keys_max_ttl) {
    if (grant_keys_max_ttl) return grant_keys_max_ttl;
    const char* raw = std::getenv(vgi_rpc::kGrantMaxTtlEnv);
    if (!raw) return std::nullopt;
    std::string text(raw);
    const auto first = text.find_first_not_of(" \t");
    if (first == std::string::npos) return std::nullopt;
    text = text.substr(first, text.find_last_not_of(" \t") - first + 1);
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || value <= 0) {
        throw std::invalid_argument(std::string(vgi_rpc::kGrantMaxTtlEnv) + "='" + text +
                                    "' must be a positive integer");
    }
    return value;
}

}  // namespace vgi
