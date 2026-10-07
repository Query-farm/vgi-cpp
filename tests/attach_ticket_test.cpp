// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Attach tickets against the shared vectors, vendored verbatim from
// vgi-python's vgi/_test_fixtures/attach_ticket_vectors.json. Every mint token
// is reproduced byte for byte; every accept, reject and redeem case is applied
// exactly as the spec (docs/protocol/vgi-attach-tickets.md) states.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <nlohmann/json.hpp>
#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/errors.h>
#include <vgi_rpc/server.h>

#include "attach_ticket_service.h"
#include "dispatcher.h"
#include "vgi/attach_ticket.h"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "wire.h"

#ifndef VGI_ATTACH_TICKET_VECTORS
#error "VGI_ATTACH_TICKET_VECTORS must name attach_ticket_vectors.json"
#endif

namespace {

using json = nlohmann::json;

const json& vectors() {
    static const json loaded = [] {
        // Read as bytes and parsed as UTF-8: the file is UTF-8 by contract.
        std::ifstream in(VGI_ATTACH_TICKET_VECTORS, std::ios::binary);
        REQUIRE(in.good());
        std::stringstream text;
        text << in.rdbuf();
        return json::parse(text.str());
    }();
    return loaded;
}

const json& defaults() {
    return vectors().at("defaults");
}

std::string hex(const std::string& bytes) {
    return vgi_rpc::crypto::hex_encode(reinterpret_cast<const uint8_t*>(bytes.data()),
                                       bytes.size());
}

vgi::AttachTicketKey key_of(const json& c) {
    const auto b64 = c.contains("signing_key_b64")
                         ? c.at("signing_key_b64").get<std::string>()
                         : defaults().at("signing_key_b64").get<std::string>();
    return vgi::attach_ticket_key(vgi::wire::base64_decode(b64));
}

double now_of(const json& c) {
    return c.contains("now") ? c.at("now").get<double>() : defaults().at("now").get<double>();
}

std::string kind_of(const std::function<void()>& call) {
    try {
        call();
    } catch (const vgi_rpc::KindedError& error) {
        return error.kind();
    }
    return "<accepted>";
}

// A `CatalogAttachRequest` whose options are `options` (all VARCHAR), as the
// engine sends it; `client_capabilities` set so redemption can be seen to keep it.
std::shared_ptr<arrow::RecordBatch> attach_request(const std::map<std::string, std::string>& opts,
                                                   const std::string& name = "twin_a") {
    arrow::FieldVector fields;
    std::vector<std::shared_ptr<arrow::Array>> values;
    for (const auto& [key, value] : opts) {
        arrow::StringBuilder builder;
        VGI_RPC_THROW_NOT_OK(builder.Append(value));
        fields.push_back(arrow::field(key, arrow::utf8(), /*nullable=*/true));
        values.push_back(vgi_rpc::unwrap(builder.Finish()));
    }
    const auto options = arrow::RecordBatch::Make(arrow::schema(fields), 1, values);

    arrow::StringBuilder name_builder;
    VGI_RPC_THROW_NOT_OK(name_builder.Append(name));
    std::shared_ptr<arrow::Array> name_array = vgi_rpc::unwrap(name_builder.Finish());
    arrow::BinaryBuilder options_builder;
    VGI_RPC_THROW_NOT_OK(options_builder.Append(vgi::wire::encode_ipc(options)));
    std::shared_ptr<arrow::Array> options_array = vgi_rpc::unwrap(options_builder.Finish());
    std::shared_ptr<arrow::Array> no_version =
        vgi_rpc::unwrap(arrow::MakeArrayOfNull(arrow::utf8(), 1));
    std::shared_ptr<arrow::Array> no_implementation =
        vgi_rpc::unwrap(arrow::MakeArrayOfNull(arrow::utf8(), 1));
    arrow::BinaryBuilder caps_builder;
    VGI_RPC_THROW_NOT_OK(caps_builder.Append("caps"));
    std::shared_ptr<arrow::Array> caps = vgi_rpc::unwrap(caps_builder.Finish());
    return arrow::RecordBatch::Make(
        vgi::generated::CatalogAttachRequestSchema(), 1,
        {name_array, options_array, no_version, no_implementation, caps});
}

std::optional<std::string> optional_text(const std::shared_ptr<arrow::RecordBatch>& batch,
                                         const std::string& field) {
    return vgi::wire::get_optional_string(batch, field);
}

}  // namespace

TEST_CASE("mint vectors reproduce byte for byte", "[attach_ticket][vectors]") {
    REQUIRE(vectors().at("mint").size() >= 1);
    for (const auto& c : vectors().at("mint")) {
        INFO(c.at("name").get<std::string>());
        vgi::AttachTicketClaims claims;
        claims.issued_at = c.at("issued_at").get<int64_t>();
        claims.expires_at = c.at("expires_at").get<int64_t>();
        claims.ticket_id = c.at("ticket_id").get<std::string>();
        claims.catalog_name = c.at("catalog_name").get<std::string>();
        claims.data_version_spec = c.at("data_version_spec").get<std::string>();
        claims.implementation_version = c.at("implementation_version").get<std::string>();
        claims.options_ipc = vgi::wire::base64_decode(c.at("options_ipc_b64").get<std::string>());

        const auto principal = c.at("principal").get<std::string>();
        CHECK(hex(vgi::attach_ticket_aad(principal)) == c.at("aad_hex").get<std::string>());
        CHECK(hex(vgi::encode_attach_ticket_payload(claims)) ==
              c.at("payload_hex").get<std::string>());

        const auto nonce_bytes = vgi_rpc::crypto::hex_decode(c.at("nonce_hex").get<std::string>());
        REQUIRE(nonce_bytes);
        REQUIRE(nonce_bytes->size() == 24);
        std::array<uint8_t, 24> nonce{};
        std::copy(nonce_bytes->begin(), nonce_bytes->end(), nonce.begin());
        CHECK(vgi::mint_attach_ticket(key_of(c), principal, claims, nonce) ==
              c.at("token").get<std::string>());
    }
}

TEST_CASE("accept vectors open to exactly their claims", "[attach_ticket][vectors]") {
    REQUIRE(vectors().at("accept").size() >= 1);
    for (const auto& c : vectors().at("accept")) {
        INFO(c.at("name").get<std::string>());
        const auto claims =
            vgi::open_attach_ticket(key_of(c), c.at("token").get<std::string>(),
                                    c.at("principal").get<std::string>(), now_of(c));
        const auto& expected = c.at("claims");
        CHECK(claims.issued_at == expected.at("issued_at").get<int64_t>());
        CHECK(claims.expires_at == expected.at("expires_at").get<int64_t>());
        CHECK(claims.ticket_id == expected.at("ticket_id").get<std::string>());
        CHECK(claims.catalog_name == expected.at("catalog_name").get<std::string>());
        CHECK(claims.data_version_spec == expected.at("data_version_spec").get<std::string>());
        CHECK(claims.implementation_version ==
              expected.at("implementation_version").get<std::string>());
        CHECK(claims.options_ipc ==
              vgi::wire::base64_decode(expected.at("options_ipc_b64").get<std::string>()));
    }
}

TEST_CASE("reject vectors are refused with exactly their error kind", "[attach_ticket][vectors]") {
    REQUIRE(vectors().at("reject").size() >= 1);
    for (const auto& c : vectors().at("reject")) {
        INFO(c.at("name").get<std::string>());
        const auto kind = kind_of([&] {
            (void)vgi::open_attach_ticket(key_of(c), c.at("token").get<std::string>(),
                                          c.at("principal").get<std::string>(), now_of(c));
        });
        CHECK(kind == c.at("error_kind").get<std::string>());
    }
}

TEST_CASE("redeem vectors restore the attach or refuse it", "[attach_ticket][vectors]") {
    REQUIRE(vectors().at("redeem").size() >= 1);
    const auto key = key_of(json::object());
    for (const auto& c : vectors().at("redeem")) {
        INFO(c.at("name").get<std::string>());
        std::map<std::string, std::string> options;
        for (const auto& [name, value] : c.at("options").items()) {
            options[name] = value.get<std::string>();
        }
        const auto request = attach_request(options);
        const auto principal = c.at("principal").get<std::string>();
        if (c.contains("error_kind")) {
            CHECK(kind_of([&] {
                      (void)vgi::redeem_attach_ticket(request, key, principal, now_of(c));
                  }) == c.at("error_kind").get<std::string>());
            continue;
        }
        const auto restored = vgi::redeem_attach_ticket(request, key, principal, now_of(c));
        const auto& result = c.at("result");
        if (result.is_null()) {
            CHECK(restored == nullptr);
            continue;
        }
        REQUIRE(restored != nullptr);
        CHECK(vgi::wire::get_string(restored, "name") ==
              result.at("catalog_name").get<std::string>());
        const auto expect_text = [&](const char* field) -> std::optional<std::string> {
            if (result.at(field).is_null()) return std::nullopt;
            return result.at(field).get<std::string>();
        };
        CHECK(optional_text(restored, "data_version_spec") == expect_text("data_version_spec"));
        CHECK(optional_text(restored, "implementation_version") ==
              expect_text("implementation_version"));
        CHECK(vgi::wire::get_optional_binary(restored, "client_capabilities") ==
              std::optional<std::string>("caps"));

        std::map<std::string, std::string> got;
        if (const auto encoded = vgi::wire::get_optional_binary(restored, "options")) {
            const auto batch = vgi::wire::decode_ipc(*encoded);
            for (int i = 0; i < batch->num_columns(); ++i) {
                got[batch->schema()->field(i)->name()] =
                    vgi::wire::get_string(batch, batch->schema()->field(i)->name());
            }
        }
        std::map<std::string, std::string> want;
        for (const auto& [name, value] : result.at("options").items()) {
            want[name] = value.get<std::string>();
        }
        CHECK(got == want);
    }
}

TEST_CASE("without a signing key no ticket opens", "[attach_ticket]") {
    const auto& c = vectors().at("redeem").at(0);
    std::map<std::string, std::string> options;
    for (const auto& [name, value] : c.at("options").items()) options[name] = value;
    CHECK(kind_of([&] {
              (void)vgi::redeem_attach_ticket(attach_request(options), std::nullopt, "alice",
                                              now_of(c));
          }) == "attach_ticket_invalid");
}

TEST_CASE("a fresh ticket round-trips and expires", "[attach_ticket]") {
    const auto key = vgi::attach_ticket_key("a-signing-key");
    vgi::AttachTicketClaims claims;
    claims.issued_at = 1000;
    claims.expires_at = 2000;
    claims.ticket_id = std::string(32, 'a');
    claims.catalog_name = "ticket_probe";
    const auto token = vgi::mint_attach_ticket(key, "alice", claims);
    CHECK(vgi::open_attach_ticket(key, token, "alice", 1500.0).catalog_name == "ticket_probe");
    CHECK(kind_of([&] { (void)vgi::open_attach_ticket(key, token, "alice", 2060.0); }) ==
          "attach_ticket_expired");
    CHECK(kind_of([&] { (void)vgi::open_attach_ticket(key, token, "alice", 2059.0); }) ==
          "<accepted>");
    CHECK(kind_of([&] { (void)vgi::open_attach_ticket(key, token, "bob", 1500.0); }) ==
          "attach_ticket_invalid");
    // Two mints of the same claims never share a nonce.
    CHECK(vgi::mint_attach_ticket(key, "alice", claims) != token);
}

TEST_CASE("vgi_attach_ticket is a reserved attach-option name", "[attach_ticket]") {
    CHECK(vgi::is_reserved_attach_option("vgi_attach_ticket"));
    CHECK(vgi::is_reserved_attach_option("VGI_Attach_Ticket"));
    CHECK_FALSE(vgi::is_reserved_attach_option("api_key"));

    vgi::Dispatcher dispatcher;
    vgi::CatalogModel model;
    model.name = "reserved";
    model.attach_options = {{"VGI_ATTACH_TICKET", "nope", arrow::utf8(), nullptr, false}};
    dispatcher.set_catalog(std::move(model));
    vgi_rpc::ServerBuilder builder;
    CHECK_THROWS_AS(dispatcher.install(builder), std::invalid_argument);
}

TEST_CASE("the ticket lifetime ceiling is the grant maximum", "[attach_ticket]") {
    CHECK(vgi::attach_ticket_max_ttl(3600) == std::optional<int64_t>(3600));
    ::unsetenv("VGI_RPC_GRANT_MAX_TTL_SECONDS");
    CHECK(vgi::attach_ticket_max_ttl(std::nullopt) == std::nullopt);
    ::setenv("VGI_RPC_GRANT_MAX_TTL_SECONDS", "120", 1);
    CHECK(vgi::attach_ticket_max_ttl(std::nullopt) == std::optional<int64_t>(120));
    ::setenv("VGI_RPC_GRANT_MAX_TTL_SECONDS", "0", 1);
    CHECK_THROWS_AS(vgi::attach_ticket_max_ttl(std::nullopt), std::invalid_argument);
    ::unsetenv("VGI_RPC_GRANT_MAX_TTL_SECONDS");
}
