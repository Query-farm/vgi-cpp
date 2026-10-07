// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The vgi-rpc cross-SDK fixtures this worker hosts beside vgi.v2.
//
// `conformance.Secondary.v1` is the shared fixture protocol every vgi-rpc
// conformance worker and every VGI SDK fixture worker hosts
// (vgi-rpc's tools/cross-port/specs/MULTI_PROTOCOL_HOSTING.md §2). It reaches
// the server through Worker::set_hosted_protocols -- the public hook, not a
// special case -- so `vgi-rpc-test-hosted --expect vgi.v2,conformance.Secondary.v1`
// checks the hook on every transport.
//
// The identity policy is IDENTITY_CONFORMANCE_FIXTURE.md's, enabled only by
// `--conformance-identity` (or VGI_FIXTURE_IDENTITY=1) on `--http`. It trusts a
// spoofable X-Conformance-Principal header for authentication: a test fixture
// that must never be deployed.

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#include <arrow/builder.h>
#include <arrow/type.h>
#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/errors.h>
#include <vgi_rpc/grants.h>
#include <vgi_rpc/http_config.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/token_identity.h>
#include <vgi/worker.h>

#include "registry.h"

namespace example {

namespace {

using json = nlohmann::json;

constexpr const char* kSecondaryName = "conformance.Secondary.v1";
constexpr const char* kEchoPrefix = "secondary:";
constexpr const char* kProbeType = "conformance.Secondary.v1.Probe";

std::shared_ptr<arrow::Schema> utf8_result() {
    static auto s = arrow::schema({arrow::field("result", arrow::utf8(), /*nullable=*/false)});
    return s;
}

std::shared_ptr<arrow::Field> required(const std::string& name,
                                       std::shared_ptr<arrow::DataType> type) {
    return arrow::field(name, std::move(type), /*nullable=*/false);
}

json fail_details(double retry_delay_seconds) {
    auto details = json::array();
    details.push_back(vgi_rpc::ErrorInfo{{{"fixture", kSecondaryName}}}.to_json());
    if (retry_delay_seconds > 0) {
        details.push_back(vgi_rpc::RetryInfo{retry_delay_seconds}.to_json());
    }
    details.push_back(
        json{{"@type", kProbeType}, {"note", "clients ignore detail types they do not know"}});
    return details;
}

vgi_rpc::ProtocolBuilder secondary_protocol() {
    using vgi_rpc::CallContext;
    using vgi_rpc::Code;
    using vgi_rpc::Request;
    using vgi_rpc::StatusError;

    // No protocol_version, deliberately: a server gating every call against
    // vgi.v2's version would refuse these.
    vgi_rpc::ProtocolBuilder secondary(kSecondaryName);
    secondary.add_unary(
        "echo_string", arrow::schema({required("value", arrow::utf8())}), utf8_result(),
        [](const Request& req, CallContext&) -> vgi_rpc::Result {
            arrow::StringBuilder builder;
            VGI_RPC_THROW_NOT_OK(builder.Append(kEchoPrefix + req.get<std::string>("value")));
            return vgi_rpc::Result::value(utf8_result(), {vgi_rpc::unwrap(builder.Finish())});
        },
        "Echo with the prefix that makes a mis-route visible.");
    secondary.add_void(
        "fail",
        arrow::schema({required("code", arrow::utf8()), required("kind", arrow::utf8()),
                       required("retry_delay_seconds", arrow::float64())}),
        [](const Request& req, CallContext&) {
            const auto code_text = req.get<std::string>("code");
            const auto kind = req.get<std::string>("kind");
            const auto delay = req.get<double>("retry_delay_seconds");
            const auto code = vgi_rpc::code_from_name(code_text);
            if (!code) {
                throw StatusError(
                    "'" + code_text + "' is not a canonical error code", Code::INVALID_ARGUMENT,
                    "invalid_code",
                    {vgi_rpc::BadRequest{{{"code", "must be a canonical code name"}}}});
            }
            std::string message = std::string(kSecondaryName) + " fail: " + code_text;
            if (!kind.empty()) message += " " + kind;
            throw StatusError(message, *code, kind, fail_details(delay));
        },
        "Raise an error with the requested code, kind and the fixed details.");
    secondary.add_void(
        "fail_oversized", arrow::schema(arrow::FieldVector{}),
        [](const Request&, CallContext&) {
            throw StatusError(std::string(kSecondaryName) + " fail_oversized: details exceed 4 KiB",
                              Code::RESOURCE_EXHAUSTED, "details_oversized",
                              {vgi_rpc::RetryInfo{1.0},
                               vgi_rpc::ErrorInfo{{{"padding", std::string(5000, 'x')}}}});
        },
        "Raise an error whose details exceed the 4 KiB cap.");
    return secondary;
}

// IDENTITY_CONFORMANCE_FIXTURE.md's policy, verbatim.
constexpr const char* kSubject = "subject@conformance.example";
constexpr const char* kSubjectTokenName = "conformance-subject";
constexpr double kGrantExpiresAt = 1893456000.0;

std::optional<vgi_rpc::TokenIdentity> resolve_token(const std::string& token) {
    if (token == "conformance-unavailable-token") {
        throw vgi_rpc::IdentityUnavailableError("conformance: mapping store unreachable", 5);
    }
    if (token == "conformance-auth-unavailable-token") {
        // The transport-auth error, which the framework must translate.
        throw vgi_rpc::AuthUnavailableError("conformance: authority unreachable", 7);
    }
    if (token == "conformance-unknown-token") return std::nullopt;
    if (token == "conformance-zero-ttl-token") {
        return vgi_rpc::TokenIdentity{kSubject, kSubjectTokenName, 0};
    }
    if (token == "conformance-minimal-token") {
        vgi_rpc::TokenIdentity identity;
        identity.principal = kSubject;
        return identity;
    }
    if (token == "  conformance-padded-probe  ") {
        return vgi_rpc::TokenIdentity{kSubject, "conformance-padded", 300};
    }
    return vgi_rpc::TokenIdentity{kSubject, kSubjectTokenName, 300};
}

vgi_rpc::IssuedGrant mint_grant(const std::string& principal, const std::string& purpose,
                                const std::vector<std::string>& scopes, int64_t) {
    if (purpose == "conformance-auth-unavailable") {
        throw vgi_rpc::AuthUnavailableError("conformance: grant store unreachable", 7);
    }
    if (purpose == "conformance-refused") {
        throw vgi_rpc::GrantRefusedError("conformance: this purpose is refused");
    }
    std::string token = "conformance-grant-for:" + principal + "|";
    for (size_t i = 0; i < scopes.size(); ++i) {
        if (i != 0) token += ",";
        token += scopes[i];
    }
    if (purpose == "conformance-minimal") {
        vgi_rpc::IssuedGrant grant;
        grant.token = std::move(token);
        grant.expires_at = kGrantExpiresAt;
        return grant;
    }
    return vgi_rpc::IssuedGrant{std::move(token), kGrantExpiresAt, "conformance-grant-id"};
}

// vgi-python's `_optional_bearer`: `vgi-test-alice` / `vgi-test-bob` are
// fresh logins (auth_time = now, so issue_grant's freshness guard passes); a
// sealed grant is not ours, and falls through to the grant authenticator; any
// other bearer stays anonymous. One difference vgi-rpc-cpp imposes: with a
// bearer authenticator configured, a request with *no* Authorization header is
// a 401 rather than anonymous, so this is opt-in (VGI_FIXTURE_TEST_BEARERS=1).
std::optional<vgi_rpc::AuthContext> test_bearer(const std::string& token) {
    if (token.rfind(vgi_rpc::kGrantTokenPrefix, 0) == 0) return std::nullopt;
    std::string principal;
    if (token == "vgi-test-alice") principal = "alice";
    if (token == "vgi-test-bob") principal = "bob";
    if (principal.empty()) return vgi_rpc::AuthContext::anonymous();
    vgi_rpc::AuthContext auth;
    auth.domain = "bearer";
    auth.authenticated = true;
    auth.principal = principal;
    auth.claims = nlohmann::json{
        {"auth_time",
         std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
             .count()}};
    return auth;
}

}  // namespace

void register_conformance_fixtures(vgi::Worker& worker, int argc, char** argv) {
    worker.set_hosted_protocols(
        [] { return std::vector<vgi_rpc::ProtocolBuilder>{secondary_protocol()}; });

    bool identity = false;
    bool principal_header = false;
    if (const char* env = std::getenv("VGI_FIXTURE_IDENTITY"); env && std::string(env) == "1") {
        identity = true;
    }
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--conformance-identity") identity = true;
        // The fixture's header authentication alone, with no identity hooks:
        // with `--grant-key`, the framework is then the minter, and a caller
        // with a fresh X-Conformance-Auth-Time can mint a sealed grant.
        if (std::string(argv[i]) == "--conformance-principal-header") principal_header = true;
    }
    // vgi-python's fixture test bearers (vgi-test-alice / vgi-test-bob), as
    // fresh logins, so an ATTACH can mint a grant with nothing but a bearer.
    const char* bearers_env = std::getenv("VGI_FIXTURE_TEST_BEARERS");
    const bool test_bearers = bearers_env && std::string(bearers_env) == "1";
    const bool header_auth = identity || principal_header;
    if (identity) {
        worker.set_resolve_token(resolve_token);
        worker.set_mint_grant(mint_grant);
        // The allowlist is deliberately *not* set here: the fixture takes it
        // the way a deployment does (--introspect-principals
        // conformance-introspector, or VGI_INTROSPECT_PRINCIPALS), so starting
        // it without one exercises the refuse-to-start rule.
    }
    if (!header_auth && !test_bearers) return;
    worker.configure_http([header_auth, test_bearers](vgi_rpc::HttpConfig& config) {
        // The fixture's spoofable header authentication (X-Conformance-Principal,
        // X-Conformance-Auth-Time). Test-only.
        if (header_auth) config.sticky_header_auth = true;
        if (test_bearers) config.bearer_authenticate = test_bearer;
    });
}

}  // namespace example
