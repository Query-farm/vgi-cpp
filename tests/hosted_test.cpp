// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Orchard hosted mode's Pass verifier and environment parsing, in-process
// with a fake JWKS endpoint and injected clocks. The black-box contract over
// HTTP is the extension repo's `test/hosted/` harness; this pins the rules
// that need a clock or a counter to see: the JWKS cache bound, fail-closed
// expiry, and the unknown-kid re-fetch limit.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <jwt-cpp/jwt.h>
#include <nlohmann/json.hpp>

#include <vgi_rpc/errors.h>
#include <vgi_rpc/http_config.h>

#include "vgi/hosted.h"
#include "pass_minter.h"

namespace {

using pass_minter::b64u;
using pass_minter::SigningKey;
using vgi::hosted::Config;
using vgi::hosted::JwksResponse;
using vgi::hosted::PassVerifier;
using vgi_rpc::AuthReason;
using vgi_rpc::AuthRejectedError;

const std::string kIssuer = "https://auth.orchard.test";
const std::string kAudience = "https://desk.orchard.test/d/dep_123";
const int64_t kNow = 1'900'000'000;

nlohmann::json base_claims() {
    return {{"iss", kIssuer},
            {"aud", kAudience},
            {"sub", "user_01M7KNJYC18T"},
            {"kind", "user"},
            {"email", "ada@example.com"},
            {"email_verified", true},
            {"name", "Ada Lovelace"},
            {"org_id", "org_01M7"},
            {"org_name", "Acme Analytics"},
            {"federated_iss", "https://accounts.google.com"},
            {"federated_sub", "1081723"},
            {"client_id", "vgi-duckdb"},
            {"iat", kNow},
            {"nbf", kNow},
            {"exp", kNow + 900},
            {"jti", "jti-1"}};
}

std::string mint(const SigningKey& key, const nlohmann::json& claims = base_claims(),
                 nlohmann::json header = nullptr) {
    return pass_minter::mint_pass(key, claims, std::move(header));
}

// A JWKS endpoint the test controls, and clocks it moves by hand.
struct Fixture {
    nlohmann::json keys = nlohmann::json::array();
    std::string cache_control;
    bool down = false;
    int fetches = 0;
    int64_t wall = kNow;
    std::chrono::steady_clock::time_point mono{std::chrono::hours(1)};

    std::unique_ptr<PassVerifier> verifier(std::vector<std::string> algorithms = {"ES256"}) {
        Config config;
        config.issuer = kIssuer;
        config.audience = kAudience;
        config.jwks_uri = "https://auth.orchard.test/.well-known/jwks.json";
        config.algorithms = std::move(algorithms);
        PassVerifier::Options options;
        options.fetch = [this](const std::string& uri, std::chrono::seconds) -> JwksResponse {
            ++fetches;
            CHECK(uri == "https://auth.orchard.test/.well-known/jwks.json");
            if (down) throw std::runtime_error("connection refused");
            return {nlohmann::json{{"keys", keys}}.dump(), cache_control};
        };
        options.now = [this] { return wall; };
        options.monotonic = [this] { return mono; };
        return std::make_unique<PassVerifier>(std::move(config), std::move(options));
    }
};

AuthReason refusal(const PassVerifier& verifier, const std::string& token) {
    try {
        verifier.verify(token);
    } catch (const AuthRejectedError& e) {
        return e.reason();
    }
    FAIL("the Pass was accepted");
    return AuthReason::NONE;
}

std::string tamper(const std::string& token, const std::string& field, const nlohmann::json& v) {
    const auto first = token.find('.');
    const auto second = token.find('.', first + 1);
    std::string payload = token.substr(first + 1, second - first - 1);
    auto claims = nlohmann::json::parse(jwt::base::decode<jwt::alphabet::base64url>(
        jwt::base::pad<jwt::alphabet::base64url>(payload)));
    claims[field] = v;
    return token.substr(0, first) + "." + b64u(claims.dump()) + token.substr(second);
}

class ScopedEnv {
public:
    explicit ScopedEnv(std::vector<std::pair<std::string, std::optional<std::string>>> values) {
        for (auto& [name, value] : values) {
            const char* old = std::getenv(name.c_str());
            saved_.emplace_back(name, old ? std::optional<std::string>(old) : std::nullopt);
            if (value) {
                ::setenv(name.c_str(), value->c_str(), 1);
            } else {
                ::unsetenv(name.c_str());
            }
        }
    }
    ~ScopedEnv() {
        for (auto& [name, value] : saved_) {
            if (value) {
                ::setenv(name.c_str(), value->c_str(), 1);
            } else {
                ::unsetenv(name.c_str());
            }
        }
    }

private:
    std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

}  // namespace

TEST_CASE("a valid Pass carries its claims with their JSON types", "[hosted]") {
    Fixture f;
    const auto key = SigningKey::ec("desk-es256-1");
    f.keys.push_back(key.jwk);
    const auto verifier = f.verifier();
    const auto auth = verifier->verify(mint(key));
    CHECK(auth.authenticated);
    CHECK(auth.domain == "orchard");
    CHECK(auth.principal == std::optional<std::string>("user_01M7KNJYC18T"));
    CHECK(auth.claims["email_verified"].is_boolean());
    CHECK(auth.claims["email_verified"] == true);
    CHECK(auth.claims["exp"].is_number_integer());
    for (const char* name : {"sub", "kind", "email", "name", "org_id", "org_name", "federated_iss",
                             "federated_sub", "client_id", "iat", "nbf", "exp", "jti"}) {
        CHECK(auth.claims.contains(name));
    }
    // One fetch serves many Passes.
    verifier->verify(mint(key));
    CHECK(f.fetches == 1);
}

TEST_CASE("every malformed or mis-issued Pass is refused", "[hosted]") {
    Fixture f;
    const auto key = SigningKey::ec("desk-es256-1");
    const auto rsa = SigningKey::rsa("desk-rs256-1");
    const auto p384 = SigningKey::ec("desk-es384-1", "P-384", "ES384", 48);
    f.keys = {key.jwk, rsa.jwk, p384.jwk};
    const auto verifier = f.verifier();
    const auto invalid = AuthReason::INVALID_CREDENTIAL;
    auto claims = [](const char* name, nlohmann::json value) {
        auto c = base_claims();
        if (value.is_null()) {
            c.erase(name);
        } else {
            c[name] = std::move(value);
        }
        return c;
    };

    CHECK(refusal(*verifier, "not-a-jwt") == invalid);
    CHECK(refusal(*verifier, "a.b.c.d") == invalid);
    CHECK(refusal(*verifier, mint(key) + "=") == invalid);
    CHECK(refusal(*verifier, tamper(mint(key), "org_id", "org_EVIL")) == invalid);
    CHECK(refusal(*verifier, mint(key, claims("aud", "https://desk.orchard.test/d/other"))) ==
          invalid);
    CHECK(refusal(*verifier, mint(key, claims("aud", {kAudience, "https://x.test"}))) == invalid);
    CHECK(refusal(*verifier, mint(key, claims("aud", nlohmann::json::array()))) == invalid);
    CHECK(refusal(*verifier, mint(key, claims("iss", "https://evil.test"))) == invalid);
    CHECK(refusal(*verifier, mint(key, claims("exp", nullptr))) == invalid);
    CHECK(refusal(*verifier, mint(key, claims("exp", "1900000900"))) == invalid);
    CHECK(refusal(*verifier, mint(key, claims("sub", nullptr))) == invalid);
    CHECK(refusal(*verifier, mint(key, claims("nbf", kNow + 61))) == invalid);
    CHECK(refusal(*verifier, mint(key, claims("iat", "yesterday"))) == invalid);
    CHECK(refusal(*verifier,
                  mint(key, base_claims(), {{"alg", "ES256"}, {"typ", "JWT"}, {"kid", key.kid}})) ==
          invalid);
    CHECK(refusal(*verifier, mint(key, base_claims(), {{"alg", "ES256"}, {"kid", key.kid}})) ==
          invalid);
    CHECK(
        refusal(*verifier,
                mint(key, base_claims(),
                     {{"alg", "ES256"}, {"typ", "at+jwt"}, {"kid", key.kid}, {"crit", {"exp"}}})) ==
        invalid);
    // Algorithms not allowed, whatever key signed them.
    CHECK(refusal(*verifier, mint(rsa)) == invalid);
    CHECK(refusal(*verifier, mint(p384)) == invalid);
    {
        const std::string input = b64u(R"({"alg":"HS256","typ":"at+jwt","kid":"desk-es256-1"})") +
                                  "." + b64u(base_claims().dump());
        std::error_code ec;
        const auto mac = jwt::algorithm::hs256(std::string(32, 'k')).sign(input, ec);
        CHECK(refusal(*verifier, input + "." + b64u(mac)) == invalid);
        CHECK(refusal(*verifier, b64u(R"({"alg":"none","typ":"at+jwt"})") + "." +
                                     b64u(base_claims().dump()) + ".") == invalid);
    }
    // A Pass that names an ES256 key but is signed by another one.
    const auto stranger = SigningKey::ec("desk-es256-1");
    CHECK(refusal(*verifier, mint(stranger)) == invalid);
}

TEST_CASE("exp and nbf get 60 seconds of leeway; a lapsed Pass is expired", "[hosted]") {
    Fixture f;
    const auto key = SigningKey::ec("k1");
    f.keys.push_back(key.jwk);
    const auto verifier = f.verifier();
    auto with = [](int64_t exp, int64_t nbf) {
        auto c = base_claims();
        c["exp"] = exp;
        c["nbf"] = nbf;
        return c;
    };
    CHECK_NOTHROW(verifier->verify(mint(key, with(kNow - 59, kNow))));
    CHECK_NOTHROW(verifier->verify(mint(key, with(kNow - 60, kNow))));
    CHECK(refusal(*verifier, mint(key, with(kNow - 61, kNow))) == AuthReason::EXPIRED_CREDENTIAL);
    CHECK_NOTHROW(verifier->verify(mint(key, with(kNow + 900, kNow + 60))));
    CHECK(refusal(*verifier, mint(key, with(kNow + 900, kNow + 61))) ==
          AuthReason::INVALID_CREDENTIAL);
}

TEST_CASE("typ compares case-insensitively with application/ optional", "[hosted]") {
    Fixture f;
    const auto key = SigningKey::ec("k1");
    f.keys.push_back(key.jwk);
    const auto verifier = f.verifier();
    for (const char* typ : {"at+jwt", "AT+JWT", "application/at+jwt", "Application/AT+jwt"}) {
        INFO(typ);
        CHECK_NOTHROW(verifier->verify(
            mint(key, base_claims(), {{"alg", "ES256"}, {"typ", typ}, {"kid", "k1"}})));
    }
    auto single = base_claims();
    single["aud"] = nlohmann::json::array({kAudience});
    CHECK_NOTHROW(verifier->verify(mint(key, single)));
}

TEST_CASE("VGI_JWT_ALGORITHMS can admit RS256 but never HS* or none", "[hosted]") {
    Fixture f;
    const auto rsa = SigningKey::rsa("r1");
    f.keys.push_back(rsa.jwk);
    CHECK_NOTHROW(f.verifier({"RS256"})->verify(mint(rsa)));
    CHECK_THROWS_AS(f.verifier({"ES256", "HS256"}), vgi::hosted::ConfigError);
    CHECK_THROWS_AS(f.verifier({"none"}), vgi::hosted::ConfigError);
    CHECK_THROWS_AS(f.verifier({"EdDSA"}), vgi::hosted::ConfigError);
    CHECK_THROWS_AS(f.verifier({}), vgi::hosted::ConfigError);
}

TEST_CASE("the JWKS is cached for at most its max-age, capped at ten minutes", "[hosted][jwks]") {
    Fixture f;
    const auto key = SigningKey::ec("k1");
    f.keys.push_back(key.jwk);
    const auto verifier = f.verifier();

    f.cache_control = "public, max-age=30";
    verifier->verify(mint(key));
    f.mono += std::chrono::seconds(29);
    verifier->verify(mint(key));
    CHECK(f.fetches == 1);
    f.mono += std::chrono::seconds(1);
    verifier->verify(mint(key));
    CHECK(f.fetches == 2);

    // A longer max-age is capped at 600 s.
    f.cache_control = "max-age=86400";
    f.mono += std::chrono::seconds(31);
    verifier->verify(mint(key));
    CHECK(f.fetches == 3);
    f.mono += std::chrono::seconds(599);
    verifier->verify(mint(key));
    CHECK(f.fetches == 3);
    f.mono += std::chrono::seconds(1);
    verifier->verify(mint(key));
    CHECK(f.fetches == 4);
}

TEST_CASE("expired keys are discarded; a failed re-fetch is 503 until it recovers",
          "[hosted][jwks]") {
    Fixture f;
    const auto key = SigningKey::ec("k1");
    f.keys.push_back(key.jwk);
    const auto verifier = f.verifier();

    // Down from the start: 503, never a 401.
    f.down = true;
    CHECK_THROWS_AS(verifier->verify(mint(key)), vgi_rpc::AuthUnavailableError);
    f.down = false;
    f.cache_control = "max-age=60";
    verifier->verify(mint(key));

    f.down = true;
    f.mono += std::chrono::seconds(60);
    for (int i = 0; i < 3; ++i) {
        CHECK_THROWS_AS(verifier->verify(mint(key)), vgi_rpc::AuthUnavailableError);
        f.mono += std::chrono::seconds(1);
    }
    f.down = false;
    CHECK_NOTHROW(verifier->verify(mint(key)));
}

TEST_CASE("an unknown kid re-fetches at most once a minute", "[hosted][jwks]") {
    Fixture f;
    const auto old_key = SigningKey::ec("old");
    const auto new_key = SigningKey::ec("new");
    f.keys.push_back(old_key.jwk);
    const auto verifier = f.verifier();
    verifier->verify(mint(old_key));
    CHECK(f.fetches == 1);

    // Rotation inside the first minute: refused without a fetch.
    f.keys.push_back(new_key.jwk);
    f.mono += std::chrono::seconds(30);
    CHECK(refusal(*verifier, mint(new_key)) == AuthReason::INVALID_CREDENTIAL);
    CHECK(f.fetches == 1);

    // After the interval one fetch picks the new key up.
    f.mono += std::chrono::seconds(30);
    CHECK_NOTHROW(verifier->verify(mint(new_key)));
    CHECK(f.fetches == 2);

    // A kid nobody publishes: one fetch per minute, however many tries.
    const auto ghost = SigningKey::ec("ghost");
    f.mono += std::chrono::seconds(60);
    for (int i = 0; i < 5; ++i)
        CHECK(refusal(*verifier, mint(ghost)) == AuthReason::INVALID_CREDENTIAL);
    CHECK(f.fetches == 3);
}

TEST_CASE("JWKS keys that cannot verify a Pass are skipped", "[hosted][jwks]") {
    Fixture f;
    const auto key = SigningKey::ec("k1");
    auto enc = key.jwk;
    enc["use"] = "enc";
    auto oct = nlohmann::json{{"kty", "oct"}, {"k", b64u("secret")}, {"kid", "k1"}};
    auto wrong_alg = key.jwk;
    wrong_alg["alg"] = "ES384";
    f.keys = {enc, oct, wrong_alg};
    const auto verifier = f.verifier();
    CHECK(refusal(*verifier, mint(key)) == AuthReason::INVALID_CREDENTIAL);
    f.keys.push_back(key.jwk);
    f.mono += std::chrono::seconds(61);
    CHECK_NOTHROW(verifier->verify(mint(key)));
}

TEST_CASE("the environment switch and settings fail closed", "[hosted][env]") {
    for (const char* on : {"1", "true", "YES", " on "}) {
        ScopedEnv env({{"VGI_ORCHARD_HOSTED", on}});
        CHECK(vgi::hosted::hosted_mode_enabled());
    }
    for (const char* off : {"0", "false", "", "off", "No"}) {
        ScopedEnv env({{"VGI_ORCHARD_HOSTED", off}});
        CHECK_FALSE(vgi::hosted::hosted_mode_enabled());
    }
    {
        ScopedEnv env({{"VGI_ORCHARD_HOSTED", std::nullopt}});
        CHECK_FALSE(vgi::hosted::hosted_mode_enabled());
    }
    {
        ScopedEnv env({{"VGI_ORCHARD_HOSTED", "yes please"}});
        CHECK_THROWS_AS(vgi::hosted::hosted_mode_enabled(), vgi::hosted::ConfigError);
    }

    const std::vector<std::pair<std::string, std::optional<std::string>>> good = {
        {"VGI_JWT_ISSUER", kIssuer},
        {"VGI_JWT_AUDIENCE", kAudience},
        {"VGI_JWT_JWKS_URI", "https://auth.orchard.test/jwks"},
        {"VGI_JWT_ALGORITHMS", std::nullopt}};
    {
        ScopedEnv env(good);
        const auto config = vgi::hosted::config_from_env();
        CHECK(config.algorithms == std::vector<std::string>{"ES256"});
    }
    for (const char* name : {"VGI_JWT_ISSUER", "VGI_JWT_AUDIENCE", "VGI_JWT_JWKS_URI"}) {
        for (const std::optional<std::string>& bad :
             {std::optional<std::string>(), std::optional<std::string>("  "),
              std::optional<std::string>("a,b")}) {
            ScopedEnv base(good);
            ScopedEnv env({{name, bad}});
            try {
                vgi::hosted::config_from_env();
                FAIL("accepted " << name);
            } catch (const vgi::hosted::ConfigError& e) {
                CHECK(std::string(e.what()).find(name) != std::string::npos);
            }
        }
    }
    for (const char* algorithms : {"ES256,HS256", "none", "hs512", "EdDSA"}) {
        ScopedEnv base(good);
        ScopedEnv env({{"VGI_JWT_ALGORITHMS", algorithms}});
        CHECK_THROWS_AS(vgi::hosted::config_from_env(), vgi::hosted::ConfigError);
    }
    {
        ScopedEnv base(good);
        ScopedEnv env({{"VGI_JWT_ALGORITHMS", " ES256 , RS256 "}});
        CHECK(vgi::hosted::config_from_env().algorithms ==
              std::vector<std::string>{"ES256", "RS256"});
    }
}
