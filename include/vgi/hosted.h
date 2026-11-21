// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Orchard hosted mode (`VGI_ORCHARD_HOSTED=1`).
//
// A hosted worker (a *Field Worker*) serves VGI over HTTP to callers who hold a
// *Pass*: a short-lived RFC 9068 access token (`typ: at+jwt`, ES256) issued by
// the Orchard Front Desk for exactly one canonical Desk Address. The worker
// verifies Passes locally against the Front Desk's JWKS and accepts nothing
// else. Everything here fails closed:
//
// - `Worker::run` refuses to start unless `VGI_JWT_ISSUER`, `VGI_JWT_AUDIENCE`
//   and `VGI_JWT_JWKS_URI` each hold one non-empty value, `VGI_SIGNING_KEY` is
//   at least 32 bytes, the transport is `--http`, and no other way in is
//   configured (static bearer tokens, sealed-grant keys, `resolve_token` /
//   `mint_grant`, OAuth resource metadata, a proxy proof, Iroh or other peer
//   identity, or a `configure_http` hook that changes the authentication).
// - Every request except `HEAD /health` and CORS preflights must present a
//   valid Pass, or gets a 401 with `WWW-Authenticate: Bearer
//   error="invalid_token"`; the check runs before the body is read.
// - The JWKS is cached for at most 10 minutes; once that passes and the
//   re-fetch fails, verification fails (HTTP 503) rather than using stale keys.
//
// vgi-python's `vgi/hosted.py` is the reference; the design is in the
// extension repo's `docs/location-delegation.md`.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <vgi_rpc/identity.h>

namespace vgi::hosted {

// The environment variable that turns hosted mode on. Baked into the hosted
// base image, not set per deployment.
inline constexpr const char* kHostedEnv = "VGI_ORCHARD_HOSTED";
// The challenge on every hosted-mode 401. No `resource_metadata`: a Field
// Worker never sends a client off to discover an authorization server.
inline constexpr const char* kChallenge = "Bearer error=\"invalid_token\"";
// `AuthContext::domain` of a Pass-authenticated caller.
inline constexpr const char* kDomain = "orchard";
// Required JOSE `typ` (RFC 9068), compared case-insensitively with the
// `application/` prefix optional.
inline constexpr const char* kRequiredTyp = "at+jwt";
// Clock skew allowed on `exp` / `nbf`.
inline constexpr int64_t kLeewaySeconds = 60;
// Hard upper bound on how long JWKS keys are trusted.
inline constexpr std::chrono::seconds kJwksMaxAge{600};
// Minimum time between JWKS re-fetches triggered by an unknown `kid`.
inline constexpr std::chrono::seconds kJwksRefreshMinInterval{60};
// Timeout for each JWKS request.
inline constexpr std::chrono::seconds kJwksFetchTimeout{10};
// The shortest `VGI_SIGNING_KEY` a hosted worker accepts.
inline constexpr std::size_t kMinSigningKeyBytes = 32;
// The request-body cap in hosted mode when `VGI_RPC_MAX_REQUEST_BYTES` is
// unset. Advertised as `VGI-Max-Request-Bytes`; a larger body is 413.
inline constexpr int64_t kDefaultMaxRequestBytes = int64_t{32} * 1024 * 1024;
// The longest Pass looked at; anything longer is refused unparsed.
inline constexpr std::size_t kMaxPassBytes = 8192;

// A hosted-mode configuration error: the worker must not start.
class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Whether `VGI_ORCHARD_HOSTED` turns hosted mode on: true for
// 1/true/yes/on, false when unset or 0/false/no/off (any case, surrounding
// whitespace ignored). Any other value throws `ConfigError`, so a typo cannot
// quietly start a worker without authentication.
bool hosted_mode_enabled();

// The verified hosted-mode settings.
struct Config {
    std::string issuer;    // the Front Desk issuer (`iss`), matched exactly
    std::string audience;  // the one canonical Desk Address (`aud`), matched exactly
    std::string jwks_uri;  // where the Front Desk publishes its signing keys
    // Accepted JWS algorithms: `ES256` unless `VGI_JWT_ALGORITHMS` overrides.
    // `none` and `HS*` are always refused.
    std::vector<std::string> algorithms{"ES256"};
};

// Read and validate `VGI_JWT_ISSUER`, `VGI_JWT_AUDIENCE`, `VGI_JWT_JWKS_URI`
// and `VGI_JWT_ALGORITHMS`. Throws `ConfigError` naming the variable at fault.
Config config_from_env();

// One JWKS response. A fetcher throws (anything) when the keys could not be
// fetched; the verifier turns that into a 503.
struct JwksResponse {
    std::string body;
    std::string cache_control;  // the `Cache-Control` header, or empty
};
using JwksFetcher =
    std::function<JwksResponse(const std::string& uri, std::chrono::seconds timeout)>;

// The default fetcher: one GET over libcurl, no redirects, http(s) only, a
// 2xx answer of at most 1 MiB.
JwksResponse fetch_jwks(const std::string& uri, std::chrono::seconds timeout);

// Verifies Passes against the Front Desk's JWKS. Thread-safe.
class PassVerifier {
public:
    struct Options {
        JwksFetcher fetch = fetch_jwks;
        // Wall-clock seconds for `exp` / `nbf`; monotonic time for the JWKS
        // cache. Injectable for tests; empty means the system clocks.
        std::function<int64_t()> now;
        std::function<std::chrono::steady_clock::time_point()> monotonic;
    };

    // Throws `ConfigError` for an empty issuer, audience or JWKS URI, or an
    // algorithm list that is empty, names `none` / `HS*`, or names one this
    // build cannot verify.
    explicit PassVerifier(Config config);
    PassVerifier(Config config, Options options);
    ~PassVerifier();
    PassVerifier(const PassVerifier&) = delete;
    PassVerifier& operator=(const PassVerifier&) = delete;

    // The caller's context for a valid Pass: `domain = "orchard"`, `principal`
    // = `sub`, and every claim exactly as decoded (booleans stay booleans,
    // numbers stay numbers). Throws `vgi_rpc::AuthRejectedError`
    // (`EXPIRED_CREDENTIAL` for a lapsed Pass, `INVALID_CREDENTIAL` for every
    // other refusal) or `vgi_rpc::AuthUnavailableError` when no usable JWKS can
    // be fetched. Never logs the token.
    vgi_rpc::AuthContext verify(const std::string& token) const;

    const Config& config() const noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

// The `HttpConfig::bearer_authenticate` hosted mode installs. A named type, so
// `Worker::run` can tell its own verifier from a replacement a
// `configure_http` hook put in its place.
struct PassAuthenticator {
    std::shared_ptr<const PassVerifier> verifier;
    std::optional<vgi_rpc::AuthContext> operator()(const std::string& token) const {
        return verifier->verify(token);
    }
};

}  // namespace vgi::hosted
