// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Orchard hosted mode: the environment switch and the Pass verifier.
//
// Signatures and JWK-to-key conversion come from jwt-cpp (over the OpenSSL the
// RPC framework already links). Everything around them -- the compact
// serialization, the header, the claims, the JWKS cache -- is done here,
// strictly, because jwt-cpp's own verifier makes choices (an `iat` in the
// future is refused, an absent `exp` is accepted, any listed audience
// matches) that are not the Pass contract.

#include "vgi/hosted.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <system_error>
#include <utility>

#include <curl/curl.h>
#include <jwt-cpp/jwt.h>
#include <nlohmann/json.hpp>

#include <vgi_rpc/errors.h>
#include <vgi_rpc/http_config.h>

namespace vgi::hosted {

namespace {

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

// The algorithms this build verifies, and what each needs of a JWK.
struct AlgorithmSpec {
    const char* kty;
    const char* crv;              // EC only
    std::size_t signature_bytes;  // EC only: the raw R || S length
};

const std::map<std::string, AlgorithmSpec>& supported_algorithms() {
    static const std::map<std::string, AlgorithmSpec> table = {
        {"ES256", {"EC", "P-256", 64}},  {"ES384", {"EC", "P-384", 96}},
        {"ES512", {"EC", "P-521", 132}}, {"RS256", {"RSA", "", 0}},
        {"RS384", {"RSA", "", 0}},       {"RS512", {"RSA", "", 0}},
        {"PS256", {"RSA", "", 0}},       {"PS384", {"RSA", "", 0}},
        {"PS512", {"RSA", "", 0}},
    };
    return table;
}

std::string supported_list() {
    std::string out;
    for (const auto& [name, _] : supported_algorithms()) out += (out.empty() ? "" : ", ") + name;
    return out;
}

bool forbidden_algorithm(const std::string& name) {
    const std::string l = lower(trim(name));
    return l == "none" || l.rfind("hs", 0) == 0;
}

void validate_algorithms(const std::vector<std::string>& algorithms, const std::string& raw) {
    if (algorithms.empty() ||
        std::any_of(algorithms.begin(), algorithms.end(), forbidden_algorithm)) {
        throw ConfigError("VGI_JWT_ALGORITHMS='" + raw +
                          "' is not allowed; Passes are asymmetric (default ES256), never 'none' "
                          "or HS*.");
    }
    for (const auto& name : algorithms) {
        if (supported_algorithms().count(name) == 0) {
            throw ConfigError("VGI_JWT_ALGORITHMS names '" + name +
                              "', which this SDK cannot verify (supported: " + supported_list() +
                              ").");
        }
    }
}

std::string single_value(const char* name) {
    const std::string value = trim(env_or_empty(name));
    if (value.empty()) {
        throw ConfigError(std::string(name) + " must be set (unset and empty are both errors).");
    }
    if (value.find(',') != std::string::npos) {
        throw ConfigError(std::string(name) +
                          " must be exactly one value; got a comma-separated list. A Pass is "
                          "issued for one canonical Desk Address and one issuer.");
    }
    return value;
}

// Strict base64url (RFC 7515 §2): the URL-safe alphabet, no padding, no
// whitespace. Anything else is not a compact JWS.
std::optional<std::string> base64url_decode(const std::string& in) {
    if (in.size() % 4 == 1) return std::nullopt;
    std::string out;
    out.reserve(in.size() * 3 / 4);
    uint32_t buffer = 0;
    int bits = 0;
    for (const char c : in) {
        int v;
        if (c >= 'A' && c <= 'Z') {
            v = c - 'A';
        } else if (c >= 'a' && c <= 'z') {
            v = c - 'a' + 26;
        } else if (c >= '0' && c <= '9') {
            v = c - '0' + 52;
        } else if (c == '-') {
            v = 62;
        } else if (c == '_') {
            v = 63;
        } else {
            return std::nullopt;
        }
        buffer = (buffer << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((buffer >> bits) & 0xff));
        }
    }
    return out;
}

using SignatureCheck =
    std::function<void(const std::string& data, const std::string& signature, std::error_code&)>;

// One usable JWKS key: its id, and a signature check per algorithm it may
// verify (only algorithms the deployment allows).
struct Key {
    std::optional<std::string> kid;
    std::map<std::string, SignatureCheck> checks;
};

struct KeySet {
    std::vector<Key> keys;
};

std::optional<std::string> string_member(const nlohmann::json& object, const char* name) {
    const auto it = object.find(name);
    if (it == object.end() || !it->is_string()) return std::nullopt;
    return it->get<std::string>();
}

template <typename Algorithm>
SignatureCheck check_for(const std::string& public_pem) {
    // Constructing validates the key (on-curve point, curve size for the
    // algorithm); a key that fails is skipped by the caller.
    auto algorithm = std::make_shared<Algorithm>(public_pem);
    return [algorithm](const std::string& data, const std::string& signature, std::error_code& ec) {
        algorithm->verify(data, signature, ec);
    };
}

std::optional<SignatureCheck> rsa_check(const std::string& alg, const std::string& pem) {
    if (alg == "RS256") return check_for<jwt::algorithm::rs256>(pem);
    if (alg == "RS384") return check_for<jwt::algorithm::rs384>(pem);
    if (alg == "RS512") return check_for<jwt::algorithm::rs512>(pem);
    if (alg == "PS256") return check_for<jwt::algorithm::ps256>(pem);
    if (alg == "PS384") return check_for<jwt::algorithm::ps384>(pem);
    if (alg == "PS512") return check_for<jwt::algorithm::ps512>(pem);
    return std::nullopt;
}

std::optional<SignatureCheck> ec_check(const std::string& alg, const std::string& pem) {
    if (alg == "ES256") return check_for<jwt::algorithm::es256>(pem);
    if (alg == "ES384") return check_for<jwt::algorithm::es384>(pem);
    if (alg == "ES512") return check_for<jwt::algorithm::es512>(pem);
    return std::nullopt;
}

// One JWK, or nothing when it is not a verification key this deployment can
// use (another `use`, a symmetric or unknown `kty`, an algorithm not allowed,
// a malformed or weak key). Skipping rather than failing the whole set: a
// Front Desk may publish keys for other consumers beside the Pass keys.
std::optional<Key> import_key(const nlohmann::json& jwk, const std::vector<std::string>& allowed) {
    if (!jwk.is_object()) return std::nullopt;
    if (jwk.contains("use") && string_member(jwk, "use") != std::optional<std::string>("sig")) {
        return std::nullopt;
    }
    if (jwk.contains("key_ops")) {
        const auto& ops = jwk["key_ops"];
        if (!ops.is_array() || std::find(ops.begin(), ops.end(), "verify") == ops.end()) {
            return std::nullopt;
        }
    }
    Key key;
    if (jwk.contains("kid")) {
        key.kid = string_member(jwk, "kid");
        if (!key.kid) return std::nullopt;
    }
    const auto kty = string_member(jwk, "kty");
    if (!kty) return std::nullopt;
    const auto declared_alg = string_member(jwk, "alg");
    if (jwk.contains("alg") && !declared_alg) return std::nullopt;

    std::string pem;
    try {
        std::error_code ec;
        if (*kty == "EC") {
            const auto crv = string_member(jwk, "crv");
            const auto x = string_member(jwk, "x");
            const auto y = string_member(jwk, "y");
            if (!crv || !x || !y) return std::nullopt;
            // jwt-cpp takes the JWK curve name and maps it to OpenSSL's.
            if (*crv != "P-256" && *crv != "P-384" && *crv != "P-521") return std::nullopt;
            pem = jwt::helper::create_public_key_from_ec_components(*crv, *x, *y, ec);
        } else if (*kty == "RSA") {
            const auto n = string_member(jwk, "n");
            const auto e = string_member(jwk, "e");
            if (!n || !e) return std::nullopt;
            // At least 2048 bits (RFC 7518 §3.3).
            const auto modulus = base64url_decode(*n);
            if (!modulus) return std::nullopt;
            const auto significant = modulus->find_first_not_of('\0');
            if (significant == std::string::npos || modulus->size() - significant < 256) {
                return std::nullopt;
            }
            pem = jwt::helper::create_public_key_from_rsa_components(*n, *e, ec);
        } else {
            return std::nullopt;
        }
        if (ec || pem.empty()) return std::nullopt;

        for (const auto& alg : allowed) {
            const auto& spec = supported_algorithms().at(alg);
            if (spec.kty != *kty) continue;
            if (declared_alg && *declared_alg != alg) continue;
            if (*kty == "EC" && string_member(jwk, "crv") != std::optional<std::string>(spec.crv)) {
                continue;
            }
            auto check = *kty == "EC" ? ec_check(alg, pem) : rsa_check(alg, pem);
            if (check) key.checks.emplace(alg, std::move(*check));
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
    if (key.checks.empty()) return std::nullopt;
    return key;
}

std::shared_ptr<const KeySet> parse_jwks(const std::string& body,
                                         const std::vector<std::string>& allowed) {
    const auto document = nlohmann::json::parse(body);  // throws: a failed fetch
    if (!document.is_object() || !document.contains("keys") || !document["keys"].is_array()) {
        throw std::runtime_error("JWKS document has no 'keys' array");
    }
    auto set = std::make_shared<KeySet>();
    for (const auto& jwk : document["keys"]) {
        if (auto key = import_key(jwk, allowed)) set->keys.push_back(std::move(*key));
    }
    return set;
}

// `Cache-Control: max-age`, the way vgi-rpc-python reads it: the first
// `max-age=N` directive (optionally quoted), case-insensitively.
std::optional<std::chrono::seconds> cache_control_max_age(const std::string& value) {
    static const std::regex pattern(R"re((?:^|,)\s*max-age\s*=\s*"?(\d+)"?\s*(?=,|$))re",
                                    std::regex::icase | std::regex::ECMAScript);
    std::smatch match;
    if (!std::regex_search(value, match, pattern)) return std::nullopt;
    const std::string digits = match[1].str();
    if (digits.size() > 9) return std::chrono::seconds(999999999);  // capped below anyway
    return std::chrono::seconds(std::stoll(digits));
}

std::string normalize_typ(const std::string& value) {
    std::string out = lower(trim(value));
    constexpr std::string_view prefix = "application/";
    if (out.rfind(prefix, 0) == 0) out.erase(0, prefix.size());
    return out;
}

[[noreturn]] void reject(vgi_rpc::AuthReason reason, const char* why) {
    // Only the category: never the token, a claim value or the signature.
    std::cerr << "vgi hosted: Pass rejected (" << why << ")\n";
    throw vgi_rpc::AuthRejectedError(reason, std::string("Pass rejected: ") + why);
}

[[noreturn]] void reject(const char* why) {
    reject(vgi_rpc::AuthReason::INVALID_CREDENTIAL, why);
}

bool is_numeric_date(const nlohmann::json& value) {
    return value.is_number() && !value.is_boolean();
}

double as_double(const nlohmann::json& value) {
    return value.get<double>();
}

void ensure_curl() {
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

constexpr std::size_t kMaxJwksBytes = 1 << 20;

struct FetchBuffer {
    std::string body;
    std::string cache_control;
    bool too_large = false;
};

size_t write_body(char* data, size_t size, size_t count, void* user) {
    auto* buffer = static_cast<FetchBuffer*>(user);
    const size_t bytes = size * count;
    if (buffer->body.size() + bytes > kMaxJwksBytes) {
        buffer->too_large = true;
        return 0;  // aborts the transfer
    }
    buffer->body.append(data, bytes);
    return bytes;
}

size_t write_header(char* data, size_t size, size_t count, void* user) {
    auto* buffer = static_cast<FetchBuffer*>(user);
    const size_t bytes = size * count;
    const std::string line(data, bytes);
    if (line.rfind("HTTP/", 0) == 0) {
        buffer->cache_control.clear();  // a new response (after a 1xx)
        return bytes;
    }
    const auto colon = line.find(':');
    if (colon != std::string::npos && lower(trim(line.substr(0, colon))) == "cache-control") {
        const std::string value = trim(line.substr(colon + 1));
        if (!buffer->cache_control.empty()) buffer->cache_control += ", ";
        buffer->cache_control += value;
    }
    return bytes;
}

}  // namespace

bool hosted_mode_enabled() {
    const char* raw = std::getenv(kHostedEnv);
    if (raw == nullptr) return false;
    const std::string value = lower(trim(raw));
    static const std::set<std::string> on = {"1", "true", "yes", "on"};
    static const std::set<std::string> off = {"", "0", "false", "no", "off"};
    if (on.count(value)) return true;
    if (off.count(value)) return false;
    throw ConfigError(std::string(kHostedEnv) + " must be 1 or 0 (or true/false), got '" + raw +
                      "'");
}

Config config_from_env() {
    Config config;
    config.issuer = single_value("VGI_JWT_ISSUER");
    config.audience = single_value("VGI_JWT_AUDIENCE");
    config.jwks_uri = single_value("VGI_JWT_JWKS_URI");
    const std::string raw = env_or_empty("VGI_JWT_ALGORITHMS");
    if (!trim(raw).empty()) {
        config.algorithms.clear();
        size_t begin = 0;
        while (begin <= raw.size()) {
            const auto end = raw.find(',', begin);
            const std::string item =
                trim(raw.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
            if (!item.empty()) config.algorithms.push_back(item);
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        validate_algorithms(config.algorithms, raw);
    }
    return config;
}

JwksResponse fetch_jwks(const std::string& uri, std::chrono::seconds timeout) {
    ensure_curl();
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
    if (!curl) throw std::runtime_error("cannot initialise libcurl");
    FetchBuffer buffer;
    curl_slist* headers = curl_slist_append(nullptr, "Accept: application/json");
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> header_guard(headers,
                                                                             curl_slist_free_all);
    CURL* h = curl.get();
    curl_easy_setopt(h, CURLOPT_URL, uri.c_str());
    curl_easy_setopt(h, CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 0L);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout.count() * 1000));
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_body);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, write_header);
    curl_easy_setopt(h, CURLOPT_HEADERDATA, &buffer);
    // The OpenSSL conventions, for images whose CA bundle is not where libcurl
    // was built to look.
    if (const char* file = std::getenv("SSL_CERT_FILE"); file && *file) {
        curl_easy_setopt(h, CURLOPT_CAINFO, file);
    }
    if (const char* dir = std::getenv("SSL_CERT_DIR"); dir && *dir) {
        curl_easy_setopt(h, CURLOPT_CAPATH, dir);
    }
    const CURLcode code = curl_easy_perform(h);
    if (buffer.too_large) throw std::runtime_error("JWKS response exceeds 1 MiB");
    if (code != CURLE_OK) throw std::runtime_error(curl_easy_strerror(code));
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    if (status < 200 || status >= 300) {
        throw std::runtime_error("JWKS endpoint answered HTTP " + std::to_string(status));
    }
    return {std::move(buffer.body), std::move(buffer.cache_control)};
}

// ---------------------------------------------------------------------------
// PassVerifier
// ---------------------------------------------------------------------------

struct PassVerifier::State {
    Config config;
    Options options;

    std::mutex mutex;
    // The keys and the monotonic time they stop being usable. Null: none.
    std::shared_ptr<const KeySet> keys;
    std::chrono::steady_clock::time_point expires_at{};
    std::optional<std::chrono::steady_clock::time_point> last_fetch_attempt;

    std::chrono::steady_clock::time_point monotonic() const {
        return options.monotonic ? options.monotonic() : std::chrono::steady_clock::now();
    }
    int64_t now() const {
        if (options.now) return options.now();
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }
    bool usable_locked(std::chrono::steady_clock::time_point at) const {
        return keys != nullptr && at < expires_at;
    }

    // Fetch the JWKS and replace the cache. Caller holds `mutex`.
    std::shared_ptr<const KeySet> fetch_locked() {
        const auto started = monotonic();
        last_fetch_attempt = started;
        std::shared_ptr<const KeySet> fetched;
        std::optional<std::chrono::seconds> advertised;
        try {
            const JwksResponse response = options.fetch(config.jwks_uri, kJwksFetchTimeout);
            fetched = parse_jwks(response.body, config.algorithms);
            advertised = cache_control_max_age(response.cache_control);
        } catch (const std::exception& error) {
            // Never fall back to stale keys: drop an expired set so nothing
            // can read it, and fail closed.
            if (!usable_locked(monotonic())) keys.reset();
            std::cerr << "vgi hosted: JWKS fetch failed (" << error.what() << ")\n";
            throw vgi_rpc::AuthUnavailableError("JWKS endpoint unavailable");
        } catch (...) {
            if (!usable_locked(monotonic())) keys.reset();
            std::cerr << "vgi hosted: JWKS fetch failed\n";
            throw vgi_rpc::AuthUnavailableError("JWKS endpoint unavailable");
        }
        auto ttl = std::chrono::duration_cast<std::chrono::seconds>(kJwksMaxAge);
        if (advertised && *advertised < ttl) ttl = *advertised;
        // Measured from when the fetch started, so a slow response cannot
        // stretch the bound.
        expires_at = started + ttl;
        keys = fetched;
        return fetched;
    }

    std::shared_ptr<const KeySet> current() {
        std::lock_guard<std::mutex> lock(mutex);
        if (usable_locked(monotonic())) return keys;
        return fetch_locked();
    }

    // Re-fetch after an unknown `kid`; null when rate limited. A refresh
    // another thread finished while this one waited is reused rather than
    // stampeding the endpoint during a rotation.
    std::shared_ptr<const KeySet> refresh_for_unknown_kid(
        const std::shared_ptr<const KeySet>& seen) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto at = monotonic();
        if (keys != nullptr && keys != seen && usable_locked(at)) return keys;
        if (last_fetch_attempt && at - *last_fetch_attempt < kJwksRefreshMinInterval) {
            return nullptr;
        }
        return fetch_locked();
    }
};

PassVerifier::PassVerifier(Config config) : PassVerifier(std::move(config), Options{}) {}

PassVerifier::PassVerifier(Config config, Options options) : state_(std::make_unique<State>()) {
    if (config.issuer.empty() || config.audience.empty() || config.jwks_uri.empty()) {
        throw ConfigError("hosted mode needs an issuer, an audience and a JWKS URI");
    }
    validate_algorithms(config.algorithms, [&] {
        std::string joined;
        for (const auto& a : config.algorithms) joined += (joined.empty() ? "" : ",") + a;
        return joined;
    }());
    if (!options.fetch) options.fetch = fetch_jwks;
    state_->config = std::move(config);
    state_->options = std::move(options);
}

PassVerifier::~PassVerifier() = default;

const Config& PassVerifier::config() const noexcept {
    return state_->config;
}

vgi_rpc::AuthContext PassVerifier::verify(const std::string& token) const {
    const Config& config = state_->config;
    if (token.empty() || token.size() > kMaxPassBytes) reject("not a compact JWS");

    // Compact serialization: exactly three base64url segments.
    const auto first = token.find('.');
    const auto second = first == std::string::npos ? first : token.find('.', first + 1);
    if (first == std::string::npos || second == std::string::npos ||
        token.find('.', second + 1) != std::string::npos || first == 0 || second == first + 1 ||
        second + 1 == token.size()) {
        reject("not a compact JWS");
    }
    const std::string header_b64 = token.substr(0, first);
    const std::string payload_b64 = token.substr(first + 1, second - first - 1);
    const auto header_raw = base64url_decode(header_b64);
    const auto payload_raw = base64url_decode(payload_b64);
    const auto signature = base64url_decode(token.substr(second + 1));
    if (!header_raw || !payload_raw || !signature) reject("not a compact JWS");

    nlohmann::json header;
    nlohmann::json claims;
    try {
        header = nlohmann::json::parse(*header_raw);
        claims = nlohmann::json::parse(*payload_raw);
    } catch (const std::exception&) {
        reject("malformed header or claims");
    }
    if (!header.is_object() || !claims.is_object()) reject("malformed header or claims");

    // Header: an allowed `alg`, `typ: at+jwt`, no extension we would have to
    // understand (RFC 7515 §4.1.11).
    const auto alg = string_member(header, "alg");
    if (!alg || std::find(config.algorithms.begin(), config.algorithms.end(), *alg) ==
                    config.algorithms.end()) {
        reject("algorithm not allowed");
    }
    const auto typ = string_member(header, "typ");
    if (!typ || normalize_typ(*typ) != kRequiredTyp) reject("typ is not at+jwt");
    if (header.contains("crit")) reject("unsupported critical header");
    std::optional<std::string> kid;
    if (header.contains("kid")) {
        kid = string_member(header, "kid");
        if (!kid) reject("malformed kid");
    }
    const auto& spec = supported_algorithms().at(*alg);
    if (spec.signature_bytes != 0 && signature->size() != spec.signature_bytes) {
        reject("bad signature");
    }

    // The key: by `kid` when the Pass names one, else the only key that
    // verifies this algorithm.
    const auto find_check = [&](const KeySet& set, bool& kid_known) -> const SignatureCheck* {
        kid_known = !kid.has_value();
        const SignatureCheck* found = nullptr;
        size_t candidates = 0;
        for (const auto& key : set.keys) {
            if (kid) {
                if (key.kid != kid) continue;
                kid_known = true;
            }
            const auto check = key.checks.find(*alg);
            if (check == key.checks.end()) continue;
            if (found == nullptr) found = &check->second;
            ++candidates;
        }
        if (!kid && candidates != 1) return nullptr;
        return found;
    };

    auto keys = state_->current();
    bool kid_known = false;
    const SignatureCheck* check = find_check(*keys, kid_known);
    if (check == nullptr && !kid_known) {
        // Unknown kid: refresh (rate limited) and look once more.
        auto refreshed = state_->refresh_for_unknown_kid(keys);
        if (refreshed == nullptr) reject("unknown kid");
        keys = std::move(refreshed);
        check = find_check(*keys, kid_known);
        if (check == nullptr) reject(kid_known ? "no key for this algorithm" : "unknown kid");
    } else if (check == nullptr) {
        reject("no key for this algorithm");
    }
    std::error_code ec;
    (*check)(header_b64 + "." + payload_b64, *signature, ec);
    if (ec) reject("bad signature");

    // Claims (RFC 9068 §2.2, §4).
    const int64_t now = state_->now();
    const auto iss = claims.find("iss");
    if (iss == claims.end() || !iss->is_string() || iss->get<std::string>() != config.issuer) {
        reject("wrong issuer");
    }
    const auto aud = claims.find("aud");
    const bool aud_ok =
        aud != claims.end() && ((aud->is_string() && aud->get<std::string>() == config.audience) ||
                                (aud->is_array() && aud->size() == 1 && (*aud)[0].is_string() &&
                                 (*aud)[0].get<std::string>() == config.audience));
    if (!aud_ok) reject("wrong audience");
    const auto exp = claims.find("exp");
    if (exp == claims.end() || !is_numeric_date(*exp)) reject("missing or malformed exp");
    if (as_double(*exp) < static_cast<double>(now - kLeewaySeconds)) {
        reject(vgi_rpc::AuthReason::EXPIRED_CREDENTIAL, "expired");
    }
    if (const auto nbf = claims.find("nbf"); nbf != claims.end()) {
        if (!is_numeric_date(*nbf)) reject("malformed nbf");
        if (as_double(*nbf) > static_cast<double>(now + kLeewaySeconds)) reject("not yet valid");
    }
    if (const auto iat = claims.find("iat"); iat != claims.end() && !is_numeric_date(*iat)) {
        reject("malformed iat");
    }
    const auto sub = claims.find("sub");
    if (sub == claims.end() || !sub->is_string() || sub->get<std::string>().empty()) {
        reject("missing sub");
    }

    vgi_rpc::AuthContext auth;
    auth.domain = kDomain;
    auth.authenticated = true;
    auth.principal = sub->get<std::string>();
    auth.claims = std::move(claims);
    return auth;
}

}  // namespace vgi::hosted
