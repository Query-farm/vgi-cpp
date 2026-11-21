// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Test-only Pass minting for the hosted-mode tests: ES256 / ES384 / RS256 keys
// generated with OpenSSL, their public JWKs, and compact JWS signing with
// full control of the header and claims (so a test can mint a Pass with no
// exp, two audiences, a crit header, ...).

#pragma once

#include <string>
#include <system_error>

#include <catch2/catch_test_macros.hpp>
#include <jwt-cpp/jwt.h>
#include <nlohmann/json.hpp>
#include <openssl/bio.h>
#include <openssl/core_names.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

namespace pass_minter {

inline std::string b64u(const std::string& raw) {
    return jwt::base::trim<jwt::alphabet::base64url>(
        jwt::base::encode<jwt::alphabet::base64url>(raw));
}

inline std::string bn_bytes(EVP_PKEY* key, const char* param, int width) {
    BIGNUM* bn = nullptr;
    REQUIRE(EVP_PKEY_get_bn_param(key, param, &bn) == 1);
    std::string out(width > 0 ? width : BN_num_bytes(bn), '\0');
    REQUIRE(BN_bn2binpad(bn, reinterpret_cast<unsigned char*>(out.data()),
                         static_cast<int>(out.size())) > 0);
    BN_free(bn);
    return out;
}

inline std::string private_pem(EVP_PKEY* key) {
    BIO* bio = BIO_new(BIO_s_mem());
    REQUIRE(PEM_write_bio_PrivateKey(bio, key, nullptr, nullptr, 0, nullptr, nullptr) == 1);
    char* data = nullptr;
    const long size = BIO_get_mem_data(bio, &data);
    std::string pem(data, static_cast<size_t>(size));
    BIO_free(bio);
    return pem;
}

// A signing key and its public JWK.
struct SigningKey {
    std::string kid;
    std::string alg;
    std::string pem;
    nlohmann::json jwk;

    static SigningKey ec(const std::string& kid, const char* curve = "P-256",
                         const std::string& alg = "ES256", int width = 32) {
        EVP_PKEY* key = EVP_EC_gen(curve);
        REQUIRE(key != nullptr);
        SigningKey out{kid, alg, private_pem(key), {}};
        out.jwk = {{"kty", "EC"},
                   {"crv", curve},
                   {"x", b64u(bn_bytes(key, OSSL_PKEY_PARAM_EC_PUB_X, width))},
                   {"y", b64u(bn_bytes(key, OSSL_PKEY_PARAM_EC_PUB_Y, width))},
                   {"kid", kid},
                   {"use", "sig"}};
        EVP_PKEY_free(key);
        return out;
    }

    static SigningKey rsa(const std::string& kid) {
        EVP_PKEY* key = EVP_RSA_gen(2048);
        REQUIRE(key != nullptr);
        SigningKey out{kid, "RS256", private_pem(key), {}};
        out.jwk = {{"kty", "RSA"},
                   {"n", b64u(bn_bytes(key, OSSL_PKEY_PARAM_RSA_N, 0))},
                   {"e", b64u(bn_bytes(key, OSSL_PKEY_PARAM_RSA_E, 0))},
                   {"kid", kid}};
        EVP_PKEY_free(key);
        return out;
    }

    std::string sign(const std::string& data) const {
        std::error_code ec;
        std::string signature;
        if (alg == "ES256") signature = jwt::algorithm::es256("", pem).sign(data, ec);
        if (alg == "ES384") signature = jwt::algorithm::es384("", pem).sign(data, ec);
        if (alg == "RS256") signature = jwt::algorithm::rs256("", pem).sign(data, ec);
        REQUIRE_FALSE(ec);
        return signature;
    }
};

inline std::string mint_pass(const SigningKey& key, const nlohmann::json& claims,
                             nlohmann::json header = nullptr) {
    if (header.is_null()) header = {{"alg", key.alg}, {"typ", "at+jwt"}, {"kid", key.kid}};
    const std::string signing_input = b64u(header.dump()) + "." + b64u(claims.dump());
    return signing_input + "." + b64u(key.sign(signing_input));
}

}  // namespace pass_minter
