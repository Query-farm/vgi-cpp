// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Secret scope matching against DuckDB's own rule (BaseSecret::MatchScore,
// Haybarn 2.0.1 engine, duckdb#28684): an authority-only URI scope has an
// implicit trailing `/`. data/scope_match_vectors.json holds 3025
// {scope, path, match} cases generated from the engine's C++, copied verbatim
// from the cross-SDK vector file shipped with the vgi-python fix.
#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <nlohmann/json.hpp>
#include <vgi_rpc/arrow_utils.h>

#include "vgi/settings.h"
#include "wire.h"

#ifndef VGI_SCOPE_MATCH_VECTORS
#error "VGI_SCOPE_MATCH_VECTORS must name scope_match_vectors.json"
#endif

namespace {

struct Case {
    const char* scope;
    const char* path;
    bool match;
};

// The first 25 cases of the vector file, inline so they read as documentation.
const Case kCurated[] = {
    {"snmp://10.0.0.1", "snmp://10.0.0.1", true},
    {"snmp://10.0.0.1", "snmp://10.0.0.1/x", true},
    {"snmp://10.0.0.1", "snmp://10.0.0.15", false},
    {"snmp://10.0.0.1", "snmp://10.0.0.1:161", false},
    {"s3://bucket", "s3://bucket/key", true},
    {"s3://bucket", "s3://bucket-evil/key", false},
    {"s3://bucket", "s3://bucket?x=1", false},
    {"dlt://vgi_dlt_fixture", "dlt://vgi_dlt_fixture:shop", false},
    {"dlt://vgi_dlt_fixture", "dlt://vgi_dlt_fixture/shop", true},
    {"s3://bucket/", "s3://bucket/key", true},
    {"s3://bucket/pre", "s3://bucket/prefix/x", true},
    {"s3://bucket/pre", "s3://bucket/other", false},
    {"https://h?a", "https://h?ab", true},
    {"https://h#f", "https://h#fx", true},
    {"a://b\\c", "a://b\\cd", true},
    {"snmp://", "snmp://anything", true},
    {"snmp://10.0.0.", "snmp://10.0.0.15:161", false},
    {"foo", "foobar", true},
    {"c:\\data", "c:\\data2\\x", true},
    {"x:/y://z", "x:/y://zz", true},
    {"s3://bucket", "s3://buck", false},
    {"s3://bucket", "s3://bucket", true},
    {"vgi-conformance://host1", "vgi-conformance://host1/data", true},
    {"vgi-conformance://host1", "vgi-conformance://host10/data", false},
    {"vgi-conformance://host1", "vgi-conformance://host1:9000", false},
};

std::shared_ptr<arrow::Array> one_string(const std::string& value) {
    arrow::StringBuilder builder;
    VGI_RPC_THROW_NOT_OK(builder.Append(value));
    return vgi_rpc::unwrap(builder.Finish());
}

// One secret column the way the extension ships it: a one-row struct of
// string fields.
std::shared_ptr<arrow::Array> secret(const std::string& scope, const std::string& api_key) {
    return vgi_rpc::unwrap(
        arrow::StructArray::Make({one_string("snmp"), one_string(scope), one_string(api_key)},
                                 std::vector<std::string>{"type", "scope", "api_key"}));
}

}  // namespace

TEST_CASE("scope_matches: curated cases", "[secrets]") {
    for (const auto& c : kCurated) {
        INFO("scope=" << c.scope << " path=" << c.path);
        CHECK(vgi::scope_matches(c.scope, c.path) == c.match);
    }
    CHECK_FALSE(vgi::scope_matches("", "s3://bucket/x"));
}

TEST_CASE("scope_matches: agrees with DuckDB on every shared vector", "[secrets]") {
    std::ifstream in(VGI_SCOPE_MATCH_VECTORS, std::ios::binary);
    REQUIRE(in.good());
    std::stringstream text;
    text << in.rdbuf();
    const auto vectors = nlohmann::json::parse(text.str());
    const auto& cases = vectors.at("cases");
    REQUIRE(cases.size() > 3000);
    size_t mismatches = 0;
    for (const auto& c : cases) {
        const auto scope = c.at("scope").get<std::string>();
        const auto path = c.at("path").get<std::string>();
        const bool expected = c.at("match").get<bool>();
        if (vgi::scope_matches(scope, path) != expected) {
            ++mismatches;
            UNSCOPED_INFO("scope=" << nlohmann::json(scope).dump() << " path="
                                   << nlohmann::json(path).dump() << " expected=" << expected);
        }
    }
    CHECK(mismatches == 0);
}

TEST_CASE("for_scope_of_type stops at the authority boundary", "[secrets]") {
    auto schema = arrow::schema({
        arrow::field("host1", secret("", "")->type()),
        arrow::field("host15", secret("", "")->type()),
        arrow::field("fallback", secret("", "")->type()),
    });
    auto batch =
        arrow::RecordBatch::Make(schema, 1,
                                 {secret("snmp://10.0.0.1", "host1"),
                                  secret("snmp://10.0.0.15", "host15"), secret("", "default")});
    const auto secrets = vgi::Secrets::parse(vgi::wire::encode_ipc(batch));

    const std::pair<const char*, const char*> expectations[] = {
        {"snmp://10.0.0.1/sys", "host1"},
        {"snmp://10.0.0.15", "host15"},
        {"snmp://10.0.0.1:161", "default"},
    };
    for (const auto& [path, key] : expectations) {
        INFO("path=" << path);
        const auto* found = secrets.for_scope_of_type(path, "snmp");
        REQUIRE(found != nullptr);
        CHECK(found->at("api_key") == key);
    }
}

TEST_CASE("for_scope_of_type matches each newline-joined prefix", "[secrets]") {
    auto schema = arrow::schema({arrow::field("multi", secret("", "")->type())});
    auto batch =
        arrow::RecordBatch::Make(schema, 1, {secret("snmp://10.0.0.9\nsnmp://10.0.0.1", "multi")});
    const auto secrets = vgi::Secrets::parse(vgi::wire::encode_ipc(batch));
    const auto* found = secrets.for_scope_of_type("snmp://10.0.0.1/sys", "snmp");
    REQUIRE(found != nullptr);
    CHECK(found->at("api_key") == "multi");
    CHECK(secrets.for_scope_of_type("snmp://10.0.0.15", "snmp") == nullptr);
}
