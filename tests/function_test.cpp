// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <vector>

#include <arrow/builder.h>
#include <arrow/type.h>

#include "vgi/function.h"
#include "vgi/catalog.h"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "split_token.h"
#include "wire.h"

namespace {

class Fixed : public vgi::ScalarFunction {
public:
    std::string name() const override { return "fixed"; }
    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.return_type = arrow::int64();
        return md;
    }
    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }
    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams&, const std::shared_ptr<arrow::RecordBatch>&) const override {
        return nullptr;
    }
};

class Dynamic : public Fixed {
public:
    std::string name() const override { return "dynamic"; }
    vgi::FunctionMetadata metadata() const override { return {}; }
};

std::string from_hex(const std::string& hex) {
    std::string out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<char>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
}

vgi_rpc::AuthContext authenticated_as(const std::string& principal) {
    auto auth = vgi_rpc::AuthContext::anonymous();
    auth.domain = "test";
    auth.authenticated = true;
    auth.principal = principal;
    return auth;
}

}  // namespace

TEST_CASE("a fixed return type binds without an override", "[function]") {
    Fixed f;
    auto schema = f.bind({});
    REQUIRE(schema->num_fields() == 1);
    REQUIRE(schema->field(0)->name() == "result");
    REQUIRE(schema->field(0)->type()->Equals(*arrow::int64()));
}

TEST_CASE("a function with no fixed return type owes an override", "[function]") {
    // Silently returning a null schema here would surface as a confusing
    // failure deep in dispatch, so bind() refuses instead.
    Dynamic d;
    REQUIRE_THROWS(d.bind({}));
}

TEST_CASE("argument specs carry their kind", "[function]") {
    auto col = vgi::ArgSpec::column("v", 0, "varchar");
    REQUIRE(col.index.has_value());
    REQUIRE_FALSE(col.constant);
    REQUIRE(col.required);

    auto k = vgi::ArgSpec::constant_arg("k", 1, "bigint");
    REQUIRE(k.constant);

    auto n = vgi::ArgSpec::named("opt", "varchar");
    REQUIRE_FALSE(n.index.has_value());
    REQUIRE_FALSE(n.required);
}

TEST_CASE("catalog schemas preserve arbitrary nested paths", "[catalog]") {
    vgi::CatalogModel catalog;
    const vgi::SchemaPath nested{"analytics", "sales", "quarterly"};
    auto& schema = catalog.schema(nested);

    REQUIRE(schema.path == nested);
    REQUIRE(catalog.find_schema(nested) == &schema);
    REQUIRE(catalog.find_schema("analytics") == nullptr);
    REQUIRE(catalog.schema_paths().back() == nested);
}

TEST_CASE("protocol v2 named schemas use path lists", "[protocol]") {
    const auto schema_info = vgi::generated::SchemaInfoSchema();
    REQUIRE(schema_info->GetFieldByName("path")->type()->id() == arrow::Type::LIST);
    REQUIRE(schema_info->GetFieldByName("name") == nullptr);

    const auto foreign_key = vgi::generated::ForeignKeyInfoSchema();
    REQUIRE(foreign_key->GetFieldByName("referenced_schema_path")->type()->id() ==
            arrow::Type::LIST);
    REQUIRE(foreign_key->GetFieldByName("referenced_schema") == nullptr);

    const auto capabilities = vgi::generated::ClientCapabilitiesSchema();
    REQUIRE(capabilities->GetFieldByName("engine")->type()->id() == arrow::Type::STRING);
    REQUIRE(capabilities->GetFieldByName("native_formats")->type()->id() == arrow::Type::LIST);
    REQUIRE(capabilities->GetFieldByName("catalogs")->type()->id() == arrow::Type::LIST);
    REQUIRE(capabilities->GetFieldByName("can_stream")->type()->id() == arrow::Type::BOOL);
    REQUIRE(capabilities->GetFieldByName("filter_encodings")->type()->id() == arrow::Type::LIST);

    const auto function_info = vgi::generated::FunctionInfoSchema();
    REQUIRE(function_info->GetFieldByName("parameter_default_values")->nullable());
    REQUIRE(vgi::generated::BindRequestSchema()->GetFieldByName("argument_names")->nullable());
    REQUIRE(
        vgi::generated::AggregateBindRequestSchema()->GetFieldByName("argument_names")->nullable());
}

TEST_CASE("wire schema paths round trip every component", "[wire]") {
    const auto schema = arrow::schema(
        {arrow::field("schema_path", arrow::list(arrow::utf8()), /*nullable=*/false)});
    const vgi::SchemaPath nested{"analytics", "sales"};
    const auto batch =
        vgi::wire::ResultBuilder(schema).set_string_list("schema_path", nested).finish();
    REQUIRE(vgi::wire::get_schema_path(batch) == nested);
}

TEST_CASE("wire argument names preserve unnamed varargs", "[wire]") {
    auto values = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder builder(arrow::default_memory_pool(), values);
    REQUIRE(builder.Append().ok());
    REQUIRE(values->Append("left").ok());
    REQUIRE(values->AppendNull().ok());
    REQUIRE(values->Append("scale").ok());
    std::shared_ptr<arrow::Array> names;
    REQUIRE(builder.Finish(&names).ok());

    const auto schema = arrow::schema(
        {arrow::field("argument_names", arrow::list(arrow::utf8()), /*nullable=*/true)});
    const auto batch = arrow::RecordBatch::Make(schema, 1, {names});
    const auto decoded = vgi::wire::get_optional_string_list(batch, "argument_names");
    REQUIRE(decoded.has_value());
    REQUIRE(decoded->size() == 3);
    REQUIRE(decoded->at(0) == "left");
    REQUIRE_FALSE(decoded->at(1).has_value());
    REQUIRE(decoded->at(2) == "scale");
}

TEST_CASE("split token shared vectors reach their declared verdict", "[split-token]") {
    // Byte-for-byte copies of the canonical vectors generated by
    // vgi-python/tests/data/split_tokens/generate.py. Keeping the raw bytes in
    // the test catches a parser that is self-consistent but disagrees with the
    // other SDKs about the envelope layout.
    struct Vector {
        const char* name;
        const char* token_hex;
        vgi::split_token::OpenError verdict;
        bool keyed;
    };
    const std::vector<Vector> vectors = {
        {"valid_unsealed",
         "01000800000102030405060708090a0b0c0d0e0f2f0000000000000066696c653d333b763d3437",
         vgi::split_token::OpenError::None, false},
        {"valid_sealed",
         "01010800000102030405060708090a0b0c0d0e0f2f0000000000000001551ed7558e77fdcea243a"
         "2394af27e800146335fb36c7c1f99ae3b13773a7466926cc4f5c1e3701d1d9234182a3b7588852364",
         vgi::split_token::OpenError::None, true},
        {"bad_flags_unsealed_but_key_present",
         "01000800000102030405060708090a0b0c0d0e0f2f000000000000004f544845522054454e414e54204441544"
         "1",
         vgi::split_token::OpenError::Invalid, true},
        {"bad_fingerprint",
         "01000800eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee2f0000000000000066696c653d333b763d3437",
         vgi::split_token::OpenError::Invalid, false},
        {"stale_anchor",
         "01000800000102030405060708090a0b0c0d0e0f010000000000000066696c653d333b763d3437",
         vgi::split_token::OpenError::SnapshotExpired, false},
        {"truncated", "01000800000102030405", vgi::split_token::OpenError::Invalid, false},
        {"reserved_flag_bit",
         "01020800000102030405060708090a0b0c0d0e0f2f0000000000000066696c653d333b763d3437",
         vgi::split_token::OpenError::Invalid, false},
        {"bad_version",
         "09000800000102030405060708090a0b0c0d0e0f2f0000000000000066696c653d333b763d3437",
         vgi::split_token::OpenError::Invalid, false},
        {"anchor_len_overrun",
         "01000f27000102030405060708090a0b0c0d0e0f2f0000000000000066696c653d333b763d3437",
         vgi::split_token::OpenError::Invalid, false},
    };

    vgi::split_token::SigningKey key{};
    for (size_t i = 0; i < key.size(); ++i) key[i] = static_cast<uint8_t>(i);
    const auto fingerprint = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto anchor = from_hex("2f00000000000000");
    const auto anonymous = vgi_rpc::AuthContext::anonymous();

    for (const auto& vector : vectors) {
        CAPTURE(vector.name);
        const std::optional<vgi::split_token::SigningKey> signing_key =
            vector.keyed ? std::optional<vgi::split_token::SigningKey>(key) : std::nullopt;
        const auto opened = vgi::split_token::open(from_hex(vector.token_hex), fingerprint, anchor,
                                                   signing_key, anonymous);
        REQUIRE(opened.error == vector.verdict);
        if (vector.verdict == vgi::split_token::OpenError::None) {
            REQUIRE(opened.payload == std::optional<std::string>("file=3;v=47"));
        } else {
            REQUIRE_FALSE(opened.payload.has_value());
        }
    }
}

TEST_CASE("split token stamping matches the shared unsealed vector", "[split-token]") {
    const auto fingerprint = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto anchor = vgi::split_token::anchor_for(47);
    const auto token = vgi::split_token::build("file=3;v=47", fingerprint, anchor, std::nullopt,
                                               vgi_rpc::AuthContext::anonymous());
    REQUIRE(
        token ==
        from_hex("01000800000102030405060708090a0b0c0d0e0f2f0000000000000066696c653d333b763d3437"));
}

TEST_CASE("sealed split tokens are bound to key, header, and identity", "[split-token]") {
    vgi::split_token::SigningKey key{};
    key.fill(0x11);
    auto wrong_key = key;
    wrong_key[0] ^= 0xff;
    const std::string fingerprint(16, '\x05');
    const auto anchor = vgi::split_token::anchor_for(1);
    const auto alice = authenticated_as("alice");
    const auto bob = authenticated_as("bob");

    auto token = vgi::split_token::build("tenant=alice", fingerprint, anchor, key, alice);
    REQUIRE((static_cast<uint8_t>(token[1]) & vgi::split_token::kFlagPayloadSealed) != 0);
    REQUIRE(token.find("tenant=alice") == std::string::npos);
    REQUIRE(vgi::split_token::open(token, fingerprint, anchor, key, alice).payload ==
            std::optional<std::string>("tenant=alice"));
    REQUIRE(vgi::split_token::open(token, fingerprint, anchor, key, bob).error ==
            vgi::split_token::OpenError::Invalid);
    REQUIRE(vgi::split_token::open(token, fingerprint, anchor, wrong_key, alice).error ==
            vgi::split_token::OpenError::Invalid);

    token[4] ^= static_cast<char>(0xff);
    REQUIRE(vgi::split_token::open(token, token.substr(4, 16), anchor, key, alice).error ==
            vgi::split_token::OpenError::Invalid);
}

TEST_CASE("a keyed worker refuses alg-none split tokens", "[split-token]") {
    vgi::split_token::SigningKey key{};
    key.fill(0x2a);
    const std::string fingerprint(16, '\x07');
    const auto anchor = vgi::split_token::anchor_for(47);
    const auto anonymous = vgi_rpc::AuthContext::anonymous();

    const auto forged =
        vgi::split_token::build("file=evil", fingerprint, anchor, std::nullopt, anonymous);
    REQUIRE(vgi::split_token::open(forged, fingerprint, anchor, key, anonymous).error ==
            vgi::split_token::OpenError::Invalid);

    const auto sealed = vgi::split_token::build("file=ok", fingerprint, anchor, key, anonymous);
    REQUIRE(vgi::split_token::open(sealed, fingerprint, anchor, std::nullopt, anonymous).error ==
            vgi::split_token::OpenError::Invalid);
}

TEST_CASE("split token bind failures precede stale anchors", "[split-token]") {
    const std::string fingerprint(16, '\x09');
    const std::string other_fingerprint(16, '\x0a');
    const auto old_anchor = vgi::split_token::anchor_for(47);
    const auto current_anchor = vgi::split_token::anchor_for(48);
    const auto anonymous = vgi_rpc::AuthContext::anonymous();
    const auto token =
        vgi::split_token::build("file=1", fingerprint, old_anchor, std::nullopt, anonymous);

    REQUIRE(
        vgi::split_token::open(token, other_fingerprint, current_anchor, std::nullopt, anonymous)
            .error == vgi::split_token::OpenError::Invalid);
    REQUIRE(
        vgi::split_token::open(token, fingerprint, current_anchor, std::nullopt, anonymous).error ==
        vgi::split_token::OpenError::SnapshotExpired);
}
