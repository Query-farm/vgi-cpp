// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include <catch2/catch_test_macros.hpp>

#include <arrow/type.h>

#include "vgi/function.h"
#include "vgi/catalog.h"
#include "vgi/generated/vgi_protocol_schemas.hpp"
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
}

TEST_CASE("wire schema paths round trip every component", "[wire]") {
    const auto schema = arrow::schema(
        {arrow::field("schema_path", arrow::list(arrow::utf8()), /*nullable=*/false)});
    const vgi::SchemaPath nested{"analytics", "sales"};
    const auto batch =
        vgi::wire::ResultBuilder(schema).set_string_list("schema_path", nested).finish();
    REQUIRE(vgi::wire::get_schema_path(batch) == nested);
}
