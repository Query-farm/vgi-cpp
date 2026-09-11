// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include <catch2/catch_test_macros.hpp>

#include <arrow/builder.h>
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

    const auto function_info = vgi::generated::FunctionInfoSchema();
    REQUIRE(function_info->GetFieldByName("parameter_default_values")->nullable());
    REQUIRE(function_info->GetFieldByName("filter_semantic_profiles")->type()->id() ==
            arrow::Type::LIST);
    REQUIRE(function_info->GetFieldByName("additional_filter_functions")->type()->id() ==
            arrow::Type::LIST);
    REQUIRE(function_info->GetFieldByName("runtime_filter_algorithms")->type()->id() ==
            arrow::Type::LIST);
    REQUIRE(function_info->GetFieldByName("filter_evaluation_contexts")->type()->id() ==
            arrow::Type::LIST);
    REQUIRE(function_info->GetFieldByName("filters_exactly_applied")->type()->id() ==
            arrow::Type::BOOL);
    REQUIRE(vgi::generated::BindRequestSchema()->GetFieldByName("argument_names")->nullable());
    REQUIRE(
        vgi::generated::AggregateBindRequestSchema()->GetFieldByName("argument_names")->nullable());
}

TEST_CASE("filter pushdown advertises only the implemented semantic profile", "[protocol]") {
    vgi::FunctionMetadata plain;
    REQUIRE(plain.resolved_filter_semantic_profiles().empty());

    vgi::FunctionMetadata filtering;
    filtering.filter_pushdown = true;
    REQUIRE(filtering.resolved_filter_semantic_profiles() ==
            std::vector<std::string>{vgi::filter_semantic_profiles::kDuckDBStandardV1});
    filtering.auto_apply_filters = true;
    REQUIRE_FALSE(filtering.filters_exactly_applied);

    filtering.filter_semantic_profiles = {"vgi.duckdb.standard.v2"};
    REQUIRE_THROWS(filtering.resolved_filter_semantic_profiles());
}

TEST_CASE("filter capability structs preserve generated FunctionInfo shape", "[protocol]") {
    const auto batch = vgi::wire::ResultBuilder(vgi::generated::FunctionInfoSchema())
                           .set_string_list("filter_semantic_profiles", {"vgi.duckdb.standard.v1"})
                           .set_filter_identities("additional_filter_functions",
                                                  {{"duckdb.spatial", "intersects_extent", 1}})
                           .set_filter_identities("runtime_filter_algorithms",
                                                  {{"duckdb.runtime_filter", "bloom", 2}})
                           .set_evaluation_contexts("filter_evaluation_contexts",
                                                    {{"vgi.duckdb.session.v1", "duckdb-icu:test"},
                                                     {"vgi.duckdb.session.v1", std::nullopt}})
                           .fill_defaults()
                           .finish();

    const auto identities = std::static_pointer_cast<arrow::ListArray>(
        batch->GetColumnByName("additional_filter_functions"));
    REQUIRE(identities->value_length(0) == 1);
    const auto identity = std::static_pointer_cast<arrow::StructArray>(identities->values());
    const auto namespaces =
        std::static_pointer_cast<arrow::StringArray>(identity->GetFieldByName("namespace"));
    const auto names =
        std::static_pointer_cast<arrow::StringArray>(identity->GetFieldByName("name"));
    const auto versions =
        std::static_pointer_cast<arrow::UInt64Array>(identity->GetFieldByName("version"));
    REQUIRE(namespaces->GetString(0) == "duckdb.spatial");
    REQUIRE(names->GetString(0) == "intersects_extent");
    REQUIRE(versions->Value(0) == 1);

    const auto contexts = std::static_pointer_cast<arrow::ListArray>(
        batch->GetColumnByName("filter_evaluation_contexts"));
    REQUIRE(contexts->value_length(0) == 2);
    const auto context = std::static_pointer_cast<arrow::StructArray>(contexts->values());
    const auto fingerprints = std::static_pointer_cast<arrow::StringArray>(
        context->GetFieldByName("provider_fingerprint"));
    REQUIRE(fingerprints->GetString(0) == "duckdb-icu:test");
    REQUIRE(fingerprints->IsNull(1));
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
