// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include <catch2/catch_test_macros.hpp>

#include <arrow/builder.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include "vgi/function.h"
#include "vgi/catalog.h"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "vgi/pushdown.h"
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

std::string filter_batch(const std::string& document,
                         const std::shared_ptr<arrow::Array>& payload = nullptr) {
    arrow::StringBuilder spec_builder;
    REQUIRE(spec_builder.Append(document).ok());
    std::shared_ptr<arrow::Array> spec;
    REQUIRE(spec_builder.Finish(&spec).ok());
    arrow::FieldVector fields{arrow::field("filter_spec", arrow::utf8(), false)};
    arrow::ArrayVector arrays{spec};
    if (payload) {
        fields.push_back(arrow::field("value_0", payload->type(), true));
        arrays.push_back(payload);
    }
    auto metadata = arrow::key_value_metadata(
        {"vgi_filter_encoding", "vgi_filter_version", "vgi_evaluation_context"},
        {"vgi.filters.v2", "2", "vgi.none.v1"});
    return vgi::wire::encode_ipc(
        arrow::RecordBatch::Make(arrow::schema(fields, metadata), 1, arrays));
}

std::shared_ptr<arrow::Array> int64_values(std::initializer_list<int64_t> values) {
    arrow::Int64Builder builder;
    for (const auto value : values) REQUIRE(builder.Append(value).ok());
    std::shared_ptr<arrow::Array> result;
    REQUIRE(builder.Finish(&result).ok());
    return result;
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

TEST_CASE("Filter v2 snapshot binds and applies typed comparison payloads", "[filter-v2]") {
    const auto document =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[{"id":"p","revision":0,"mode":"required","source":"query","expression":{"node":"comparison","op":"ge","left":{"node":"column_ref","column_index":0,"column_name":"n"},"right":{"node":"literal","value_ref":0}}}]})";
    auto filters = vgi::PushdownFilters::parse(filter_batch(document, int64_values({3})), {},
                                               arrow::schema({arrow::field("n", arrow::int64())}));
    REQUIRE(filters.format() == "n >= 3");
    REQUIRE(filters.filtered_columns() == std::vector<std::string>{"n"});
    REQUIRE(filters.column_bounds("n").min == 3);

    auto input = arrow::RecordBatch::Make(arrow::schema({arrow::field("n", arrow::int64())}), 5,
                                          {int64_values({1, 2, 3, 4, 5})});
    const auto output = filters.apply(input);
    REQUIRE(output->num_rows() == 3);
}

TEST_CASE("Filter v2 dynamic deltas update snapshot state atomically", "[filter-v2]") {
    const auto snapshot =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[]})";
    auto filters = vgi::PushdownFilters::parse(filter_batch(snapshot), {},
                                               arrow::schema({arrow::field("n", arrow::int64())}));
    const auto delta =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"delta","updates":[{"operation":"upsert","id":"topn","revision":1,"mode":"advisory","source":"top_n","expression":{"node":"comparison","op":"lt","left":{"node":"column_ref","column_index":0,"column_name":"n"},"right":{"node":"literal","value_ref":0}}}]})";
    filters.apply_delta(filter_batch(delta, int64_values({4})));
    REQUIRE(filters.format_repr() == "PushdownFilters([ConstantFilter(n < 4)])");
    auto input = arrow::RecordBatch::Make(arrow::schema({arrow::field("n", arrow::int64())}), 5,
                                          {int64_values({1, 2, 3, 4, 5})});
    REQUIRE(filters.apply(input)->num_rows() == 3);
}
