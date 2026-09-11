// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <optional>

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

std::string filter_batch_with_payloads(
    const std::string& document,
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& payloads) {
    arrow::StringBuilder spec_builder;
    REQUIRE(spec_builder.Append(document).ok());
    std::shared_ptr<arrow::Array> spec;
    REQUIRE(spec_builder.Finish(&spec).ok());
    arrow::FieldVector fields{arrow::field("filter_spec", arrow::utf8(), false)};
    arrow::ArrayVector arrays{spec};
    for (const auto& [name, payload] : payloads) {
        fields.push_back(arrow::field(name, payload->type(), true));
        arrays.push_back(payload);
    }
    auto metadata = arrow::key_value_metadata(
        {"vgi_filter_encoding", "vgi_filter_version", "vgi_evaluation_context"},
        {"vgi.filters.v2", "2", "vgi.none.v1"});
    return vgi::wire::encode_ipc(
        arrow::RecordBatch::Make(arrow::schema(fields, metadata), 1, arrays));
}

std::string filter_batch(const std::string& document,
                         const std::shared_ptr<arrow::Array>& payload = nullptr) {
    if (payload) return filter_batch_with_payloads(document, {{"value_0", payload}});
    return filter_batch_with_payloads(document, {});
}

std::shared_ptr<arrow::Array> int64_values(std::initializer_list<int64_t> values) {
    arrow::Int64Builder builder;
    for (const auto value : values) REQUIRE(builder.Append(value).ok());
    std::shared_ptr<arrow::Array> result;
    REQUIRE(builder.Finish(&result).ok());
    return result;
}

std::shared_ptr<arrow::Array> nullable_int64_values(
    std::initializer_list<std::optional<int64_t>> values) {
    arrow::Int64Builder builder;
    for (const auto value : values) {
        if (value)
            REQUIRE(builder.Append(*value).ok());
        else
            REQUIRE(builder.AppendNull().ok());
    }
    std::shared_ptr<arrow::Array> result;
    REQUIRE(builder.Finish(&result).ok());
    return result;
}

std::shared_ptr<arrow::Array> int64_list(std::initializer_list<std::optional<int64_t>> values) {
    auto child = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder builder(arrow::default_memory_pool(), child);
    REQUIRE(builder.Append().ok());
    for (const auto value : values) {
        if (value)
            REQUIRE(child->Append(*value).ok());
        else
            REQUIRE(child->AppendNull().ok());
    }
    std::shared_ptr<arrow::Array> result;
    REQUIRE(builder.Finish(&result).ok());
    return result;
}

std::shared_ptr<arrow::Array> double_values(std::initializer_list<double> values) {
    arrow::DoubleBuilder builder;
    for (const auto value : values) REQUIRE(builder.Append(value).ok());
    std::shared_ptr<arrow::Array> result;
    REQUIRE(builder.Finish(&result).ok());
    return result;
}

std::shared_ptr<arrow::Array> null_value(const std::shared_ptr<arrow::DataType>& type) {
    auto result = arrow::MakeArrayOfNull(type, 1);
    REQUIRE(result.ok());
    return result.MoveValueUnsafe();
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

TEST_CASE("Filter v2 IN follows SQL three-valued NULL semantics", "[filter-v2]") {
    const auto in_document =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[{"id":"p","revision":0,"mode":"required","source":"query","expression":{"node":"in","expression":{"node":"column_ref","column_index":0,"column_name":"n"},"set":{"kind":"literal","value_ref":0},"negated":false}}]})";
    auto in_filter =
        vgi::PushdownFilters::parse(filter_batch(in_document, int64_list({1, std::nullopt})), {},
                                    arrow::schema({arrow::field("n", arrow::int64())}));
    auto input = arrow::RecordBatch::Make(arrow::schema({arrow::field("n", arrow::int64())}), 3,
                                          {nullable_int64_values({1, 2, std::nullopt})});
    REQUIRE(in_filter.apply(input)->num_rows() == 1);

    const auto not_in_document =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[{"id":"p","revision":0,"mode":"required","source":"query","expression":{"node":"in","expression":{"node":"column_ref","column_index":0,"column_name":"n"},"set":{"kind":"literal","value_ref":0},"negated":true}}]})";
    auto not_in_with_null =
        vgi::PushdownFilters::parse(filter_batch(not_in_document, int64_list({1, std::nullopt})),
                                    {}, arrow::schema({arrow::field("n", arrow::int64())}));
    REQUIRE(not_in_with_null.apply(input)->num_rows() == 0);

    auto not_in_without_null =
        vgi::PushdownFilters::parse(filter_batch(not_in_document, int64_list({1})), {},
                                    arrow::schema({arrow::field("n", arrow::int64())}));
    auto null_input = arrow::RecordBatch::Make(arrow::schema({arrow::field("n", arrow::int64())}),
                                               1, {nullable_int64_values({std::nullopt})});
    REQUIRE(not_in_without_null.apply(null_input)->num_rows() == 0);
}

TEST_CASE("Filter v2 field references propagate parent struct validity", "[filter-v2]") {
    const auto document =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[{"id":"p","revision":0,"mode":"required","source":"query","expression":{"node":"comparison","op":"eq","left":{"node":"field_ref","expression":{"node":"column_ref","column_index":0,"column_name":"s"},"field_index":0,"field_name":"n"},"right":{"node":"literal","value_ref":0}}}]})";
    const auto struct_type = arrow::struct_({arrow::field("n", arrow::int64())});
    auto filters = vgi::PushdownFilters::parse(filter_batch(document, int64_values({5})), {},
                                               arrow::schema({arrow::field("s", struct_type)}));

    arrow::BooleanBuilder validity_builder;
    REQUIRE(validity_builder.Append(false).ok());
    REQUIRE(validity_builder.Append(true).ok());
    std::shared_ptr<arrow::Array> validity;
    REQUIRE(validity_builder.Finish(&validity).ok());
    auto maybe_struct = arrow::StructArray::Make(
        {int64_values({5, 5})}, {arrow::field("n", arrow::int64())},
        std::static_pointer_cast<arrow::BooleanArray>(validity)->values(), 1);
    REQUIRE(maybe_struct.ok());
    auto input = arrow::RecordBatch::Make(arrow::schema({arrow::field("s", struct_type)}), 2,
                                          {maybe_struct.MoveValueUnsafe()});
    REQUIRE(filters.apply(input)->num_rows() == 1);
    REQUIRE_FALSE(filters.column_values("s"));
    REQUIRE_FALSE(filters.column_bounds("s").min);
    REQUIRE_FALSE(filters.column_bounds("s").max);
}

TEST_CASE("Filter v2 comparisons use DuckDB NaN and signed-zero ordering", "[filter-v2]") {
    auto input = arrow::RecordBatch::Make(arrow::schema({arrow::field("n", arrow::float64())}), 4,
                                          {double_values({NAN, -0.0, 0.0, 1.0})});
    const auto rows_for = [&](const std::string& op, double literal) {
        const auto document =
            std::string(
                R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[{"id":"p","revision":0,"mode":"required","source":"query","expression":{"node":"comparison","op":")") +
            op +
            R"(","left":{"node":"column_ref","column_index":0,"column_name":"n"},"right":{"node":"literal","value_ref":0}}}]})";
        auto filters = vgi::PushdownFilters::parse(filter_batch(document, double_values({literal})),
                                                   {}, input->schema());
        return filters.apply(input)->num_rows();
    };
    REQUIRE(rows_for("eq", NAN) == 1);
    REQUIRE(rows_for("lt", NAN) == 3);
    REQUIRE(rows_for("distinct_from", NAN) == 3);
    REQUIRE(rows_for("not_distinct_from", NAN) == 1);
    REQUIRE(rows_for("eq", 0.0) == 2);
}

TEST_CASE("Filter v2 rejects duplicate JSON keys and malformed stale upserts", "[filter-v2]") {
    const auto duplicate =
        R"({"encoding":"vgi.filters.v2","encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[]})";
    REQUIRE_THROWS(vgi::PushdownFilters::parse(filter_batch(duplicate), {}, arrow::schema({})));

    const auto snapshot =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[]})";
    auto filters = vgi::PushdownFilters::parse(filter_batch(snapshot), {}, arrow::schema({}));
    const auto fresh =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"delta","updates":[{"operation":"remove","id":"p","revision":2}]})";
    filters.apply_delta(filter_batch(fresh));
    const auto stale_malformed =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"delta","updates":[{"operation":"upsert","id":"p","revision":1,"mode":"advisory","source":"top_n","expression":{"node":"bogus"}}]})";
    REQUIRE_THROWS(filters.apply_delta(filter_batch(stale_malformed)));
}

TEST_CASE("Filter v2 arithmetic is checked and unsafe casts are declined", "[filter-v2]") {
    const auto arithmetic =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[{"id":"p","revision":0,"mode":"required","source":"query","expression":{"node":"comparison","op":"eq","left":{"node":"arithmetic","op":"add","left":{"node":"column_ref","column_index":0,"column_name":"n"},"right":{"node":"literal","value_ref":0}},"right":{"node":"literal","value_ref":1}}}]})";
    auto filters = vgi::PushdownFilters::parse(
        filter_batch_with_payloads(
            arithmetic, {{"value_0", int64_values({1})}, {"value_1", int64_values({0})}}),
        {}, arrow::schema({arrow::field("n", arrow::int64())}));
    auto overflow = arrow::RecordBatch::Make(arrow::schema({arrow::field("n", arrow::int64())}), 1,
                                             {int64_values({std::numeric_limits<int64_t>::max()})});
    REQUIRE_THROWS(filters.apply(overflow));

    const auto unsafe_cast =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[{"id":"p","revision":0,"mode":"required","source":"query","expression":{"node":"comparison","op":"eq","left":{"node":"cast","expression":{"node":"column_ref","column_index":0,"column_name":"n"},"type_ref":0},"right":{"node":"literal","value_ref":0}}}]})";
    REQUIRE_THROWS(vgi::PushdownFilters::parse(
        filter_batch_with_payloads(
            unsafe_cast, {{"type_0", null_value(arrow::int64())}, {"value_0", int64_values({1})}}),
        {}, arrow::schema({arrow::field("n", arrow::float64())})));
}

TEST_CASE("Filter v2 fails closed when a runtime batch changes column type", "[filter-v2]") {
    const auto document =
        R"({"encoding":"vgi.filters.v2","semantics":"vgi.duckdb.standard.v1","kind":"snapshot","predicates":[{"id":"p","revision":0,"mode":"required","source":"query","expression":{"node":"comparison","op":"eq","left":{"node":"column_ref","column_index":0,"column_name":"n"},"right":{"node":"literal","value_ref":0}}}]})";
    auto filters = vgi::PushdownFilters::parse(filter_batch(document, int64_values({1})), {},
                                               arrow::schema({arrow::field("n", arrow::int64())}));
    arrow::StringBuilder builder;
    REQUIRE(builder.Append("1").ok());
    std::shared_ptr<arrow::Array> strings;
    REQUIRE(builder.Finish(&strings).ok());
    auto changed =
        arrow::RecordBatch::Make(arrow::schema({arrow::field("n", arrow::utf8())}), 1, {strings});
    REQUIRE_THROWS(filters.apply(changed));
}
