// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Canonical constraint records: the public builder and decoder, the serve-time
// checks that mirror the DuckDB extension's ATTACH contract (ENFORCED +
// ALL_ROWS only; Arrow nullability == NOT_NULL/PRIMARY_KEY paths), and the
// read profiles a declarative catalog with records offers on its own.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sys/socket.h>
#include <unistd.h>

#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <arrow/array/builder_primitive.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <vgi_rpc/client.h>
#include <vgi_rpc/errors.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/wire.h>

#include "dispatcher.h"
#include "vgi/catalog.h"
#include "vgi/constraints.h"
#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"
#include "wire.h"

using Catch::Matchers::ContainsSubstring;

namespace {

namespace gen = ::vgi::generated;

const std::string kProtocol(gen::VGI_PROTOCOL_NAME);
const std::string kVersion(gen::VGI_PROTOCOL_VERSION);

vgi::ConstraintRecordSpec spec(const std::string& id, vgi::ConstraintKind kind,
                               std::vector<vgi::ColumnPath> columns) {
    vgi::ConstraintRecordSpec result;
    result.schema_path = {"main"};
    result.table = "t";
    result.id = id;
    result.kind = kind;
    result.columns = std::move(columns);
    result.enforcement = vgi::ConstraintEnforcement::Enforced;
    result.validation = vgi::ConstraintValidation::AllRows;
    return result;
}

std::string record(const std::string& id, vgi::ConstraintKind kind,
                   std::vector<vgi::ColumnPath> columns) {
    return vgi::constraint_record(spec(id, kind, std::move(columns)));
}

std::shared_ptr<arrow::Field> required(const std::string& name,
                                       std::shared_ptr<arrow::DataType> type) {
    return arrow::field(name, std::move(type), /*nullable=*/false);
}

// A Dispatcher served on its own thread over a socket pair.
class Served {
public:
    explicit Served(vgi::Dispatcher& dispatcher) {
        vgi_rpc::ServerBuilder builder;
        builder.protocol(kProtocol).protocol_version(kVersion);
        dispatcher.install(builder);
        server_ = builder.build();
        REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds_) == 0);
        thread_ = std::thread([this] {
            const auto input = std::make_shared<vgi_rpc::FdInputStream>(fds_[1]);
            const auto output = std::make_shared<vgi_rpc::FdOutputStream>(fds_[1]);
            try {
                while (server_->serve_one(input, output)) {
                }
            } catch (const std::exception&) {
            }
        });
        vgi_rpc::RpcClientOptions options;
        options.protocol = kProtocol;
        options.protocol_version = kVersion;
        client_.emplace(vgi_rpc::ClientTransport::from_streams(
                            std::make_shared<vgi_rpc::FdInputStream>(fds_[0]),
                            std::make_shared<vgi_rpc::FdOutputStream>(fds_[0])),
                        options);
    }
    ~Served() {
        client_.reset();
        ::shutdown(fds_[0], SHUT_RDWR);
        thread_.join();
        ::close(fds_[0]);
        ::close(fds_[1]);
    }
    Served(const Served&) = delete;
    Served& operator=(const Served&) = delete;

    std::shared_ptr<arrow::RecordBatch> call(const std::string& method,
                                             const std::shared_ptr<arrow::RecordBatch>& params) {
        const auto response = client_->call_unary(method, params);
        return vgi::wire::decode_ipc(vgi::wire::get_binary(response.batch, "result"));
    }
    std::string refused(const std::string& method,
                        const std::shared_ptr<arrow::RecordBatch>& params) {
        try {
            (void)client_->call_unary(method, params);
        } catch (const vgi_rpc::RpcException& error) {
            return error.what();
        }
        return "<accepted>";
    }

private:
    std::unique_ptr<vgi_rpc::Server> server_;
    int fds_[2] = {-1, -1};
    std::thread thread_;
    std::optional<vgi_rpc::RpcClient> client_;
};

std::string capabilities(const std::vector<std::string>& read) {
    return vgi::wire::encode_ipc(vgi::wire::ResultBuilder(gen::CatalogCapabilitiesSchema())
                                     .set_string_list("read", read)
                                     .set_string_list("ddl", {})
                                     .set_string_list("write", {})
                                     .set_string_list("transaction", {})
                                     .finish());
}

std::shared_ptr<arrow::RecordBatch> attach_params(
    const std::optional<std::vector<std::string>>& offered_read) {
    auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema());
    request.set_string("name", "c");
    if (offered_read) {
        auto client = vgi::wire::ResultBuilder(gen::ClientCapabilitiesSchema())
                          .set_string("engine", "test")
                          .set_binary("catalog", capabilities(*offered_read))
                          .fill_defaults()
                          .finish();
        request.set_binary("client_capabilities", vgi::wire::encode_ipc(client));
    }
    return vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
        .set_binary("request", vgi::wire::encode_ipc(request.fill_defaults().finish()))
        .finish();
}

std::vector<std::string> read_profiles(const std::shared_ptr<arrow::RecordBatch>& result,
                                       const std::string& field) {
    auto bytes = vgi::wire::get_optional_binary(result, field);
    if (!bytes) return {"<absent>"};
    return vgi::wire::get_string_list(vgi::wire::decode_ipc(*bytes), "read");
}

// A declarative catalog: one table with a primary key, no explicit profiles.
vgi::CatalogModel constrained_model(std::shared_ptr<arrow::Schema> columns,
                                    std::vector<std::string> constraints) {
    vgi::CatalogModel model;
    model.name = "c";
    vgi::CatalogTable table;
    table.name = "t";
    table.scan_function = "t_scan";
    table.columns = std::move(columns);
    table.constraints = std::move(constraints);
    model.schema("main").tables.push_back(std::move(table));
    return model;
}

}  // namespace

TEST_CASE("constraint_record round-trips through read_constraint_record", "[constraints]") {
    SECTION("NOT_NULL on a nested path, with a declared name") {
        auto in = spec("t.nn.bbox.xmin", vgi::ConstraintKind::NotNull, {{"bbox", "xmin"}});
        in.declared_name = "xmin_required";
        auto out = vgi::read_constraint_record(vgi::constraint_record(in));
        CHECK(out.schema_path == vgi::SchemaPath{"main"});
        CHECK(out.table == "t");
        CHECK(out.id == "t.nn.bbox.xmin");
        CHECK(out.kind == vgi::ConstraintKind::NotNull);
        CHECK(out.columns == std::vector<vgi::ColumnPath>{{"bbox", "xmin"}});
        CHECK(out.enforcement == vgi::ConstraintEnforcement::Enforced);
        CHECK(out.validation == vgi::ConstraintValidation::AllRows);
        CHECK(out.declared_name == "xmin_required");
        CHECK_FALSE(out.check_sql);
        CHECK_FALSE(out.references);
    }
    SECTION("composite PRIMARY_KEY, attested METADATA_ONLY / UNVALIDATED") {
        auto in = spec("t.pk", vgi::ConstraintKind::PrimaryKey, {{"a"}, {"b"}});
        in.enforcement = vgi::ConstraintEnforcement::MetadataOnly;
        in.validation = vgi::ConstraintValidation::Unvalidated;
        auto out = vgi::read_constraint_record(vgi::constraint_record(in));
        CHECK(out.kind == vgi::ConstraintKind::PrimaryKey);
        CHECK(out.columns == std::vector<vgi::ColumnPath>{{"a"}, {"b"}});
        CHECK(out.enforcement == vgi::ConstraintEnforcement::MetadataOnly);
        CHECK(out.validation == vgi::ConstraintValidation::Unvalidated);
        CHECK_FALSE(out.declared_name);
    }
    SECTION("CHECK carries its SQL") {
        auto in = spec("t.check", vgi::ConstraintKind::Check, {});
        in.check_sql = "budget >= 0";
        auto out = vgi::read_constraint_record(vgi::constraint_record(in));
        CHECK(out.kind == vgi::ConstraintKind::Check);
        CHECK(out.columns.empty());
        CHECK(out.check_sql == "budget >= 0");
    }
    SECTION("FOREIGN_KEY carries its referenced key") {
        auto in = spec("t.fk", vgi::ConstraintKind::ForeignKey, {{"dept_id"}});
        in.references = vgi::ConstraintReference{{"main"}, "dept", {{"id"}}, "dept.pk"};
        auto out = vgi::read_constraint_record(vgi::constraint_record(in));
        CHECK(out.kind == vgi::ConstraintKind::ForeignKey);
        REQUIRE(out.references);
        CHECK(out.references->schema_path == vgi::SchemaPath{"main"});
        CHECK(out.references->table == "dept");
        CHECK(out.references->columns == std::vector<vgi::ColumnPath>{{"id"}});
        CHECK(out.references->constraint_id == "dept.pk");
    }
    SECTION("garbage is refused") {
        CHECK_THROWS_AS(vgi::read_constraint_record("not ipc"), std::invalid_argument);
    }
}

TEST_CASE("constraint_record refuses an inconsistent spec", "[constraints]") {
    auto unstated = spec("t.nn", vgi::ConstraintKind::NotNull, {{"id"}});
    unstated.enforcement.reset();
    CHECK_THROWS_WITH(vgi::constraint_record(unstated),
                      ContainsSubstring("enforcement and validation must be stated explicitly"));
    unstated = spec("t.nn", vgi::ConstraintKind::NotNull, {{"id"}});
    unstated.validation.reset();
    CHECK_THROWS_AS(vgi::constraint_record(unstated), std::invalid_argument);

    CHECK_THROWS_WITH(record("", vgi::ConstraintKind::NotNull, {{"id"}}),
                      ContainsSubstring("constraint id"));
    CHECK_THROWS_WITH(record("t.nn", vgi::ConstraintKind::NotNull, {{"a"}, {"b"}}),
                      ContainsSubstring("exactly one column path"));
    CHECK_THROWS_WITH(record("t.pk", vgi::ConstraintKind::PrimaryKey, {}),
                      ContainsSubstring("PRIMARY_KEY requires column paths"));
    CHECK_THROWS_WITH(record("t.u", vgi::ConstraintKind::Unique, {{"a"}, {"a"}}),
                      ContainsSubstring("duplicate constraint column path"));
    CHECK_THROWS_WITH(record("t.u", vgi::ConstraintKind::Unique, {{}}),
                      ContainsSubstring("path must be nonempty"));
    CHECK_THROWS_WITH(record("t.c", vgi::ConstraintKind::Check, {}),
                      ContainsSubstring("check_sql is required for CHECK"));
    auto stray_check = spec("t.u", vgi::ConstraintKind::Unique, {{"a"}});
    stray_check.check_sql = "a > 0";
    CHECK_THROWS_AS(vgi::constraint_record(stray_check), std::invalid_argument);
    CHECK_THROWS_WITH(record("t.fk", vgi::ConstraintKind::ForeignKey, {{"a"}}),
                      ContainsSubstring("references is required for FOREIGN_KEY"));
    auto arity = spec("t.fk", vgi::ConstraintKind::ForeignKey, {{"a"}});
    arity.references = vgi::ConstraintReference{{"main"}, "d", {{"x"}, {"y"}}, "d.pk"};
    CHECK_THROWS_WITH(vgi::constraint_record(arity), ContainsSubstring("arity must match"));
}

TEST_CASE("only ENFORCED + ALL_ROWS records are served", "[constraints]") {
    auto columns = arrow::schema({required("id", arrow::int64())});
    for (auto [enforcement, validation] :
         {std::pair{vgi::ConstraintEnforcement::MetadataOnly, vgi::ConstraintValidation::AllRows},
          std::pair{vgi::ConstraintEnforcement::Enforced, vgi::ConstraintValidation::FutureWrites},
          std::pair{vgi::ConstraintEnforcement::Enforced,
                    vgi::ConstraintValidation::Unvalidated}}) {
        auto weak = spec("t.nn.id", vgi::ConstraintKind::NotNull, {{"id"}});
        weak.enforcement = enforcement;
        weak.validation = validation;
        CHECK_THROWS_WITH(
            vgi::validate_constraint_records("main.t", *columns, {vgi::constraint_record(weak)}),
            ContainsSubstring("Table 'main.t': constraint 't.nn.id' is ") &&
                ContainsSubstring("DuckDB accepts only ENFORCED + ALL_ROWS records"));
    }
    CHECK_NOTHROW(vgi::validate_constraint_records(
        "main.t", *columns, {record("t.nn.id", vgi::ConstraintKind::NotNull, {{"id"}})}));
}

namespace {
// The nullability of `path` in `schema`, descending through STRUCT children.
bool nullable_at(const arrow::Schema& schema, const vgi::ColumnPath& path) {
    auto field = schema.GetFieldByName(path[0]);
    for (size_t i = 1; i < path.size(); ++i)
        field = std::static_pointer_cast<arrow::StructType>(field->type())->GetFieldByName(path[i]);
    return field->nullable();
}
}  // namespace

TEST_CASE("nullability is served to match the records, never refused", "[constraints]") {
    SECTION("an unnamed non-nullable field is served nullable") {
        auto columns = arrow::schema({required("id", arrow::int64())});
        CHECK_NOTHROW(vgi::validate_constraint_records("main.t", *columns, {}));
        CHECK(nullable_at(*vgi::served_columns(*columns, {}), {"id"}));
    }
    SECTION("a record-named nullable field is served non-nullable") {
        auto pk = record("t.pk", vgi::ConstraintKind::PrimaryKey, {{"a"}, {"b"}});
        auto columns =
            arrow::schema({required("a", arrow::int64()), arrow::field("b", arrow::utf8())});
        CHECK_NOTHROW(vgi::validate_constraint_records("main.t", *columns, {pk}));
        auto served = vgi::served_columns(*columns, {pk});
        CHECK_FALSE(nullable_at(*served, {"a"}));
        CHECK_FALSE(nullable_at(*served, {"b"}));
    }
    SECTION("UNIQUE alone does not make a field non-nullable") {
        auto columns = arrow::schema({required("a", arrow::int64())});
        CHECK(nullable_at(
            *vgi::served_columns(*columns, {record("t.u", vgi::ConstraintKind::Unique, {{"a"}})}),
            {"a"}));
    }
    SECTION("ordinary STRUCTs are descended into, nested children included") {
        auto bbox = arrow::struct_(
            {required("xmin", arrow::float64()), required("xmax", arrow::float64())});
        auto columns = arrow::schema({arrow::field("bbox", bbox, false)});
        auto nested = record("t.nn.xmin", vgi::ConstraintKind::NotNull, {{"bbox", "xmin"}});
        auto served = vgi::served_columns(*columns, {nested});
        CHECK(nullable_at(*served, {"bbox"}));
        CHECK_FALSE(nullable_at(*served, {"bbox", "xmin"}));
        CHECK(nullable_at(*served, {"bbox", "xmax"}));
        CHECK_NOTHROW(vgi::validate_constraint_records("main.t", *columns, {nested}));
    }
    SECTION("a logical type's storage struct is left as it is") {
        auto storage = arrow::struct_({required("x", arrow::float64())});
        for (const auto& key : {"ARROW:extension:name", "vgi.logical_type.name"}) {
            INFO(key);
            auto field =
                arrow::field("point", storage, true, arrow::key_value_metadata({key}, {"p"}));
            auto served = vgi::served_columns(*arrow::schema({field}), {});
            CHECK_FALSE(nullable_at(*served, {"point", "x"}));
            CHECK(served->field(0)->metadata()->Contains(key));
            CHECK_THROWS_WITH(
                vgi::validate_constraint_records(
                    "main.t", *arrow::schema({field}),
                    {record("t.nn.px", vgi::ConstraintKind::NotNull, {{"point", "x"}})}),
                ContainsSubstring("NOT_NULL column point.x is not a column"));
        }
    }
    SECTION("the row-id column is untouched and not constrainable") {
        auto rowid = arrow::field("rowid", arrow::int64(), false,
                                  arrow::key_value_metadata({"is_row_id"}, {"true"}));
        auto columns = arrow::schema({rowid, arrow::field("v", arrow::utf8())});
        CHECK_FALSE(vgi::served_columns(*columns, {})->field(0)->nullable());
        CHECK_THROWS_WITH(vgi::validate_constraint_records(
                              "main.t", *columns,
                              {record("t.nn.rowid", vgi::ConstraintKind::NotNull, {{"rowid"}})}),
                          ContainsSubstring("NOT_NULL column rowid is not a column"));
    }
    SECTION("a record naming a missing column is refused") {
        CHECK_THROWS_WITH(
            vgi::validate_constraint_records(
                "main.t", *arrow::schema({arrow::field("id", arrow::int64())}),
                {record("t.nn.x", vgi::ConstraintKind::NotNull, {{"x"}})}),
            ContainsSubstring("Table 'main.t': constraint 't.nn.x' NOT_NULL column x"));
    }
}

TEST_CASE("records are held to the extension's ATTACH rules", "[constraints]") {
    const auto columns =
        arrow::schema({arrow::field("id", arrow::int64()), arrow::field("budget", arrow::float64()),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("s", arrow::struct_({arrow::field("x", arrow::int32())}))});
    auto refused = [&](const std::vector<std::string>& records) -> std::string {
        try {
            vgi::validate_constraint_records({"main"}, "t", *columns, records);
        } catch (const std::invalid_argument& error) {
            return error.what();
        }
        return "<accepted>";
    };
    auto check = [](const std::string& id, const std::string& sql) {
        auto in = spec(id, vgi::ConstraintKind::Check, {});
        in.check_sql = sql;
        return vgi::constraint_record(in);
    };
    SECTION("records owned by another table or schema") {
        auto other = spec("t.nn", vgi::ConstraintKind::NotNull, {{"id"}});
        other.table = "u";
        CHECK_THAT(refused({vgi::constraint_record(other)}),
                   ContainsSubstring("Table 'main.t': constraint 't.nn' belongs to main.u"));
        other = spec("t.nn", vgi::ConstraintKind::NotNull, {{"id"}});
        other.schema_path = {"data"};
        CHECK_THAT(refused({vgi::constraint_record(other)}),
                   ContainsSubstring("belongs to data.t"));
        other.schema_path = {"a", "b"};
        CHECK_THAT(refused({vgi::constraint_record(other)}),
                   ContainsSubstring("one-component schema path"));
    }
    SECTION("duplicate ids and declared names, and a second primary key") {
        CHECK_THAT(refused({record("k", vgi::ConstraintKind::NotNull, {{"id"}}),
                            record("k", vgi::ConstraintKind::Unique, {{"name"}})}),
                   ContainsSubstring("constraint 'k' duplicates another constraint id"));
        auto first = spec("a", vgi::ConstraintKind::Unique, {{"id"}});
        first.declared_name = "Key";
        auto second = spec("b", vgi::ConstraintKind::Unique, {{"name"}});
        second.declared_name = "KEY";
        CHECK_THAT(refused({vgi::constraint_record(first), vgi::constraint_record(second)}),
                   ContainsSubstring("reuses declared name 'KEY'"));
        CHECK_THAT(refused({record("p1", vgi::ConstraintKind::PrimaryKey, {{"id"}}),
                            record("p2", vgi::ConstraintKind::PrimaryKey, {{"name"}})}),
                   ContainsSubstring("constraint 'p2' is a second PRIMARY_KEY"));
    }
    SECTION("key columns must be top-level columns") {
        CHECK_THAT(refused({record("u", vgi::ConstraintKind::Unique, {{"s", "x"}})}),
                   ContainsSubstring("UNIQUE column s.x is nested"));
        CHECK_THAT(refused({record("u", vgi::ConstraintKind::Unique, {{"missing"}})}),
                   ContainsSubstring("UNIQUE column missing is not a column"));
        CHECK(refused({record("nn", vgi::ConstraintKind::NotNull, {{"s", "x"}})}) == "<accepted>");
    }
    SECTION("foreign keys stay in the schema") {
        auto fk = spec("fk", vgi::ConstraintKind::ForeignKey, {{"id"}});
        fk.references = vgi::ConstraintReference{{"other"}, "d", {{"id"}}, "d.pk"};
        CHECK_THAT(refused({vgi::constraint_record(fk)}),
                   ContainsSubstring("identified key of a table in the same schema"));
        fk.references = vgi::ConstraintReference{{"main"}, "d", {{"id"}}, "d.pk"};
        CHECK(refused({vgi::constraint_record(fk)}) == "<accepted>");
    }
    SECTION("CHECK must be a numeric comparison the extension parses") {
        for (const auto* sql : {"budget >= 0", "id <> -1", "(0 < budget)", "ID = 2.5", "id != 1e3",
                                "\"budget\" <= 10"}) {
            INFO(sql);
            CHECK(refused({check("c", sql)}) == "<accepted>");
        }
        for (const auto& [sql, why] : std::vector<std::pair<std::string, std::string>>{
                 {"budget + 1 > 0", "numeric column/constant comparisons"},
                 {"budget > 0 AND id > 0", "Boolean comparison"},
                 {"name > 0", "primitive signed numeric columns"},
                 {"budget > 'a'", "numeric operands"},
                 {"budget > NULL", "numeric operands"},
                 {"missing > 0", "missing or synthetic column"},
                 {"budget BETWEEN 0 AND 1", "unexpected"},
                 {"main.t.budget > 0", "unsupported character"},
                 {"budget > 0 x", "unexpected"}}) {
            INFO(sql);
            CHECK_THAT(
                refused({check("c", sql)}),
                ContainsSubstring("constraint 'c' CHECK (" + sql + ")") && ContainsSubstring(why));
        }
    }
}

TEST_CASE("RequiredFilterPath cannot be built from the old flat spelling", "[required-filters]") {
    // The old `{{"ticker", "cik"}}` against vector<vector<vector<string>>>
    // compiled through vector's iterator-pair constructor. Neither a string
    // literal nor a char converts to a path now, so that spelling -- and any
    // flat one -- is a compile error; the type traits below are its probe.
    static_assert(!std::is_convertible_v<const char*, vgi::RequiredFilterPath>);
    static_assert(!std::is_constructible_v<vgi::RequiredFilterPath, char>);
    static_assert(!std::is_constructible_v<vgi::RequiredFilterPath, const char*>);
    static_assert(!std::is_convertible_v<std::vector<std::string>, vgi::RequiredFilterPath>);
    std::vector<vgi::RequiredFilterGroup> groups = {{{"ticker"}, {"cik"}}, {{"s", "x"}}};
    REQUIRE(groups.size() == 2);
    CHECK(groups[0].size() == 2);
    CHECK(groups[1][0].components == std::vector<std::string>{"s", "x"});
}

TEST_CASE("a catalog with records offers the constraint read profiles itself",
          "[constraints][attach]") {
    vgi::Dispatcher dispatcher;
    dispatcher.set_catalog(
        constrained_model(arrow::schema({required("id", arrow::int64())}),
                          {record("t.pk", vgi::ConstraintKind::PrimaryKey, {{"id"}})}));
    Served worker(dispatcher);
    const std::vector<std::string> both{vgi::CONSTRAINT_RECORDS_READ, vgi::SCALAR_EXPRESSION_READ};

    SECTION("selected when the client offers them") {
        auto result = worker.call("catalog_attach", attach_params(both));
        CHECK(read_profiles(result, "catalog_capabilities") == both);
        CHECK(read_profiles(result, "selected_capabilities") == both);
    }
    SECTION("records alone when the client offers only records") {
        auto result =
            worker.call("catalog_attach",
                        attach_params(std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ}));
        CHECK(read_profiles(result, "selected_capabilities") ==
              std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ});
    }
    SECTION("the scalar profile never outlives records") {
        auto result = worker.call(
            "catalog_attach", attach_params(std::vector<std::string>{vgi::SCALAR_EXPRESSION_READ}));
        CHECK(read_profiles(result, "catalog_capabilities") == both);
        CHECK(read_profiles(result, "selected_capabilities").empty());
    }
    SECTION("offered but not selected when the client offers nothing") {
        auto result = worker.call("catalog_attach", attach_params(std::nullopt));
        CHECK(read_profiles(result, "catalog_capabilities") == both);
        CHECK(read_profiles(result, "selected_capabilities").empty());
    }
}

TEST_CASE("explicit profiles merge with the automatic ones", "[constraints][attach]") {
    auto model = constrained_model(arrow::schema({required("id", arrow::int64())}),
                                   {record("t.pk", vgi::ConstraintKind::PrimaryKey, {{"id"}})});
    model.catalog_read_profiles = {"vgi.index.read.v1", vgi::CONSTRAINT_RECORDS_READ};
    vgi::Dispatcher dispatcher;
    dispatcher.set_catalog(std::move(model));
    Served worker(dispatcher);
    auto result = worker.call(
        "catalog_attach",
        attach_params(std::vector<std::string>{"vgi.index.read.v1", vgi::CONSTRAINT_RECORDS_READ}));
    CHECK(read_profiles(result, "catalog_capabilities") ==
          std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ, vgi::SCALAR_EXPRESSION_READ,
                                   "vgi.index.read.v1"});
    CHECK(read_profiles(result, "selected_capabilities") ==
          std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ, "vgi.index.read.v1"});
}

TEST_CASE("a catalog without records keeps negotiation absent", "[constraints][attach]") {
    vgi::Dispatcher dispatcher;
    dispatcher.set_catalog(
        constrained_model(arrow::schema({arrow::field("id", arrow::int64())}), {}));
    Served worker(dispatcher);
    auto result = worker.call(
        "catalog_attach", attach_params(std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ}));
    CHECK(read_profiles(result, "catalog_capabilities") == std::vector<std::string>{"<absent>"});
    CHECK(read_profiles(result, "selected_capabilities") == std::vector<std::string>{"<absent>"});
}

TEST_CASE("a declarative catalog is checked when the worker starts", "[constraints][serve]") {
    auto starts = [](vgi::CatalogModel model) -> std::string {
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(std::move(model));
        try {
            dispatcher.validate_catalogs();
        } catch (const std::invalid_argument& error) {
            return error.what();
        }
        return "<accepted>";
    };
    SECTION("a valid table starts, and serves its key column non-nullable") {
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(constrained_model(
            arrow::schema({arrow::field("id", arrow::int64()), required("other", arrow::int64())}),
            {record("t.pk", vgi::ConstraintKind::PrimaryKey, {{"id"}})}));
        Served worker(dispatcher);
        auto attached =
            worker.call("catalog_attach",
                        attach_params(std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ}));
        auto params = vgi::wire::ResultBuilder(gen::CatalogTableGetParamsSchema())
                          .set_binary("attach_opaque_data",
                                      vgi::wire::get_binary(attached, "attach_opaque_data"))
                          .set_string_list("schema_path", {"main"})
                          .set_string("name", "t")
                          .fill_defaults()
                          .finish();
        auto items = vgi::wire::get_binary_list(worker.call("catalog_table_get", params), "items");
        REQUIRE(items.size() == 1);
        auto served = vgi::wire::decode_schema(
            vgi::wire::get_binary(vgi::wire::decode_ipc(items[0]), "columns"));
        CHECK_FALSE(nullable_at(*served, {"id"}));
        CHECK(nullable_at(*served, {"other"}));
    }
    SECTION("a METADATA_ONLY record fails startup") {
        auto weak = spec("t.pk", vgi::ConstraintKind::PrimaryKey, {{"id"}});
        weak.enforcement = vgi::ConstraintEnforcement::MetadataOnly;
        CHECK_THAT(starts(constrained_model(arrow::schema({required("id", arrow::int64())}),
                                            {vgi::constraint_record(weak)})),
                   ContainsSubstring("catalog 'c': Table 'main.t': constraint 't.pk'") &&
                       ContainsSubstring("ENFORCED + ALL_ROWS"));
    }
    SECTION("legacy descriptor fields name the replacement API") {
        auto model = constrained_model(arrow::schema({arrow::field("id", arrow::int64())}), {});
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        model.schema("main").tables.front().not_null = {0};
#pragma GCC diagnostic pop
        CHECK_THAT(starts(std::move(model)), ContainsSubstring("vgi::constraint_record()") &&
                                                 ContainsSubstring("Table 'main.t'"));
    }
    SECTION("an empty definition_revision is refused") {
        auto model = constrained_model(arrow::schema({arrow::field("id", arrow::int64())}), {});
        model.schema("main").tables.front().definition_revision = "";
        CHECK_THAT(starts(std::move(model)),
                   ContainsSubstring("definition_revision must be nonempty"));
    }
    SECTION("a view whose security is not regular is refused") {
        vgi::CatalogModel model;
        model.name = "c";
        vgi::CatalogView view;
        view.name = "v";
        view.definition = "SELECT 1";
        view.security = "barrier";
        model.schema("main").views.push_back(view);
        CHECK_THAT(starts(std::move(model)),
                   ContainsSubstring("View 'main.v': security 'barrier'"));
    }
}

namespace {
vgi::CatalogTable fk_table(const std::string& name, std::vector<std::string> records) {
    vgi::CatalogTable table;
    table.name = name;
    table.scan_function = name + "_scan";
    table.columns = arrow::schema({arrow::field("id", arrow::int64())});
    table.constraints = std::move(records);
    return table;
}
std::string owned(const std::string& table, const std::string& id, vgi::ConstraintKind kind,
                  std::optional<vgi::ConstraintReference> references = std::nullopt) {
    auto in = spec(id, kind, {{"id"}});
    in.table = table;
    in.references = std::move(references);
    return vgi::constraint_record(in);
}
}  // namespace

TEST_CASE("a foreign key must name a key the catalog declares", "[constraints][serve]") {
    auto starts = [](std::vector<vgi::CatalogTable> tables) -> std::string {
        vgi::Dispatcher dispatcher;
        dispatcher.catalog().name = "c";
        for (auto& table : tables) dispatcher.catalog().schema("main").tables.push_back(table);
        try {
            dispatcher.validate_catalogs();
        } catch (const std::invalid_argument& error) {
            return error.what();
        }
        return "<accepted>";
    };
    const auto pk = owned("d", "d.pk", vgi::ConstraintKind::PrimaryKey);
    auto fk = [](const std::string& key, std::vector<vgi::ColumnPath> columns = {{"id"}}) {
        return owned("e", "e.fk", vgi::ConstraintKind::ForeignKey,
                     vgi::ConstraintReference{{"main"}, "d", std::move(columns), key});
    };
    CHECK(starts({fk_table("d", {pk}), fk_table("e", {fk("d.pk")})}) == "<accepted>");
    CHECK_THAT(starts({fk_table("e", {fk("d.pk")})}),
               ContainsSubstring("constraint 'e.fk' references table main.d, which this catalog "
                                 "does not declare"));
    CHECK_THAT(starts({fk_table("d", {pk}), fk_table("e", {fk("d.other")})}),
               ContainsSubstring("references key 'd.other', which is not a PRIMARY_KEY or UNIQUE"));
}

TEST_CASE("time-travel versions serve the table's records unless they state their own",
          "[constraints][time-travel]") {
    auto versioned = [](std::optional<std::vector<std::string>> v1_records,
                        std::shared_ptr<arrow::Schema> v1_columns) {
        auto model = constrained_model(arrow::schema({arrow::field("id", arrow::int64())}),
                                       {record("t.pk", vgi::ConstraintKind::PrimaryKey, {{"id"}})});
        model.schema("main").tables.front().definition_revision = "rev-table";
        vgi::TimeTravelVersion v1;
        v1.version = 1;
        v1.columns = std::move(v1_columns);
        v1.scan_function = "t_scan_v1";
        v1.constraints = std::move(v1_records);
        vgi::TimeTravelVersion v2 = v1;
        v2.version = 2;
        v2.columns = arrow::schema({arrow::field("id", arrow::int64())});
        v2.constraints.reset();
        model.schema("main").tables.front().time_travel = {v1, v2};
        return model;
    };
    SECTION("an unstated version inherits the table's records and revision") {
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(
            versioned(std::nullopt, arrow::schema({arrow::field("id", arrow::int64())})));
        CHECK_NOTHROW(dispatcher.validate_catalogs());
        const auto& table = dispatcher.catalog().find_schema("main")->tables.front();
        auto info = vgi::wire::decode_ipc(
            vgi::Dispatcher::encode_table_info(table, {"main"}, &table.time_travel[0]));
        CHECK(vgi::wire::get_binary_list(info, "constraints").size() == 1);
        CHECK(vgi::wire::get_optional_binary(info, "definition_revision") == "rev-table");
        CHECK_FALSE(
            nullable_at(*vgi::wire::decode_schema(vgi::wire::get_binary(info, "columns")), {"id"}));
    }
    SECTION("an explicit empty list opts the version out") {
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(versioned(std::vector<std::string>{},
                                         arrow::schema({arrow::field("id", arrow::int64())})));
        const auto& table = dispatcher.catalog().find_schema("main")->tables.front();
        auto info = vgi::wire::decode_ipc(
            vgi::Dispatcher::encode_table_info(table, {"main"}, &table.time_travel[0]));
        CHECK(vgi::wire::get_binary_list(info, "constraints").empty());
        CHECK_FALSE(vgi::wire::get_optional_binary(info, "definition_revision"));
    }
    SECTION("inherited records that do not fit a version fail startup, naming it") {
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(
            versioned(std::nullopt, arrow::schema({arrow::field("key", arrow::int64())})));
        CHECK_THROWS_WITH(dispatcher.validate_catalogs(),
                          ContainsSubstring("Table 'main.t': constraint 't.pk' PRIMARY_KEY column "
                                            "id is not a column") &&
                              ContainsSubstring("(time-travel version 1)"));
    }
}

namespace {
std::shared_ptr<arrow::Field> generated(const std::string& name,
                                        std::shared_ptr<arrow::DataType> type,
                                        const std::string& sql, bool nullable = true) {
    return arrow::field(
        name, std::move(type), nullable,
        arrow::key_value_metadata({"vgi.column.generated.v1", "vgi.column.generation.v1"},
                                  {vgi::generated_expression_metadata(sql, {"main"}), "virtual"}));
}
}  // namespace

TEST_CASE("generated columns are checked when the worker starts", "[generated-columns]") {
    auto starts = [](arrow::FieldVector fields, std::vector<std::string> records = {}) {
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(constrained_model(arrow::schema(fields), std::move(records)));
        try {
            dispatcher.validate_catalogs();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what());
        }
        return std::string("<accepted>");
    };
    const auto n = arrow::field("n", arrow::int64());
    CHECK(starts({n, generated("d", arrow::int64(), "n * 2")}) == "<accepted>");
    CHECK(starts({n, generated("s", arrow::utf8(), "'item_' || CAST(n AS VARCHAR)")}) ==
          "<accepted>");
    CHECK(starts({n, generated("b", arrow::boolean(), "n IS NOT NULL AND NOT (n < 0)")}) ==
          "<accepted>");
    // Declared non-nullable but named by no record: served nullable, accepted.
    CHECK(starts({n, generated("d", arrow::int64(), "abs(n)", false)}) == "<accepted>");
    for (const auto& [field, why] :
         std::vector<std::pair<std::shared_ptr<arrow::Field>, std::string>>{
             {generated("d", arrow::float64(), "n * 2"), "signed integer, Boolean or utf8"},
             {generated("d", arrow::int64(), "n / 2"), "unsupported scalar function '/'"},
             {generated("d", arrow::int64(), "sqrt(n)"), "unsupported scalar function 'sqrt'"},
             {generated("d", arrow::int64(), "n * 2.5"), "unsupported scalar literal type"},
             {generated("d", arrow::int64(), "CAST(n AS DOUBLE)"), "unsupported scalar cast"},
             {generated("d", arrow::int64(), "missing + 1"), "missing or synthetic column"},
             {generated("d", arrow::int64(), "d + 1"), "depends on itself"},
             {generated("d", arrow::int64(), "CASE WHEN n > 0 THEN 1 END"), "unexpected"}}) {
        INFO(why);
        CHECK_THAT(starts({n, field}), ContainsSubstring("Table 'main.t': generated column 'd'") &&
                                           ContainsSubstring(why));
    }
    CHECK_THAT(starts({n, generated("d", arrow::int64(), "n * 2")},
                      {record("t.nn.d", vgi::ConstraintKind::NotNull, {{"d"}})}),
               ContainsSubstring("generated columns must be nullable"));
    CHECK_THAT(starts({generated("d", arrow::int64(), "1")}),
               ContainsSubstring("requires a physical column"));
}

TEST_CASE("legacy generated_expression metadata is translated", "[generated-columns]") {
    auto legacy = arrow::field(
        "d", arrow::int64(), true,
        arrow::key_value_metadata({"generated_expression", "comment"}, {"n * 2", "doubled"}));
    auto model = constrained_model(arrow::schema({arrow::field("n", arrow::int64()), legacy}), {});
    vgi::Dispatcher dispatcher;
    dispatcher.set_catalog(std::move(model));
    REQUIRE_NOTHROW(dispatcher.validate_catalogs());
    const auto& table = dispatcher.catalog().find_schema("main")->tables.front();
    auto served = vgi::wire::decode_schema(vgi::wire::get_binary(
        vgi::wire::decode_ipc(vgi::Dispatcher::encode_table_info(table, {"main"})), "columns"));
    const auto& metadata = *served->field(1)->metadata();
    CHECK_FALSE(metadata.Contains("generated_expression"));
    CHECK(metadata.Get("comment").ValueOrDie() == "doubled");
    CHECK(metadata.Get("vgi.column.generation.v1").ValueOrDie() == "virtual");
    CHECK(metadata.Get("vgi.column.generated.v1").ValueOrDie() ==
          vgi::generated_expression_metadata("n * 2", {"main"}));

    // Beside canonical metadata the intent is ambiguous: refused, not guessed.
    auto both = arrow::field(
        "d", arrow::int64(), true,
        arrow::key_value_metadata(
            {"generated_expression", "vgi.column.generated.v1", "vgi.column.generation.v1"},
            {"n * 2", vgi::generated_expression_metadata("n * 3", {"main"}), "virtual"}));
    vgi::Dispatcher ambiguous;
    ambiguous.set_catalog(
        constrained_model(arrow::schema({arrow::field("n", arrow::int64()), both}), {}));
    CHECK_THROWS_WITH(ambiguous.validate_catalogs(),
                      ContainsSubstring("Table 'main.t': column 'd': legacy generated_expression "
                                        "beside canonical"));
}

TEST_CASE("required_filters paths are checked against the served columns", "[required-filters]") {
    auto starts = [](std::vector<vgi::RequiredFilterGroup> groups) {
        auto model = constrained_model(
            arrow::schema({arrow::field("a", arrow::int64()),
                           arrow::field("s", arrow::struct_({arrow::field("x", arrow::int32())}))}),
            {});
        model.schema("main").tables.front().required_filters = std::move(groups);
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(std::move(model));
        try {
            dispatcher.validate_catalogs();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what());
        }
        return std::string("<accepted>");
    };
    CHECK(starts({{{"a"}, {"s", "x"}}}) == "<accepted>");
    CHECK_THAT(starts({{{"b"}}}), ContainsSubstring("Table 'main.t': required_filters: path b "
                                                    "is not a column"));
    CHECK_THAT(starts({{{"a", "x"}}}), ContainsSubstring("path a is not a STRUCT"));
    CHECK_THAT(starts({{}}), ContainsSubstring("a group is empty"));
    CHECK_THAT(starts({{vgi::RequiredFilterPath(std::vector<std::string>{})}}),
               ContainsSubstring("a path is empty"));
    CHECK_THAT(starts({{{""}}}), ContainsSubstring("empty or NUL-bearing component"));
    CHECK_THAT(starts({{{"a"}, {"a"}}}), ContainsSubstring("appears twice in one group"));
}

TEST_CASE("macro_overload encodes one overload record", "[macros]") {
    vgi::MacroOverloadSpec in;
    in.definition = "a + b";
    arrow::Int64Builder value;
    REQUIRE(value.Append(7).ok());
    in.parameters = {{"a", arrow::int64(), nullptr, "the first"},
                     {"b", nullptr, value.Finish().ValueOrDie(), std::nullopt}};
    auto out = vgi::wire::decode_ipc(vgi::macro_overload(in));
    CHECK(vgi::wire::get_string(out, "definition") == "a + b");
    CHECK(vgi::wire::get_string(out, "macro_type") == "scalar");
    CHECK(vgi::wire::get_string(out, "profile") == "duckdb.macro.v1");
    auto parameters = std::static_pointer_cast<arrow::StructArray>(
        std::static_pointer_cast<arrow::ListArray>(vgi::wire::column(out, "parameters"))->values());
    REQUIRE(parameters->length() == 2);
    CHECK_FALSE(parameters->GetFieldByName("declared_type")->IsNull(0));
    CHECK(parameters->GetFieldByName("declared_type")->IsNull(1));
    CHECK(parameters->GetFieldByName("default_value")->IsNull(0));
    CHECK_FALSE(parameters->GetFieldByName("default_value")->IsNull(1));

    // A family built from it is served.
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "c";
    vgi::CatalogMacro macro;
    macro.name = "m";
    macro.overloads = std::vector<std::string>{vgi::macro_overload(in)};
    dispatcher.catalog().schema("main").macros.push_back(macro);
    CHECK_NOTHROW(dispatcher.validate_catalogs());

    auto bad = in;
    std::swap(bad.parameters[0], bad.parameters[1]);
    CHECK_THROWS_WITH(vgi::macro_overload(bad), ContainsSubstring("follows a default"));
    bad = in;
    bad.parameters[1].name = "A";
    CHECK_THROWS_WITH(vgi::macro_overload(bad), ContainsSubstring("duplicate parameter"));
    bad = in;
    bad.definition.clear();
    CHECK_THROWS_WITH(vgi::macro_overload(bad), ContainsSubstring("definition"));
}

TEST_CASE("catalog capability selection keeps what the extension accepts together",
          "[constraints][attach]") {
    const vgi::CatalogCapabilityProfiles all{
        {vgi::CONSTRAINT_RECORDS_READ, vgi::SCALAR_EXPRESSION_READ},
        {vgi::CONSTRAINT_CREATE},
        {vgi::CONSTRAINT_ENFORCE},
        {}};
    auto selected = vgi::select_catalog_capabilities(all, all);
    CHECK(selected.read == all.read);
    CHECK(selected.ddl == all.ddl);
    CHECK(selected.write == all.write);
    // Without records nothing that describes or acts on them survives.
    selected = vgi::select_catalog_capabilities(
        all,
        {{vgi::SCALAR_EXPRESSION_READ}, {vgi::CONSTRAINT_CREATE}, {vgi::CONSTRAINT_ENFORCE}, {}});
    CHECK(selected.empty());
    // CREATE needs enforcement.
    selected = vgi::select_catalog_capabilities(
        all, {{vgi::CONSTRAINT_RECORDS_READ}, {vgi::CONSTRAINT_CREATE}, {}, {}});
    CHECK(selected.read == std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ});
    CHECK(selected.ddl.empty());
    // Encoding round-trips, and malformed names are refused.
    auto decoded = vgi::decode_catalog_capabilities(vgi::encode_catalog_capabilities(all));
    CHECK(decoded.read == all.read);
    CHECK(decoded.write == all.write);
    CHECK_THROWS(vgi::encode_catalog_capabilities({{"Not A Profile"}, {}, {}, {}}));
}

namespace {
std::string offer_all() {
    auto catalog = vgi::encode_catalog_capabilities(
        {{vgi::CONSTRAINT_RECORDS_READ, vgi::SCALAR_EXPRESSION_READ},
         {vgi::CONSTRAINT_CREATE},
         {vgi::CONSTRAINT_ENFORCE},
         {}});
    return vgi::wire::encode_ipc(vgi::wire::ResultBuilder(gen::ClientCapabilitiesSchema())
                                     .set_string("engine", "test")
                                     .set_binary("catalog", catalog)
                                     .fill_defaults()
                                     .finish());
}

std::string not_null_definition(const std::string& column, const char* kind = "NOT_NULL") {
    return vgi::wire::encode_ipc(vgi::wire::ResultBuilder(gen::ConstraintDefinitionSchema())
                                     .set_enum("kind", kind)
                                     .set_string_list_list("column_paths", {{column}})
                                     .set_null("declared_name")
                                     .set_null("check_expression")
                                     .set_null("foreign_key")
                                     .finish());
}
}  // namespace

TEST_CASE("an in-memory catalog supports CREATE TABLE ... NOT NULL", "[constraints][memory]") {
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "c";
    vgi::MemoryCatalogOptions options;
    options.name = "mem";
    dispatcher.register_memory_catalog(options);
    Served worker(dispatcher);

    auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                       .set_string("name", "mem")
                       .set_binary("client_capabilities", offer_all())
                       .fill_defaults()
                       .finish();
    auto attached =
        worker.call("catalog_attach", vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                                          .set_binary("request", vgi::wire::encode_ipc(request))
                                          .finish());
    auto selected = vgi::decode_catalog_capabilities(
        *vgi::wire::get_optional_binary(attached, "selected_capabilities"));
    CHECK(selected.read == std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ});
    CHECK(selected.ddl == std::vector<std::string>{vgi::CONSTRAINT_CREATE});
    CHECK(selected.write == std::vector<std::string>{vgi::CONSTRAINT_ENFORCE});
    const auto handle = vgi::wire::get_binary(attached, "attach_opaque_data");

    auto create = [&](const std::string& name, std::shared_ptr<arrow::Schema> columns,
                      std::vector<std::string> definitions) {
        auto table = vgi::wire::ResultBuilder(gen::TableCreateRequestSchema())
                         .set_binary("attach_opaque_data", handle)
                         .set_string_list("schema_path", {"main"})
                         .set_string("name", name)
                         .set_binary("columns", vgi::wire::encode_schema(columns))
                         .set_enum("on_conflict", "ERROR")
                         .set_binary_list("constraint_definitions", definitions)
                         .fill_defaults()
                         .finish();
        return worker.refused("catalog_table_create",
                              vgi::wire::ResultBuilder(gen::CatalogTableCreateParamsSchema())
                                  .set_binary("request", vgi::wire::encode_ipc(table))
                                  .finish());
    };
    auto served = [&](const std::string& name) {
        auto params = vgi::wire::ResultBuilder(gen::CatalogTableGetParamsSchema())
                          .set_binary("attach_opaque_data", handle)
                          .set_string_list("schema_path", {"main"})
                          .set_string("name", name)
                          .fill_defaults()
                          .finish();
        auto items = vgi::wire::get_binary_list(worker.call("catalog_table_get", params), "items");
        REQUIRE(items.size() == 1);
        return vgi::wire::decode_ipc(items[0]);
    };

    // The extension sends a NOT NULL column non-nullable with its definition.
    REQUIRE(
        create("t",
               arrow::schema({required("id", arrow::int64()), arrow::field("v", arrow::utf8())}),
               {not_null_definition("id")}) == "<accepted>");
    auto info = served("t");
    auto records = vgi::wire::get_binary_list(info, "constraints");
    REQUIRE(records.size() == 1);
    auto view = vgi::read_constraint_record(records[0]);
    CHECK(view.kind == vgi::ConstraintKind::NotNull);
    CHECK(view.columns == std::vector<vgi::ColumnPath>{{"id"}});
    CHECK(view.table == "t");
    CHECK(view.enforcement == vgi::ConstraintEnforcement::Enforced);
    CHECK(view.validation == vgi::ConstraintValidation::AllRows);
    CHECK(vgi::wire::get_optional_binary(info, "definition_revision"));
    auto columns = vgi::wire::decode_schema(vgi::wire::get_binary(info, "columns"));
    CHECK_FALSE(nullable_at(*columns, {"id"}));
    CHECK(nullable_at(*columns, {"v"}));

    // A non-nullable column no definition names is served nullable.
    REQUIRE(create("relaxed", arrow::schema({required("x", arrow::int32())}), {}) == "<accepted>");
    CHECK(nullable_at(
        *vgi::wire::decode_schema(vgi::wire::get_binary(served("relaxed"), "columns")), {"x"}));

    // What the catalog cannot enforce, it refuses by name.
    CHECK_THAT(create("pk", arrow::schema({required("id", arrow::int64())}),
                      {not_null_definition("id", "PRIMARY_KEY")}),
               ContainsSubstring("supports only NOT NULL constraints; PRIMARY_KEY"));
    CHECK_THAT(create("u",
                      arrow::schema({arrow::field(
                          "uuid", arrow::fixed_size_binary(16), true,
                          arrow::key_value_metadata({"ARROW:extension:name"}, {"arrow.uuid"}))}),
                      {}),
               ContainsSubstring("cannot store logical-type column uuid"));
}
