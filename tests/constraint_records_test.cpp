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

#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <vgi_rpc/client.h>
#include <vgi_rpc/errors.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/wire.h>

#include "dispatcher.h"
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

TEST_CASE("Arrow nullability must equal the NOT_NULL/PRIMARY_KEY paths", "[constraints]") {
    SECTION("a table without records keeps every field nullable") {
        CHECK_NOTHROW(vgi::validate_constraint_records(
            "main.t", *arrow::schema({arrow::field("id", arrow::int64())}), {}));
        CHECK_THROWS_WITH(
            vgi::validate_constraint_records(
                "main.t", *arrow::schema({required("id", arrow::int64())}), {}),
            ContainsSubstring("Table 'main.t': column id is non-nullable in Arrow but no "
                              "NOT_NULL/PRIMARY_KEY record names it"));
    }
    SECTION("a constrained column must be non-nullable") {
        auto pk = record("t.pk", vgi::ConstraintKind::PrimaryKey, {{"a"}, {"b"}});
        CHECK_NOTHROW(vgi::validate_constraint_records(
            "main.t", *arrow::schema({required("a", arrow::int64()), required("b", arrow::utf8())}),
            {pk}));
        CHECK_THROWS_WITH(
            vgi::validate_constraint_records(
                "main.t",
                *arrow::schema({required("a", arrow::int64()), arrow::field("b", arrow::utf8())}),
                {pk}),
            ContainsSubstring("Table 'main.t': column b is in a NOT_NULL/PRIMARY_KEY record but "
                              "its Arrow field is nullable"));
    }
    SECTION("UNIQUE alone does not make a field non-nullable") {
        CHECK_THROWS_WITH(
            vgi::validate_constraint_records(
                "main.t", *arrow::schema({required("a", arrow::int64())}),
                {record("t.u", vgi::ConstraintKind::Unique, {{"a"}})}),
            ContainsSubstring("column a is non-nullable"));
    }
    SECTION("ordinary STRUCTs are descended into") {
        auto bbox = arrow::struct_(
            {required("xmin", arrow::float64()), arrow::field("xmax", arrow::float64())});
        auto nested = record("t.nn.xmin", vgi::ConstraintKind::NotNull, {{"bbox", "xmin"}});
        CHECK_NOTHROW(vgi::validate_constraint_records(
            "main.t", *arrow::schema({arrow::field("bbox", bbox)}), {nested}));
        CHECK_THROWS_WITH(vgi::validate_constraint_records(
                              "main.t", *arrow::schema({arrow::field("bbox", bbox)}), {}),
                          ContainsSubstring("Table 'main.t': column bbox.xmin is non-nullable"));
        auto loose = arrow::struct_(
            {arrow::field("xmin", arrow::float64()), arrow::field("xmax", arrow::float64())});
        CHECK_THROWS_WITH(vgi::validate_constraint_records(
                              "main.t", *arrow::schema({arrow::field("bbox", loose)}), {nested}),
                          ContainsSubstring("column bbox.xmin is in a NOT_NULL/PRIMARY_KEY"));
    }
    SECTION("a logical type's storage struct is not descended into") {
        auto storage = arrow::struct_({required("x", arrow::float64())});
        for (const auto& key : {"ARROW:extension:name", "vgi.logical_type.name"}) {
            INFO(key);
            auto field = arrow::field("point", storage, true, arrow::key_value_metadata({key}, {"p"}));
            CHECK_NOTHROW(vgi::validate_constraint_records("main.t", *arrow::schema({field}), {}));
            CHECK_THROWS_WITH(
                vgi::validate_constraint_records(
                    "main.t", *arrow::schema({field}),
                    {record("t.nn.px", vgi::ConstraintKind::NotNull, {{"point", "x"}})}),
                ContainsSubstring("NOT_NULL/PRIMARY_KEY column point.x is not a column"));
        }
    }
    SECTION("the row-id column is exempt") {
        auto rowid = arrow::field("rowid", arrow::int64(), false,
                                  arrow::key_value_metadata({"is_row_id"}, {"true"}));
        CHECK_NOTHROW(vgi::validate_constraint_records(
            "main.t", *arrow::schema({rowid, arrow::field("v", arrow::utf8())}), {}));
    }
    SECTION("a record naming a missing column is refused") {
        CHECK_THROWS_WITH(vgi::validate_constraint_records(
                              "main.t", *arrow::schema({arrow::field("id", arrow::int64())}),
                              {record("t.nn.x", vgi::ConstraintKind::NotNull, {{"x"}})}),
                          ContainsSubstring("Table 'main.t': NOT_NULL/PRIMARY_KEY column x"));
    }
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
            worker.call("catalog_attach", attach_params(std::vector<std::string>{
                                              vgi::CONSTRAINT_RECORDS_READ}));
        CHECK(read_profiles(result, "selected_capabilities") ==
              std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ});
    }
    SECTION("the scalar profile never outlives records") {
        auto result =
            worker.call("catalog_attach", attach_params(std::vector<std::string>{
                                              vgi::SCALAR_EXPRESSION_READ}));
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
    dispatcher.set_catalog(constrained_model(arrow::schema({arrow::field("id", arrow::int64())}), {}));
    Served worker(dispatcher);
    auto result = worker.call("catalog_attach",
                              attach_params(std::vector<std::string>{vgi::CONSTRAINT_RECORDS_READ}));
    CHECK(read_profiles(result, "catalog_capabilities") == std::vector<std::string>{"<absent>"});
    CHECK(read_profiles(result, "selected_capabilities") == std::vector<std::string>{"<absent>"});
}

TEST_CASE("serving a table enforces the record contract", "[constraints][serve]") {
    auto contents = [](Served& worker) {
        auto attached = worker.call("catalog_attach", attach_params(std::nullopt));
        auto params =
            vgi::wire::ResultBuilder(gen::CatalogContentsParamsSchema())
                .set_binary("attach_opaque_data",
                            vgi::wire::get_binary(attached, "attach_opaque_data"))
                .set_null("if_none_match")
                .finish();
        return worker.refused("catalog_contents", params);
    };
    SECTION("a valid table is served") {
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(
            constrained_model(arrow::schema({required("id", arrow::int64())}),
                              {record("t.pk", vgi::ConstraintKind::PrimaryKey, {{"id"}})}));
        Served worker(dispatcher);
        CHECK(contents(worker) == "<accepted>");
    }
    SECTION("nullability disagreement names the table and column") {
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(constrained_model(
            arrow::schema({arrow::field("id", arrow::int64())}),
            {record("t.pk", vgi::ConstraintKind::PrimaryKey, {{"id"}})}));
        Served worker(dispatcher);
        CHECK_THAT(contents(worker),
                   ContainsSubstring("Table 'main.t': column id is in a NOT_NULL/PRIMARY_KEY"));
    }
    SECTION("a METADATA_ONLY record is refused") {
        auto weak = spec("t.pk", vgi::ConstraintKind::PrimaryKey, {{"id"}});
        weak.enforcement = vgi::ConstraintEnforcement::MetadataOnly;
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(constrained_model(arrow::schema({required("id", arrow::int64())}),
                                                 {vgi::constraint_record(weak)}));
        Served worker(dispatcher);
        CHECK_THAT(contents(worker), ContainsSubstring("ENFORCED + ALL_ROWS"));
    }
    SECTION("legacy descriptor fields name the replacement API") {
        auto model = constrained_model(arrow::schema({arrow::field("id", arrow::int64())}), {});
        model.schema("main").tables.front().not_null = {0};
        vgi::Dispatcher dispatcher;
        dispatcher.set_catalog(std::move(model));
        Served worker(dispatcher);
        CHECK_THAT(contents(worker), ContainsSubstring("vgi::constraint_record()") &&
                                         ContainsSubstring("Table 'main.t'"));
    }
}
