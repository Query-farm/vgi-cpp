// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// A function listing is built once, then served as built.
//
// `catalog_schema_contents_functions` depends on nothing but the registrations,
// the attachment's catalog, the schema and the type asked for, yet it used to
// derive and encode every function in the schema on each request: about 50 ms
// for the example worker's main-schema table listing, which the engine asks
// for on every function-set load. These tests hold that a listing is built
// once per (catalog, schema, type), and that serving it from then on changes
// nothing a caller can see — the same bytes, still scoped to the attachment's
// catalog and to the schema asked about.
//
// The worker is a real Dispatcher behind a real vgi-rpc server, served over a
// socket pair in this process so the test can see what building a listing asks
// of each function. Requests come from vgi-rpc's own client; nothing here
// builds a response.

#include <catch2/catch_test_macros.hpp>

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <vgi_rpc/client.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/wire.h>

#include "dispatcher.h"
#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"
#include "wire.h"

namespace {

namespace gen = ::vgi::generated;

const std::string kProtocol(gen::VGI_PROTOCOL_NAME);
const std::string kVersion(gen::VGI_PROTOCOL_VERSION);

// Counts how often a function is asked for its declaration. Encoding a
// FunctionInfo starts there, so a count that does not move across a listing
// means nothing in it was built again.
class Counted {
public:
    int declarations() const { return declarations_.load(); }

protected:
    vgi::FunctionMetadata declare(const std::string& description) const {
        declarations_.fetch_add(1);
        vgi::FunctionMetadata metadata;
        metadata.description = description;
        metadata.return_type = arrow::int64();
        return metadata;
    }

private:
    mutable std::atomic<int> declarations_{0};
};

class Scalar final : public vgi::ScalarFunction, public Counted {
public:
    Scalar(std::string name, std::string description)
        : name_(std::move(name)), description_(std::move(description)) {}
    std::string name() const override { return name_; }
    vgi::FunctionMetadata metadata() const override { return declare(description_); }
    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }
    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams&, const std::shared_ptr<arrow::RecordBatch>&) const override {
        return nullptr;
    }

private:
    std::string name_;
    std::string description_;
};

class Table final : public vgi::TableFunction, public Counted {
public:
    Table(std::string name, std::string description)
        : name_(std::move(name)), description_(std::move(description)) {}
    std::string name() const override { return name_; }
    vgi::FunctionMetadata metadata() const override { return declare(description_); }
    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64())});
    }
    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams&) const override {
        return nullptr;
    }

private:
    std::string name_;
    std::string description_;
};

class Aggregate final : public vgi::AggregateFunction, public Counted {
public:
    Aggregate(std::string name, std::string description)
        : name_(std::move(name)), description_(std::move(description)) {}
    std::string name() const override { return name_; }
    vgi::FunctionMetadata metadata() const override { return declare(description_); }
    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("result", arrow::int64())});
    }
    void update(std::map<int64_t, std::string>&, const arrow::Int64Array&,
                const std::vector<std::shared_ptr<arrow::Array>>&) const override {}
    std::string combine(const std::string& target, const std::string&) const override {
        return target;
    }
    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>&, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>&) const override {
        return nullptr;
    }

private:
    std::string name_;
    std::string description_;
};

// A Dispatcher served on its own thread over one end of a socket pair, with
// vgi-rpc's client on the other, for the lifetime of the object.
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
                // The client end closed mid-message; there is no one to tell.
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
        // The server reads EOF and its loop ends.
        ::shutdown(fds_[0], SHUT_RDWR);
        thread_.join();
        ::close(fds_[0]);
        ::close(fds_[1]);
    }

    Served(const Served&) = delete;
    Served& operator=(const Served&) = delete;

    // Call a void method.
    void call_void(const std::string& method, const std::shared_ptr<arrow::RecordBatch>& params) {
        (void)client_->call_unary(method, params);
    }

    // The payload a unary method answered: its `result` bytes, decoded.
    std::shared_ptr<arrow::RecordBatch> call(const std::string& method,
                                             const std::shared_ptr<arrow::RecordBatch>& params) {
        const auto response = client_->call_unary(method, params);
        return vgi::wire::decode_ipc(vgi::wire::get_binary(response.batch, "result"));
    }

private:
    std::unique_ptr<vgi_rpc::Server> server_;
    int fds_[2] = {-1, -1};
    std::thread thread_;
    std::optional<vgi_rpc::RpcClient> client_;
};

// ATTACH `catalog`, returning the sealed handle later calls carry.
std::string attach(Served& worker, const std::string& catalog) {
    const auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                             .set_string("name", catalog)
                             .fill_defaults()
                             .finish();
    const auto params = vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                            .set_binary("request", vgi::wire::encode_ipc(request))
                            .finish();
    return vgi::wire::get_binary(worker.call("catalog_attach", params), "attach_opaque_data");
}

// The items a listing answered, as encoded. `type` is the engine's filter
// (`SCALAR_FUNCTION`, …), or empty for no filter.
std::vector<std::string> list(Served& worker, const std::string& handle,
                              const vgi::SchemaPath& schema, const std::string& type) {
    const auto params = vgi::wire::ResultBuilder(gen::CatalogSchemaContentsFunctionsParamsSchema())
                            .set_binary("attach_opaque_data", handle)
                            .set_string_list("path", schema)
                            .set_enum("type", type)
                            .fill_defaults()
                            .finish();
    return vgi::wire::get_binary_list(worker.call("catalog_schema_contents_functions", params),
                                      "items");
}

// `name: description` for each item, which says whose declaration was listed.
std::vector<std::string> described(const std::vector<std::string>& items) {
    std::vector<std::string> out;
    for (const auto& item : items) {
        const auto info = vgi::wire::decode_ipc(item);
        out.push_back(vgi::wire::get_string(info, "name") + ": " +
                      vgi::wire::get_string(info, "description"));
    }
    return out;
}

std::vector<std::string> joined(std::vector<std::vector<std::string>> parts) {
    std::vector<std::string> out;
    for (auto& part : parts) out.insert(out.end(), part.begin(), part.end());
    return out;
}

using Names = std::vector<std::string>;

}  // namespace

TEST_CASE("a function listing is built once per catalog, schema and type", "[catalog][listing]") {
    // Two catalogs from one binary, declaring the same name in the same
    // schema, plus the same name again in a second schema of the first: a
    // listing shared too widely answers with someone else's declaration.
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "alpha";
    const auto alpha_scalar = std::make_shared<Scalar>("twice", "alpha main");
    const auto alpha_table = std::make_shared<Table>("rows", "alpha main");
    const auto alpha_aggregate = std::make_shared<Aggregate>("total", "alpha main");
    const auto alpha_side = std::make_shared<Scalar>("twice", "alpha side");
    const auto alpha_hidden = std::make_shared<Scalar>("backing", "alpha main");
    const auto beta_scalar = std::make_shared<Scalar>("twice", "beta main");
    dispatcher.register_scalar(alpha_scalar);
    dispatcher.register_table(alpha_table);
    dispatcher.register_aggregate(alpha_aggregate);
    dispatcher.register_scalar_in("alpha", "side", alpha_side);
    dispatcher.register_scalar(alpha_hidden);
    dispatcher.hide_function("backing");
    dispatcher.register_scalar_in("beta", "main", beta_scalar);
    const auto counts = [&] {
        return std::vector<int>{alpha_scalar->declarations(),    alpha_table->declarations(),
                                alpha_aggregate->declarations(), alpha_side->declarations(),
                                alpha_hidden->declarations(),    beta_scalar->declarations()};
    };

    Served worker(dispatcher);
    const auto alpha = attach(worker, "alpha");
    const auto beta = attach(worker, "beta");
    // A second ATTACH of the same catalog seals a different id: what a
    // listing does not depend on.
    const auto alpha_again = attach(worker, "alpha");
    REQUIRE(alpha_again != alpha);

    const auto scalars = list(worker, alpha, {"main"}, "SCALAR_FUNCTION");
    CHECK(described(scalars) == Names{"twice: alpha main"});
    CHECK(alpha_scalar->declarations() > 0);
    const auto built_once = counts();

    SECTION("asked again, it is the same bytes and nothing is built") {
        CHECK(list(worker, alpha, {"main"}, "SCALAR_FUNCTION") == scalars);
        CHECK(list(worker, alpha_again, {"main"}, "SCALAR_FUNCTION") == scalars);
        CHECK(counts() == built_once);
    }

    SECTION("each catalog, schema and type is its own listing") {
        const auto beta_scalars = list(worker, beta, {"main"}, "SCALAR_FUNCTION");
        CHECK(described(beta_scalars) == Names{"twice: beta main"});
        const auto side = list(worker, alpha, {"side"}, "SCALAR_FUNCTION");
        CHECK(described(side) == Names{"twice: alpha side"});
        const auto tables = list(worker, alpha, {"main"}, "TABLE_FUNCTION");
        CHECK(described(tables) == Names{"rows: alpha main"});
        const auto aggregates = list(worker, alpha, {"main"}, "AGGREGATE_FUNCTION");
        CHECK(described(aggregates) == Names{"total: alpha main"});
        CHECK(beta_scalar->declarations() > 0);
        CHECK(alpha_side->declarations() > 0);
        const auto all_built = counts();

        // Every one of them again, plus the unfiltered listing — the three
        // typed ones in order — and nothing is built a second time.
        CHECK(list(worker, beta, {"main"}, "SCALAR_FUNCTION") == beta_scalars);
        CHECK(list(worker, alpha, {"side"}, "SCALAR_FUNCTION") == side);
        CHECK(list(worker, alpha, {"main"}, "TABLE_FUNCTION") == tables);
        CHECK(list(worker, alpha_again, {"main"}, "AGGREGATE_FUNCTION") == aggregates);
        const auto unfiltered = list(worker, alpha, {"main"}, "");
        CHECK(unfiltered == joined({scalars, tables, aggregates}));
        CHECK(list(worker, alpha_again, {"main"}, "") == unfiltered);
        CHECK(counts() == all_built);

        // What declares nothing answers nothing.
        CHECK(list(worker, alpha, {"nowhere"}, "SCALAR_FUNCTION").empty());
        CHECK(list(worker, beta, {"main"}, "TABLE_FUNCTION").empty());
        CHECK(list(worker, beta, {"side"}, "SCALAR_FUNCTION").empty());
    }

    SECTION("an unfiltered listing shares the typed listings' encodings") {
        const auto unfiltered = list(worker, alpha, {"main"}, "");
        REQUIRE(described(unfiltered) ==
                Names{"twice: alpha main", "rows: alpha main", "total: alpha main"});
        const auto all_built = counts();
        CHECK(list(worker, alpha, {"main"}, "TABLE_FUNCTION") == Names{unfiltered[1]});
        CHECK(list(worker, alpha, {"main"}, "AGGREGATE_FUNCTION") == Names{unfiltered[2]});
        CHECK(counts() == all_built);
    }

    // A hidden function backs a table; it is registered, never listed, and
    // never needs building.
    CHECK(alpha_hidden->declarations() == 0);
}

TEST_CASE("a function registered after a listing is served is in the next one",
          "[catalog][listing]") {
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "alpha";
    dispatcher.register_scalar(std::make_shared<Scalar>("first", "alpha main"));
    {
        Served worker(dispatcher);
        const auto handle = attach(worker, "alpha");
        CHECK(described(list(worker, handle, {"main"}, "SCALAR_FUNCTION")) ==
              Names{"first: alpha main"});
    }

    dispatcher.register_scalar(std::make_shared<Scalar>("second", "alpha main"));
    dispatcher.hide_function("first");

    Served worker(dispatcher);
    const auto handle = attach(worker, "alpha");
    CHECK(described(list(worker, handle, {"main"}, "SCALAR_FUNCTION")) ==
          Names{"second: alpha main"});
}

namespace {

// A whole-input reduction that records what combine was handed.
class Reduction final : public vgi::TableBufferingFunction {
public:
    std::string name() const override { return "reduce"; }
    vgi::FunctionMetadata metadata() const override { return {}; }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input table")};
    }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64())});
    }
    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>&) override {
        return params.execution_id;
    }
    std::vector<std::string> combine(const vgi::ProcessParams& params,
                                     const std::vector<std::string>& state_ids) override {
        combined.push_back(state_ids);
        return {params.execution_id};
    }
    std::unique_ptr<vgi::TableProducer> finalize_producer(const vgi::ProcessParams&,
                                                          const std::string&) override {
        return nullptr;
    }

    std::vector<std::vector<std::string>> combined;
};

}  // namespace

TEST_CASE("combine accepts an empty state_ids list", "[buffering]") {
    // Since vgi 63eb257 the extension runs combine and finalize when a
    // table-buffering function's input is empty at runtime: no Sink thread
    // ever ran, so combine is sent an EMPTY state_ids list.  That has to reach
    // the function as an empty list -- not an error, and not skipped -- and
    // what it returns has to go back as the ids to finalize, so a reduction
    // such as sum_all_columns can answer with its zero row.
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "alpha";
    const auto reduction = std::make_shared<Reduction>();
    dispatcher.register_buffering(reduction);
    Served worker(dispatcher);

    const std::string execution_id = "exec-empty-input";
    const auto request = vgi::wire::ResultBuilder(gen::TableBufferingCombineRequestSchema())
                             .set_string("function_name", "reduce")
                             .set_binary("execution_id", execution_id)
                             .set_binary_list("state_ids", {})
                             .fill_defaults()
                             .finish();
    const auto params = vgi::wire::ResultBuilder(gen::TableBufferingCombineParamsSchema())
                            .set_binary("request", vgi::wire::encode_ipc(request))
                            .finish();
    const auto reply = worker.call("table_buffering_combine", params);

    REQUIRE(reduction->combined.size() == 1);
    CHECK(reduction->combined[0].empty());
    CHECK(vgi::wire::get_binary_list(reply, "finalize_state_ids") ==
          std::vector<std::string>{execution_id});
}

namespace {

// One schema's SchemaContents from a catalog_contents payload, by path.
struct ContentsRow {
    std::vector<std::string> tables, views, scalar_functions, aggregate_functions, table_functions,
        scalar_macros, table_macros, indexes;
    std::string schema;
};

std::vector<std::string> binary_list_at(const std::shared_ptr<arrow::Array>& column, int64_t row) {
    const auto list = std::static_pointer_cast<arrow::ListArray>(column);
    const auto values = std::static_pointer_cast<arrow::BinaryArray>(list->values());
    std::vector<std::string> out;
    for (int64_t i = list->value_offset(row); i < list->value_offset(row + 1); ++i) {
        out.push_back(values->GetString(i));
    }
    return out;
}

std::map<std::vector<std::string>, ContentsRow> contents_by_path(
    const std::shared_ptr<arrow::RecordBatch>& payload) {
    const auto schemas =
        std::static_pointer_cast<arrow::ListArray>(payload->GetColumnByName("schemas"));
    const auto rows = std::static_pointer_cast<arrow::StructArray>(schemas->values());
    std::map<std::vector<std::string>, ContentsRow> out;
    for (int64_t r = schemas->value_offset(0); r < schemas->value_offset(1); ++r) {
        const auto path_list =
            std::static_pointer_cast<arrow::ListArray>(rows->GetFieldByName("path"));
        const auto path_values = std::static_pointer_cast<arrow::StringArray>(path_list->values());
        std::vector<std::string> path;
        for (int64_t i = path_list->value_offset(r); i < path_list->value_offset(r + 1); ++i) {
            path.push_back(path_values->GetString(i));
        }
        ContentsRow row;
        row.schema = std::static_pointer_cast<arrow::BinaryArray>(rows->GetFieldByName("schema"))
                         ->GetString(r);
        row.tables = binary_list_at(rows->GetFieldByName("tables"), r);
        row.views = binary_list_at(rows->GetFieldByName("views"), r);
        row.scalar_functions = binary_list_at(rows->GetFieldByName("scalar_functions"), r);
        row.aggregate_functions = binary_list_at(rows->GetFieldByName("aggregate_functions"), r);
        row.table_functions = binary_list_at(rows->GetFieldByName("table_functions"), r);
        row.scalar_macros = binary_list_at(rows->GetFieldByName("scalar_macros"), r);
        row.table_macros = binary_list_at(rows->GetFieldByName("table_macros"), r);
        row.indexes = binary_list_at(rows->GetFieldByName("indexes"), r);
        out[path] = std::move(row);
    }
    return out;
}

std::vector<std::string> per_schema(Served& worker, const std::string& method,
                                    const std::shared_ptr<arrow::Schema>& params_schema,
                                    const std::string& handle, const vgi::SchemaPath& path,
                                    const std::string& type = "") {
    auto builder = vgi::wire::ResultBuilder(params_schema)
                       .set_binary("attach_opaque_data", handle)
                       .set_string_list("path", path);
    if (params_schema->GetFieldIndex("type") >= 0) builder.set_enum("type", type);
    return vgi::wire::get_binary_list(worker.call(method, builder.fill_defaults().finish()),
                                      "items");
}

}  // namespace

TEST_CASE("catalog_contents answers exactly what the per-schema listings do",
          "[catalog][contents]") {
    // The engine trusts either path to describe the same catalog
    // (catalog_contents_conformance.test compares them): every item in the
    // one-call answer must be the per-schema answer's item, byte for byte.
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "alpha";
    dispatcher.register_scalar(std::make_shared<Scalar>("twice", "main scalar"));
    dispatcher.register_table(std::make_shared<Table>("rows", "main table"));
    dispatcher.register_aggregate(std::make_shared<Aggregate>("total", "main aggregate"));
    dispatcher.register_scalar_in("alpha", "side", std::make_shared<Scalar>("twice", "side"));
    vgi::CatalogMacro scalar_macro;
    scalar_macro.name = "add_one";
    scalar_macro.parameters = {"x"};
    scalar_macro.definition = "x + 1";
    dispatcher.catalog().schema("main").macros.push_back(scalar_macro);
    vgi::CatalogMacro table_macro;
    table_macro.name = "one_row";
    table_macro.definition = "SELECT 1 AS n";
    table_macro.table_macro = true;
    dispatcher.catalog().schema("main").macros.push_back(table_macro);

    Served worker(dispatcher);
    const auto handle = attach(worker, "alpha");

    const auto params = vgi::wire::ResultBuilder(gen::CatalogContentsParamsSchema())
                            .set_binary("attach_opaque_data", handle)
                            .set_null("if_none_match")
                            .finish();
    const auto payload = worker.call("catalog_contents", params);
    CHECK_FALSE(
        std::static_pointer_cast<arrow::BooleanArray>(payload->GetColumnByName("not_modified"))
            ->Value(0));
    CHECK(payload->GetColumnByName("etag")->IsNull(0));

    const auto contents = contents_by_path(payload);
    const auto schema_items = vgi::wire::get_binary_list(
        worker.call("catalog_schemas", vgi::wire::ResultBuilder(gen::CatalogSchemasParamsSchema())
                                           .set_binary("attach_opaque_data", handle)
                                           .fill_defaults()
                                           .finish()),
        "items");
    REQUIRE(contents.size() == schema_items.size());

    for (const auto& path : {vgi::SchemaPath{"main"}, vgi::SchemaPath{"side"}}) {
        INFO(path.back());
        REQUIRE(contents.count(path) == 1);
        const auto& row = contents.at(path);
        const auto fn = [&](const std::string& type) {
            return per_schema(worker, "catalog_schema_contents_functions",
                              gen::CatalogSchemaContentsFunctionsParamsSchema(), handle, path,
                              type);
        };
        CHECK(row.scalar_functions == fn("SCALAR_FUNCTION"));
        CHECK(row.aggregate_functions == fn("AGGREGATE_FUNCTION"));
        CHECK(row.table_functions == fn("TABLE_FUNCTION"));
        CHECK(row.tables == per_schema(worker, "catalog_schema_contents_tables",
                                       gen::CatalogSchemaContentsTablesParamsSchema(), handle,
                                       path));
        CHECK(row.views == per_schema(worker, "catalog_schema_contents_views",
                                      gen::CatalogSchemaContentsViewsParamsSchema(), handle, path));
        CHECK(row.scalar_macros == per_schema(worker, "catalog_schema_contents_macros",
                                              gen::CatalogSchemaContentsMacrosParamsSchema(),
                                              handle, path, "SCALAR_MACRO"));
        CHECK(row.table_macros == per_schema(worker, "catalog_schema_contents_macros",
                                             gen::CatalogSchemaContentsMacrosParamsSchema(), handle,
                                             path, "TABLE_MACRO"));
        CHECK(row.indexes.empty());
        CHECK(std::find(schema_items.begin(), schema_items.end(), row.schema) !=
              schema_items.end());
    }
    CHECK(contents.at({"main"}).scalar_macros.size() == 1);
    CHECK(contents.at({"main"}).table_macros.size() == 1);
    CHECK(contents.at({"side"}).scalar_functions.size() == 1);
}

TEST_CASE("ATTACH advertises catalog_contents unless the catalog opts out", "[catalog][contents]") {
    for (const bool advertised : {true, false}) {
        vgi::Dispatcher dispatcher;
        dispatcher.catalog().name = "alpha";
        dispatcher.catalog().supports_catalog_contents = advertised;
        Served worker(dispatcher);
        const auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                                 .set_string("name", "alpha")
                                 .fill_defaults()
                                 .finish();
        const auto result =
            worker.call("catalog_attach", vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                                              .set_binary("request", vgi::wire::encode_ipc(request))
                                              .finish());
        CHECK(std::static_pointer_cast<arrow::BooleanArray>(
                  result->GetColumnByName("supports_catalog_contents"))
                  ->Value(0) == advertised);
    }
}

namespace {

std::shared_ptr<arrow::RecordBatch> contents_call(Served& worker, const std::string& handle,
                                                  const std::optional<std::string>& if_none_match) {
    auto builder = vgi::wire::ResultBuilder(gen::CatalogContentsParamsSchema())
                       .set_binary("attach_opaque_data", handle);
    builder.set_optional_string("if_none_match", if_none_match);
    return worker.call("catalog_contents", builder.finish());
}

std::optional<std::string> etag_of(const std::shared_ptr<arrow::RecordBatch>& payload) {
    return vgi::wire::get_optional_string(payload, "etag");
}

bool not_modified_of(const std::shared_ptr<arrow::RecordBatch>& payload) {
    return vgi::wire::get_bool(payload, "not_modified");
}

size_t schema_count(const std::shared_ptr<arrow::RecordBatch>& payload) {
    return contents_by_path(payload).size();
}

void create_view(Served& worker, const std::string& handle, const std::string& name,
                 const std::string& definition) {
    worker.call_void("catalog_view_create",
                     vgi::wire::ResultBuilder(gen::CatalogViewCreateParamsSchema())
                         .set_binary("attach_opaque_data", handle)
                         .set_string_list("schema_path", {"main"})
                         .set_string("name", name)
                         .set_string("definition", definition)
                         .set_enum("on_conflict", "ERROR")
                         .fill_defaults()
                         .finish());
}

int64_t version_of(Served& worker, const std::string& handle) {
    return vgi::wire::get_int64(
        worker.call("catalog_version", vgi::wire::ResultBuilder(gen::CatalogVersionParamsSchema())
                                           .set_binary("attach_opaque_data", handle)
                                           .fill_defaults()
                                           .finish()),
        "version");
}

std::vector<std::string> view_names(Served& worker, const std::string& handle) {
    std::vector<std::string> names;
    for (const auto& item :
         per_schema(worker, "catalog_schema_contents_views",
                    gen::CatalogSchemaContentsViewsParamsSchema(), handle, {"main"})) {
        names.push_back(vgi::wire::get_string(vgi::wire::decode_ipc(item), "name"));
    }
    return names;
}

}  // namespace

TEST_CASE("a memory catalog takes DDL, privately per ATTACH, and bumps its version",
          "[catalog][memory]") {
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "alpha";
    vgi::MemoryCatalogOptions options;
    options.name = "mem";
    dispatcher.register_memory_catalog(options);
    Served worker(dispatcher);

    const auto one = attach(worker, "mem");
    const auto two = attach(worker, "mem");
    REQUIRE(one != two);
    CHECK(version_of(worker, one) == 1);

    create_view(worker, one, "v", "SELECT 1 AS x");
    CHECK(version_of(worker, one) == 2);
    CHECK(view_names(worker, one) == Names{"v"});
    // The second ATTACH never sees the first's objects.
    CHECK(view_names(worker, two).empty());
    CHECK(version_of(worker, two) == 1);

    // A table definition, then the schema set and per-name lookup.
    worker.call_void("catalog_schema_create",
                     vgi::wire::ResultBuilder(gen::CatalogSchemaCreateParamsSchema())
                         .set_binary("attach_opaque_data", one)
                         .set_string_list("path", {"s2"})
                         .set_enum("on_conflict", "ERROR")
                         .fill_defaults()
                         .finish());
    const auto table_request = vgi::wire::ResultBuilder(gen::TableCreateRequestSchema())
                                   .set_binary("attach_opaque_data", one)
                                   .set_string_list("schema_path", {"main"})
                                   .set_string("name", "t1")
                                   .set_binary("columns", vgi::wire::encode_schema(arrow::schema(
                                                              {arrow::field("a", arrow::int32())})))
                                   .set_enum("on_conflict", "ERROR")
                                   .fill_defaults()
                                   .finish();
    worker.call_void("catalog_table_create",
                     vgi::wire::ResultBuilder(gen::CatalogTableCreateParamsSchema())
                         .set_binary("request", vgi::wire::encode_ipc(table_request))
                         .finish());
    CHECK(version_of(worker, one) == 4);
    const auto contents = contents_by_path(contents_call(worker, one, std::nullopt));
    REQUIRE(contents.size() == 2);
    CHECK(contents.at({"main"}).tables.size() == 1);
    CHECK(contents.at({"main"}).views.size() == 1);
    CHECK(contents.count({"s2"}) == 1);

    // Creating it again is refused; IGNORE keeps it.
    CHECK_THROWS(create_view(worker, one, "v", "SELECT 2 AS x"));

    worker.call_void("catalog_view_drop",
                     vgi::wire::ResultBuilder(gen::CatalogViewDropParamsSchema())
                         .set_binary("attach_opaque_data", one)
                         .set_string_list("schema_path", {"main"})
                         .set_string("name", "V")
                         .fill_defaults()
                         .finish());
    CHECK(view_names(worker, one).empty());

    // Detached, the state is gone.
    worker.call_void("catalog_detach", vgi::wire::ResultBuilder(gen::CatalogDetachParamsSchema())
                                           .set_binary("attach_opaque_data", one)
                                           .fill_defaults()
                                           .finish());
    CHECK_THROWS(version_of(worker, one));
    CHECK(version_of(worker, two) == 1);
}

TEST_CASE("catalog_catalogs lists memory catalogs; declared catalogs route as before",
          "[catalog][memory]") {
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "alpha";
    dispatcher.register_scalar(std::make_shared<Scalar>("twice", "alpha main"));
    vgi::MemoryCatalogOptions options;
    options.name = "mem";
    dispatcher.register_memory_catalog(options);
    REQUIRE_THROWS(dispatcher.register_memory_catalog(options));
    Served worker(dispatcher);

    std::vector<std::string> names;
    for (const auto& item : vgi::wire::get_binary_list(
             worker.call("catalog_catalogs",
                         vgi::wire::ResultBuilder(gen::CatalogCatalogsParamsSchema())
                             .fill_defaults()
                             .finish()),
             "items")) {
        names.push_back(vgi::wire::get_string(vgi::wire::decode_ipc(item), "name"));
    }
    CHECK(names == Names{"alpha", "mem"});

    const auto alpha = attach(worker, "alpha");
    CHECK(described(list(worker, alpha, {"main"}, "SCALAR_FUNCTION")) ==
          Names{"twice: alpha main"});
    // The memory catalog lists none of alpha's functions.
    CHECK(list(worker, attach(worker, "mem"), {"main"}, "SCALAR_FUNCTION").empty());
}

TEST_CASE("catalog_contents revalidates: a handler's etag and the content hash",
          "[catalog][contents]") {
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "alpha";
    int builds = 0;
    vgi::MemoryCatalogOptions reval;
    reval.name = "reval";
    reval.catalog_contents_handler = [&builds](const vgi::CatalogContentsCall& call) {
        vgi::CatalogContentsResult result;
        result.etag = "gen-" + std::to_string(call.catalog_version);
        if (call.if_none_match == result.etag) {
            result.not_modified = true;
            return result;
        }
        ++builds;
        result.schemas = call.contents();
        return result;
    };
    dispatcher.register_memory_catalog(reval);
    vgi::MemoryCatalogOptions hash;
    hash.name = "hash";
    hash.catalog_contents_etag = vgi::CatalogContentsEtag::ContentHash;
    dispatcher.register_memory_catalog(hash);
    vgi::MemoryCatalogOptions bad;
    bad.name = "bad";
    bad.catalog_contents_handler = [](const vgi::CatalogContentsCall&) {
        vgi::CatalogContentsResult result;
        result.etag = "x";
        result.not_modified = true;  // without a matching if_none_match
        return result;
    };
    dispatcher.register_memory_catalog(bad);
    Served worker(dispatcher);

    SECTION("generation etag") {
        const auto rv = attach(worker, "reval");
        const auto full = contents_call(worker, rv, std::nullopt);
        CHECK(etag_of(full) == "gen-1");
        CHECK_FALSE(not_modified_of(full));
        CHECK(builds == 1);

        const auto same = contents_call(worker, rv, std::string("gen-1"));
        CHECK(not_modified_of(same));
        CHECK(schema_count(same) == 0);
        CHECK(builds == 1);  // answered without building

        create_view(worker, rv, "v", "SELECT 7 AS x");
        const auto changed = contents_call(worker, rv, std::string("gen-1"));
        CHECK_FALSE(not_modified_of(changed));
        CHECK(etag_of(changed) == "gen-2");
    }

    SECTION("content hash") {
        const auto hs = attach(worker, "hash");
        const auto full = contents_call(worker, hs, std::nullopt);
        const auto etag = etag_of(full);
        REQUIRE(etag);
        CHECK(etag->size() == 64);
        CHECK(etag->find_first_not_of("0123456789abcdef") == std::string::npos);

        const auto same = contents_call(worker, hs, etag);
        CHECK(not_modified_of(same));
        CHECK(etag_of(same) == etag);

        create_view(worker, hs, "w", "SELECT 10 AS y");
        const auto changed = contents_call(worker, hs, etag);
        CHECK_FALSE(not_modified_of(changed));
        CHECK(etag_of(changed) != etag);

        // A catalog with no etag ignores if_none_match.
        const auto alpha = attach(worker, "alpha");
        const auto plain = contents_call(worker, alpha, std::string("anything"));
        CHECK_FALSE(not_modified_of(plain));
        CHECK_FALSE(etag_of(plain));
    }

    SECTION("not_modified needs the matching etag") {
        CHECK_THROWS(contents_call(worker, attach(worker, "bad"), std::nullopt));
    }
}

TEST_CASE("catalog_contents_digest is deterministic and covers every part", "[catalog][contents]") {
    vgi::SchemaContents schema;
    schema.path = {"main"};
    schema.schema = "info";
    schema.tables = {"t"};
    const auto digest = vgi::catalog_contents_digest({schema});
    // Computed by vgi-python's catalog_contents_digest over the same snapshot:
    // the content-hash etag must agree byte for byte across SDKs.
    CHECK(digest == "c46f04af1c42014e1663526c9a195044c7eef5346e97a7d1ffad5aa95164ee0f");
    auto moved = schema;
    moved.tables.clear();
    moved.views = {"t"};
    CHECK(vgi::catalog_contents_digest({moved}) != digest);
    // The empty answer: SHA-256 of one zero length prefix.
    CHECK(vgi::catalog_contents_digest({}) ==
          "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc");
}
