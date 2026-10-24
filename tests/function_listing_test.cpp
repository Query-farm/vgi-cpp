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
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <vgi_rpc/client.h>
#include <vgi_rpc/crypto.h>
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

// The rows of one catalog_contents collection, each re-encoded as the
// one-row IPC item the per-schema listings answer with. Null is no rows.
std::vector<std::shared_ptr<arrow::RecordBatch>> collection_rows(
    const std::shared_ptr<arrow::RecordBatch>& payload, const std::string& field) {
    const auto column =
        std::static_pointer_cast<arrow::BinaryArray>(payload->GetColumnByName(field));
    std::vector<std::shared_ptr<arrow::RecordBatch>> rows;
    if (column->IsNull(0)) return rows;
    auto source =
        std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(column->GetString(0)));
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(source).ValueOrDie();
    std::shared_ptr<arrow::RecordBatch> batch;
    while (reader->ReadNext(&batch).ok() && batch) {
        for (int64_t r = 0; r < batch->num_rows(); ++r) rows.push_back(batch->Slice(r, 1));
    }
    return rows;
}

std::map<std::vector<std::string>, ContentsRow> contents_by_path(
    const std::shared_ptr<arrow::RecordBatch>& payload) {
    std::map<std::vector<std::string>, ContentsRow> out;
    for (const auto& row : collection_rows(payload, "schemas")) {
        out[vgi::wire::get_schema_path(row, "path")].schema = vgi::wire::encode_ipc(row);
    }
    const auto place = [&](const char* field, auto pick) {
        for (const auto& row : collection_rows(payload, field)) {
            pick(out[vgi::wire::get_schema_path(row, "schema_path")], row)
                .push_back(vgi::wire::encode_ipc(row));
        }
    };
    using Row = std::shared_ptr<arrow::RecordBatch>;
    place("tables", [](ContentsRow& c, const Row&) -> auto& { return c.tables; });
    place("views", [](ContentsRow& c, const Row&) -> auto& { return c.views; });
    place("indexes", [](ContentsRow& c, const Row&) -> auto& { return c.indexes; });
    place("functions", [](ContentsRow& c, const Row& row) -> auto& {
        const auto kind = vgi::wire::get_optional_enum(row, "function_type").value_or("");
        if (kind == "SCALAR") return c.scalar_functions;
        if (kind == "AGGREGATE") return c.aggregate_functions;
        return c.table_functions;
    });
    place("macros", [](ContentsRow& c, const Row& row) -> auto& {
        return vgi::wire::get_optional_enum(row, "macro_type") == "TABLE" ? c.table_macros
                                                                          : c.scalar_macros;
    });
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
    scalar_macro.defaults = {{"x", 7}};
    scalar_macro.parameter_docs = {{"x", ""}};
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
    const auto info = vgi::wire::decode_ipc(contents.at({"main"}).scalar_macros[0]);
    CHECK(info->num_columns() == 6);
    const auto families = std::static_pointer_cast<arrow::ListArray>(info->GetColumnByName("overloads"));
    REQUIRE(families->value_length(0) == 1);
    const auto overload = std::static_pointer_cast<arrow::StructArray>(families->values());
    const auto macro_params = std::static_pointer_cast<arrow::ListArray>(overload->GetFieldByName("parameters"));
    const auto parameter = std::static_pointer_cast<arrow::StructArray>(macro_params->values());
    CHECK(parameter->GetFieldByName("declared_type")->IsNull(0));
    CHECK(parameter->GetFieldByName("default_expression")->IsNull(0));
    CHECK_FALSE(parameter->GetFieldByName("documentation")->IsNull(0));
    CHECK(std::static_pointer_cast<arrow::StringArray>(parameter->GetFieldByName("documentation"))->GetString(0).empty());
    const auto default_bytes = std::static_pointer_cast<arrow::BinaryArray>(parameter->GetFieldByName("default_value"))->GetString(0);
    const auto value = vgi::wire::decode_ipc(default_bytes);
    CHECK(value->schema()->field(0)->name() == "x");
    CHECK(std::static_pointer_cast<arrow::Int64Array>(value->column(0))->Value(0) == 7);

    // Direct provider families retain every overload, with no first-entry projection.
    const auto overload_bytes = vgi::wire::encode_ipc(arrow::RecordBatch::Make(gen::MacroOverloadSchema(), 1, overload->fields()));
    vgi::Dispatcher family_dispatcher;
    family_dispatcher.catalog().name = "family";
    vgi::CatalogMacro family_macro;
    family_macro.name = "many";
    const auto second_overload = vgi::wire::ResultBuilder(gen::MacroOverloadSchema())
        .set_string("definition", "1").set_string("profile", "duckdb.macro.v1")
        .set_string_list("schema_path", {"main"}).set_string("macro_type", "scalar")
        .set_array("evaluation_context", overload->GetFieldByName("evaluation_context"))
        .fill_defaults().finish();
    family_macro.overloads = std::vector<std::string>{overload_bytes, vgi::wire::encode_ipc(second_overload)};
    family_dispatcher.catalog().schema("main").macros.push_back(family_macro);
    Served family_worker(family_dispatcher);
    const auto family_handle = attach(family_worker, "family");
    const auto family_items = per_schema(family_worker, "catalog_schema_contents_macros",
        gen::CatalogSchemaContentsMacrosParamsSchema(), family_handle, {"main"}, "SCALAR_MACRO");
    REQUIRE(family_items.size() == 1);
    const auto restored = vgi::wire::decode_ipc(family_items[0]);
    const auto full_family = std::static_pointer_cast<arrow::ListArray>(restored->GetColumnByName("overloads"));
    CHECK(full_family->value_length(0) == 2);

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

TEST_CASE("catalog read profiles require provider and client opt-in", "[catalog][capabilities]") {
    const std::vector<std::string> profiles = {"vgi.constraint.records.read.v1",
                                              "vgi.expression.duckdb.scalar.v1"};
    for (const bool provider_enabled : {false, true}) {
        for (const int offer_count : {-1, 0, 1, 2, 3}) {
            INFO(provider_enabled << " / " << offer_count);
            vgi::Dispatcher dispatcher;
            dispatcher.catalog().name = "alpha";
            if (provider_enabled) dispatcher.catalog().catalog_read_profiles = profiles;
            Served worker(dispatcher);
            auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                               .set_string("name", "alpha");
            if (offer_count >= 0) {
                auto offer =
                    vgi::wire::ResultBuilder(gen::CatalogCapabilitiesSchema())
                        .set_string_list(
                            "read", offer_count == 3
                                        ? std::vector<std::string>{profiles[1]}
                                        : std::vector<std::string>(profiles.begin(),
                                                                   profiles.begin() + offer_count))
                        .set_string_list("ddl", {})
                        .set_string_list("write", {})
                        .set_string_list("transaction", {})
                        .finish();
                auto client = vgi::wire::ResultBuilder(gen::ClientCapabilitiesSchema())
                                  .set_binary("catalog", vgi::wire::encode_ipc(offer))
                                  .fill_defaults()
                                  .finish();
                request.set_binary("client_capabilities", vgi::wire::encode_ipc(client));
            }
            auto result = worker.call(
                "catalog_attach",
                vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                    .set_binary("request", vgi::wire::encode_ipc(request.fill_defaults().finish()))
                    .finish());
            auto selected = vgi::wire::get_optional_binary(result, "selected_capabilities");
            auto available = vgi::wire::get_optional_binary(result, "catalog_capabilities");
            if (!provider_enabled) {
                CHECK_FALSE(selected.has_value());
                CHECK_FALSE(available.has_value());
            } else {
                REQUIRE(selected.has_value());
                REQUIRE(available.has_value());
                auto agreement = vgi::wire::decode_ipc(*selected);
                CHECK(vgi::wire::get_string_list(agreement, "read").size() ==
                      static_cast<size_t>(offer_count == 3 ? 0 : std::max(offer_count, 0)));
                CHECK(vgi::wire::get_string_list(agreement, "write").empty());
                CHECK(vgi::wire::get_string_list(vgi::wire::decode_ipc(*available), "read") ==
                      profiles);
            }
        }
    }
}

TEST_CASE("catalog capability negotiation rejects invalid and trailing offers",
          "[catalog][capabilities]") {
    for (const std::string mode : {"duplicate", "invalid_name", "trailing", "provider_duplicate"}) {
        INFO(mode);
        vgi::Dispatcher dispatcher;
        dispatcher.catalog().name = "alpha";
        dispatcher.catalog().catalog_read_profiles = {"vgi.constraint.records.read.v1"};
        if (mode == "provider_duplicate")
            dispatcher.catalog().catalog_read_profiles.push_back("vgi.constraint.records.read.v1");
        Served worker(dispatcher);
        std::vector<std::string> profiles = {"vgi.constraint.records.read.v1"};
        if (mode == "duplicate") profiles.push_back(profiles.front());
        if (mode == "invalid_name") profiles[0] = "not-a-versioned-profile";
        auto offer =
            vgi::wire::encode_ipc(vgi::wire::ResultBuilder(gen::CatalogCapabilitiesSchema())
                                      .set_string_list("read", profiles)
                                      .set_string_list("ddl", {})
                                      .set_string_list("write", {})
                                      .set_string_list("transaction", {})
                                      .finish());
        if (mode == "trailing") offer.push_back('\0');
        auto client = vgi::wire::ResultBuilder(gen::ClientCapabilitiesSchema())
                          .set_binary("catalog", offer)
                          .fill_defaults()
                          .finish();
        auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                           .set_string("name", "alpha")
                           .set_binary("client_capabilities", vgi::wire::encode_ipc(client))
                           .fill_defaults()
                           .finish();
        REQUIRE_THROWS(
            worker.call("catalog_attach", vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                                              .set_binary("request", vgi::wire::encode_ipc(request))
                                              .finish()));
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
                 const std::string& definition, const std::string& security = "regular") {
    worker.call_void("catalog_view_create",
                     vgi::wire::ResultBuilder(gen::CatalogViewCreateParamsSchema())
                         .set_binary("attach_opaque_data", handle)
                         .set_string_list("schema_path", {"main"})
                         .set_string("name", name)
                         .set_string("definition", definition)
                         .set_string("security", security)
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

    const auto security_version = version_of(worker, one);
    REQUIRE_THROWS_WITH(create_view(worker, one, "v", "SELECT 2", "barrier"), Catch::Matchers::ContainsSubstring("security"));
    REQUIRE(version_of(worker, one) == security_version);
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
    // An unported provider must not manufacture an empty semantic dependency graph.
    REQUIRE_THROWS(worker.call_void("catalog_dependencies", vgi::wire::ResultBuilder(gen::CatalogDependenciesParamsSchema())
        .set_binary("attach_opaque_data", one).fill_defaults().finish()));
    REQUIRE(version_of(worker, one) == 4);
    // Unported index operations must fail before REPLACE, DROP or option binding can publish anything.
    std::ifstream index_input(VGI_CATALOG_CONTRACT_VECTORS);
    REQUIRE(index_input.good());
    const auto index_records = nlohmann::json::parse(index_input).at("records");
    const auto index_definition = vgi::wire::base64_decode(index_records.at("index_definition").get<std::string>());
    for (const auto* conflict : {"ERROR", "IGNORE", "REPLACE"}) {
        const auto rejected = vgi::wire::ResultBuilder(gen::IndexCreateRequestSchema())
            .set_binary("attach_opaque_data", one).set_string_list("schema_path", {"main"})
            .set_string("name", "t1").set_binary("definition", index_definition)
            .set_enum("on_conflict", conflict).fill_defaults().finish();
        REQUIRE_THROWS(worker.call_void("catalog_index_create", vgi::wire::ResultBuilder(gen::CatalogIndexCreateParamsSchema())
            .set_binary("request", vgi::wire::encode_ipc(rejected)).finish()));
    }
    REQUIRE_THROWS(worker.call_void("catalog_index_drop", vgi::wire::ResultBuilder(gen::CatalogIndexDropParamsSchema())
        .set_binary("attach_opaque_data", one).set_string_list("schema_path", {"main"}).set_string("name", "t1")
        .set_bool("ignore_not_found", true).set_bool("cascade", true).fill_defaults().finish()));
    REQUIRE_THROWS(worker.call_void("catalog_index_option_definitions", vgi::wire::ResultBuilder(gen::CatalogIndexOptionDefinitionsParamsSchema())
        .set_binary("attach_opaque_data", one)
        .set_binary("table", vgi::wire::base64_decode(index_records.at("object_ref").get<std::string>()))
        .set_string("algorithm", "ART").fill_defaults().finish()));
    REQUIRE(version_of(worker, one) == 4);
    for (const auto* conflict : {"ERROR", "IGNORE", "REPLACE"}) {
        const auto rejected = vgi::wire::ResultBuilder(gen::TableCreateRequestSchema())
                                  .set_binary("attach_opaque_data", one)
                                  .set_string_list("schema_path", {"main"})
                                  .set_string("name", "t1")
                                  .set_binary("columns", vgi::wire::encode_schema(arrow::schema(
                                                            {arrow::field("replacement", arrow::utf8())})))
                                  .set_enum("on_conflict", conflict)
                                  .set_binary_list("constraint_definitions", {"unsupported"})
                                  .fill_defaults()
                                  .finish();
        REQUIRE_THROWS(worker.call_void("catalog_table_create",
                           vgi::wire::ResultBuilder(gen::CatalogTableCreateParamsSchema())
                               .set_binary("request", vgi::wire::encode_ipc(rejected))
                               .finish()));
        REQUIRE(version_of(worker, one) == 4);
    }
    for (const auto& field : {
             arrow::field("required", arrow::int32(), false),
             arrow::field("nested", arrow::struct_({arrow::field("required", arrow::int32(), false)}))}) {
        const auto rejected = vgi::wire::ResultBuilder(gen::TableCreateRequestSchema())
                                  .set_binary("attach_opaque_data", one)
                                  .set_string_list("schema_path", {"main"})
                                  .set_string("name", "t1")
                                  .set_binary("columns", vgi::wire::encode_schema(arrow::schema({field})))
                                  .set_enum("on_conflict", "REPLACE")
                                  .fill_defaults()
                                  .finish();
        REQUIRE_THROWS(worker.call_void("catalog_table_create",
                           vgi::wire::ResultBuilder(gen::CatalogTableCreateParamsSchema())
                               .set_binary("request", vgi::wire::encode_ipc(rejected))
                               .finish()));
        REQUIRE(version_of(worker, one) == 4);
    }
    const auto contents = contents_by_path(contents_call(worker, one, std::nullopt));
    REQUIRE(contents.size() == 2);
    CHECK(contents.at({"main"}).tables.size() == 1);
    const auto original_table = vgi::wire::decode_ipc(contents.at({"main"}).tables.front());
    CHECK(vgi::wire::get_binary(original_table, "columns") == vgi::wire::get_binary(table_request, "columns"));
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

namespace {

// vgi-python 2.2's contents_codec.contents_digest, restated independently of
// the SDK: the content-hash etag must be this framing over the six collection
// streams the response actually carries.
std::string python_contents_digest(const std::shared_ptr<arrow::RecordBatch>& payload) {
    std::string framed("vgi.catalog.contents.2.2", 24);
    framed.push_back('\0');
    for (const char* kind : {"schemas", "tables", "views", "functions", "macros", "indexes"}) {
        framed += kind;
        framed.push_back('\0');
        const auto data = vgi::wire::get_optional_binary(payload, kind);
        framed.push_back(data ? '\1' : '\0');
        if (!data) continue;
        for (int b = 0; b < 8; ++b)
            framed.push_back(static_cast<char>(uint64_t(data->size()) >> (8 * b)));
        framed += *data;
    }
    vgi_rpc::crypto::Sha256 hash;
    hash.update(framed);
    return hash.hex_digest();
}

}  // namespace

TEST_CASE("catalog_contents_digest hashes the 2.2 collection streams", "[catalog][contents]") {
    // The empty answer (every collection null) is the one catalog whose etag
    // agrees across SDKs; this value is vgi-python a34e949's
    // contents_codec.contents_digest({kind: None for kind in CATEGORY_TYPES}).
    // A non-empty catalog cannot agree byte for byte: vgi-python packs up to
    // 1024 rows per batch, this SDK one batch per item.
    CHECK(vgi::catalog_contents_digest({}) ==
          "57f8bf1399afe1637da5a2dde8329befa007315fe4a6cb842cc152edbefaf0fc");

    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "digest";
    dispatcher.catalog().catalog_contents_etag = vgi::CatalogContentsEtag::ContentHash;
    dispatcher.register_table(std::make_shared<Table>("t", "table"));
    Served worker(dispatcher);
    const auto payload = contents_call(worker, attach(worker, "digest"), std::nullopt);
    const auto etag = etag_of(payload);
    REQUIRE(etag);
    CHECK(*etag == python_contents_digest(payload));
    // The public digest of the same snapshot (one schema, so order is moot).
    std::vector<vgi::SchemaContents> snapshot;
    for (const auto& [path, row] : contents_by_path(payload)) {
        snapshot.push_back({path, row.schema, row.tables, row.views, row.scalar_functions,
                            row.aggregate_functions, row.table_functions, row.scalar_macros,
                            row.table_macros, row.indexes});
    }
    REQUIRE(snapshot.size() == 1);
    CHECK(vgi::catalog_contents_digest(snapshot) == *etag);
}

namespace {

// A dictionary-encoded enum cell, as decoded.
std::string enum_of(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    return vgi::wire::get_enum(batch, field);
}

}  // namespace

TEST_CASE("function and macro kinds travel as the Python enum member names",
          "[catalog][listing][enum]") {
    // `FunctionInfo.function_type` and `MacroInfo.macro_type` are the
    // reference's `FunctionType` / `MacroType` *member names*, not their
    // lowercase values: a strict decoder rejects `scalar`.
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "kinds";
    dispatcher.register_scalar(std::make_shared<Scalar>("s", "scalar"));
    dispatcher.register_table(std::make_shared<Table>("t", "table"));
    dispatcher.register_aggregate(std::make_shared<Aggregate>("a", "aggregate"));
    auto& main = dispatcher.catalog().schema("main");
    main.macros.push_back({"m_scalar", {"x"}, "x + 1", /*table_macro=*/false});
    main.macros.push_back({"m_table", {}, "SELECT 1", /*table_macro=*/true});

    Served worker(dispatcher);
    const auto handle = attach(worker, "kinds");
    const auto kind_of = [&](const std::string& filter) {
        std::vector<std::string> kinds;
        for (const auto& item : list(worker, handle, {"main"}, filter)) {
            kinds.push_back(enum_of(vgi::wire::decode_ipc(item), "function_type"));
        }
        return kinds;
    };
    CHECK(kind_of("SCALAR_FUNCTION") == Names{"SCALAR"});
    CHECK(kind_of("TABLE_FUNCTION") == Names{"TABLE"});
    CHECK(kind_of("AGGREGATE_FUNCTION") == Names{"AGGREGATE"});

    const auto macros = [&](const std::string& filter) {
        const auto params = vgi::wire::ResultBuilder(gen::CatalogSchemaContentsMacrosParamsSchema())
                                .set_binary("attach_opaque_data", handle)
                                .set_string_list("path", {"main"})
                                .set_enum("type", filter)
                                .fill_defaults()
                                .finish();
        std::vector<std::string> kinds;
        for (const auto& item : vgi::wire::get_binary_list(
                 worker.call("catalog_schema_contents_macros", params), "items")) {
            kinds.push_back(enum_of(vgi::wire::decode_ipc(item), "macro_type"));
        }
        return kinds;
    };
    CHECK(macros("SCALAR_MACRO") == Names{"SCALAR"});
    CHECK(macros("TABLE_MACRO") == Names{"TABLE"});
}

namespace {

// A table-in-out that only declares itself, for registration checks.
class Sliced final : public vgi::TableInOutFunction {
public:
    Sliced(bool blended, std::optional<int64_t> rows) : blended_(blended), rows_(rows) {}
    std::string name() const override { return "sliced"; }
    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata metadata;
        metadata.input_from_args = blended_;
        metadata.input_rows_per_call = rows_;
        return metadata;
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("n", 0, "int64")};
    }
    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams&, const std::shared_ptr<arrow::RecordBatch>&) const override {
        return {};
    }

private:
    bool blended_;
    std::optional<int64_t> rows_;
};

}  // namespace

TEST_CASE("input_rows_per_call is advertised for a blended function and refused elsewhere",
          "[catalog][listing]") {
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "rows";
    CHECK_THROWS(dispatcher.register_table_in_out(std::make_shared<Sliced>(true, 0)));
    CHECK_THROWS(dispatcher.register_table_in_out(std::make_shared<Sliced>(false, 3)));
    dispatcher.register_table_in_out(std::make_shared<Sliced>(true, 3));

    Served worker(dispatcher);
    const auto items = list(worker, attach(worker, "rows"), {"main"}, "TABLE_FUNCTION");
    REQUIRE(items.size() == 1);
    CHECK(vgi::wire::get_optional_int64(vgi::wire::decode_ipc(items[0]), "input_rows_per_call") ==
          3);
}

TEST_CASE("column metadata survives CREATE and discovery and invalid metadata has no effects", "[catalog][column-metadata]") {
    std::ifstream input(VGI_CATALOG_CONTRACT_VECTORS);
    const auto vectors=nlohmann::json::parse(input).at("column_metadata_schemas");
    vgi::Dispatcher dispatcher;dispatcher.catalog().name="alpha";
    vgi::MemoryCatalogOptions options;options.name="mem";dispatcher.register_memory_catalog(options);
    Served worker(dispatcher);const auto handle=attach(worker,"mem");
    for(const auto& group:vectors.items())for(const auto& item:group.value().items()){
        INFO(group.key()+"/"+item.key());
        const auto raw=vgi::wire::base64_decode(item.value().get<std::string>());
        const auto request=vgi::wire::ResultBuilder(gen::TableCreateRequestSchema())
            .set_binary("attach_opaque_data",handle).set_string_list("schema_path",{"main"})
            .set_string("name",item.key()).set_binary("columns",raw).set_enum("on_conflict","ERROR").fill_defaults().finish();
        const auto params=vgi::wire::ResultBuilder(gen::CatalogTableCreateParamsSchema()).set_binary("request",vgi::wire::encode_ipc(request)).finish();
        const auto before=version_of(worker,handle);
        if(group.key()=="invalid"){
            REQUIRE_THROWS(worker.call_void("catalog_table_create",params));
            CHECK(version_of(worker,handle)==before);
        }else{
            worker.call_void("catalog_table_create",params);
            const auto listed=per_schema(worker,"catalog_schema_contents_tables",gen::CatalogSchemaContentsTablesParamsSchema(),handle,{"main"});
            bool found=false;
            for(const auto& bytes:listed){const auto info=vgi::wire::decode_ipc(bytes);if(vgi::wire::get_string(info,"name")==item.key()){
                CHECK(vgi::wire::get_binary(info,"columns")==raw);found=true;
            }}
            REQUIRE(found);
            vgi::Dispatcher readonly;readonly.catalog().name="meta";
            vgi::CatalogTable table;table.name=item.key();table.columns=vgi::wire::decode_schema(raw);
            readonly.catalog().schema("main").tables.push_back(table);
            Served reader(readonly);const auto token=attach(reader,"meta");
            const auto items=per_schema(reader,"catalog_schema_contents_tables",gen::CatalogSchemaContentsTablesParamsSchema(),token,{"main"});
            REQUIRE(items.size()==1);
            const auto info=vgi::wire::decode_ipc(items[0]);
            CHECK(vgi::wire::get_binary(info,"columns")==vgi::wire::encode_schema(table.columns));
        }
    }
    CHECK(per_schema(worker,"catalog_schema_contents_tables",gen::CatalogSchemaContentsTablesParamsSchema(),handle,{"main"}).size()==4);
}

TEST_CASE("canonical generated discovery and unsupported CREATE preserve authority", "[catalog][generated-columns]") {
    std::ifstream input(VGI_CATALOG_CONTRACT_VECTORS);
    const auto vectors = nlohmann::json::parse(input).at("generated_column_schemas");
    vgi::Dispatcher dispatcher; dispatcher.catalog().name = "alpha";
    vgi::MemoryCatalogOptions options; options.name = "mem"; dispatcher.register_memory_catalog(options);
    Served worker(dispatcher); const auto handle = attach(worker, "mem");
    for (const auto& group : {"valid", "invalid"}) for (const auto& item : vectors.at(group).items()) {
        INFO(std::string(group) + "/" + item.key());
        const auto raw = vgi::wire::base64_decode(item.value().get<std::string>());
        vgi::Dispatcher readonly; readonly.catalog().name = "meta";
        vgi::CatalogTable table; table.name = item.key(); table.columns = vgi::wire::decode_schema(raw);
        readonly.catalog().schema("main").tables.push_back(table);
        Served reader(readonly); const auto token = attach(reader, "meta");
        if (std::string(group) == "invalid") {
            REQUIRE_THROWS(per_schema(reader, "catalog_schema_contents_tables", gen::CatalogSchemaContentsTablesParamsSchema(), token, {"main"}));
        } else {
            const auto items = per_schema(reader, "catalog_schema_contents_tables", gen::CatalogSchemaContentsTablesParamsSchema(), token, {"main"});
            REQUIRE(items.size() == 1);
            CHECK(vgi::wire::get_binary(vgi::wire::decode_ipc(items[0]), "columns") == vgi::wire::encode_schema(table.columns));
        }
        const auto request = vgi::wire::ResultBuilder(gen::TableCreateRequestSchema())
            .set_binary("attach_opaque_data", handle).set_string_list("schema_path", {"main"})
            .set_string("name", item.key()).set_binary("columns", raw).set_enum("on_conflict", "ERROR").fill_defaults().finish();
        const auto params = vgi::wire::ResultBuilder(gen::CatalogTableCreateParamsSchema()).set_binary("request", vgi::wire::encode_ipc(request)).finish();
        const auto before = version_of(worker, handle);
        if (std::string(group) == "valid" && item.key() == "ordinary") worker.call_void("catalog_table_create", params);
        else { REQUIRE_THROWS(worker.call_void("catalog_table_create", params)); CHECK(version_of(worker, handle) == before); }
    }
    CHECK(per_schema(worker, "catalog_schema_contents_tables", gen::CatalogSchemaContentsTablesParamsSchema(), handle, {"main"}).size() == 1);
    const auto encoded = vgi::generated_expression_metadata("n * 2", {"raw.schema"});
    const auto expression = vgi::wire::decode_ipc(vgi::wire::base64_decode(encoded));
    CHECK(vgi::wire::get_string_list(expression, "schema_path") == vgi::SchemaPath{"raw.schema"});
    CHECK(vgi::wire::get_string(expression, "sql") == "n * 2");
}

namespace {
class ProjectionContractTable : public vgi::TableFunction {
public:
    std::atomic<int> mode{0};
    vgi::BindRepeatability claim = vgi::BindRepeatability::Unknown;
    vgi::BindRepeatability repeatability(const vgi::BindParams&) const override { return claim; }
    std::string name() const override { return "projection_contract"; }
    vgi::FunctionMetadata metadata() const override { return {}; }
    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override { return arrow::schema({arrow::field("n", arrow::int32(), true)}, arrow::key_value_metadata({"source"},{"keep"})); }
    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams&) const override { return nullptr; }
    std::vector<std::string> projection_expression_profiles(const vgi::BindParams&) const override { return {vgi::SCALAR_PROJECTION_PROFILE, "vendor.scalar.v3"}; }
    vgi::ProjectionAcceptance projection(const vgi::BindParams&,const vgi::ProjectionProposal& input) const override {
        if (mode == 3) return {};
        auto fields = input.source_schema->fields(); fields.push_back(input.result_schema->field(0));
        vgi::ProjectionAcceptance result{{0},arrow::schema(fields,input.source_schema->metadata()),"provider-state"};
        if (mode == 1) result.accepted_indices = {0,0};
        if (mode == 2) result.output_schema = arrow::schema(fields);
        if (mode == 4) result.output_schema = nullptr;
        if (mode == 5) result.accepted_indices = {1};
        if (mode == 6) {fields.back() = arrow::field("wide",arrow::int64(),true); result.output_schema = arrow::schema(fields,input.source_schema->metadata());}
        return result;
    }
};
}
TEST_CASE("scalar projection validates provider acceptance before changing a scan",
          "[projection]") {
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "projection";
    auto function = std::make_shared<ProjectionContractTable>();
    dispatcher.register_table(function);
    Served worker(dispatcher);
    const auto call = vgi::wire::ResultBuilder(gen::BindRequestSchema())
                          .set_string("function_name", function->name())
                          .set_enum("function_type", "TABLE")
                          .set_binary("arguments", vgi::wire::encode_schema(arrow::schema({})))
                          .set_string_list("schema_path", {"main"})
                          .fill_defaults()
                          .finish();
    for (auto claim :
         {vgi::BindRepeatability::Unknown, vgi::BindRepeatability::StableWithinQuery}) {
        function->claim = claim;
        const auto bind =
            worker.call("bind", vgi::wire::ResultBuilder(gen::BindParamsSchema())
                                    .set_binary("request", vgi::wire::encode_ipc(call))
                                    .finish());
        const auto expected_claim =
            claim == vgi::BindRepeatability::Unknown ? "UNKNOWN" : "STABLE_WITHIN_QUERY";
        CHECK(vgi::wire::get_enum(bind, "repeatability") == expected_claim);
        CHECK(vgi::wire::get_string_list(bind, "projection_expression_profiles") ==
              std::vector<std::string>{vgi::SCALAR_PROJECTION_PROFILE, "vendor.scalar.v3"});
        const auto result = arrow::schema({arrow::field(
            "wide", arrow::int64(), true, arrow::key_value_metadata({"result"}, {"keep"}))});
        const auto raw = vgi::wire::base64_decode(
            vgi::generated_expression_metadata("CAST(\"n\" AS BIGINT)", {"main"}));
        const auto request = vgi::wire::ResultBuilder(gen::TableFunctionProjectionRequestSchema())
                                 .set_binary("bind_call", vgi::wire::encode_ipc(call))
                                 .set_binary("bind_result", vgi::wire::encode_ipc(bind))
                                 .set_binary_list("expressions", {raw})
                                 .set_binary("result_schema", vgi::wire::encode_schema(result))
                                 .finish();
        const auto params = vgi::wire::ResultBuilder(gen::TableFunctionProjectionParamsSchema())
                                .set_binary("request", vgi::wire::encode_ipc(request))
                                .finish();
        for (int mode = 0; mode <= 6; ++mode) {
            function->mode = mode;
            INFO(mode);
            if (mode == 0 || mode == 3) {
                auto response = worker.call("table_function_projection", params);
                if (mode == 0) {
                    auto accepted = vgi::wire::get_ipc(response, "bind_result");
                    CHECK(vgi::wire::get_enum(accepted, "repeatability") == expected_claim);
                    CHECK(vgi::wire::get_binary(accepted, "opaque_data") == "provider-state");
                    CHECK(vgi::wire::get_schema(accepted, "output_schema")
                              ->field(1)
                              ->Equals(result->field(0), true));
                } else
                    CHECK(vgi::wire::column(response, "bind_result")->IsNull(0));
            } else
                CHECK_THROWS(worker.call("table_function_projection", params));
        }
        function->mode = 0;
        for (const auto& value : {std::string(""), std::string("unknown"), std::string("ALWAYS")}) {
            auto invalid =
                vgi::wire::ResultBuilder(gen::BindResultSchema())
                    .set_binary("output_schema", vgi::wire::get_binary(bind, "output_schema"))
                    .set_string_list("projection_expression_profiles",
                                     {vgi::SCALAR_PROJECTION_PROFILE})
                    .set_enum("repeatability", value)
                    .fill_defaults()
                    .finish();
            auto bad_request = vgi::wire::ResultBuilder(gen::TableFunctionProjectionRequestSchema())
                                   .set_binary("bind_call", vgi::wire::encode_ipc(call))
                                   .set_binary("bind_result", vgi::wire::encode_ipc(invalid))
                                   .set_binary_list("expressions", {raw})
                                   .set_binary("result_schema", vgi::wire::encode_schema(result))
                                   .finish();
            CHECK_THROWS(
                worker.call("table_function_projection",
                            vgi::wire::ResultBuilder(gen::TableFunctionProjectionParamsSchema())
                                .set_binary("request", vgi::wire::encode_ipc(bad_request))
                                .finish()));
        }
    }
    // A NULL dictionary entry may retain valid enum bytes beneath its bitmap.
    // Reading the raw string alone must not turn that semantic NULL into UNKNOWN.
    auto hidden =
        vgi::wire::ResultBuilder(gen::BindResultSchema())
            .set_binary("output_schema",
                        vgi::wire::encode_schema(function->bind(vgi::BindParams{})))
            .set_string_list("projection_expression_profiles", {vgi::SCALAR_PROJECTION_PROFILE})
            .set_enum("repeatability", "UNKNOWN")
            .fill_defaults()
            .finish();
    auto dictionary = std::static_pointer_cast<arrow::DictionaryArray>(
        vgi::wire::column(hidden, "repeatability"));
    auto null_data = dictionary->dictionary()->data()->Copy();
    null_data->buffers[0] = arrow::Buffer::FromString(std::string(1, '\0'));
    null_data->SetNullCount(1);
    auto null_entry =
        arrow::DictionaryArray::FromArrays(dictionary->indices(), arrow::MakeArray(null_data))
            .ValueOrDie();
    auto hidden_columns = hidden->columns();
    hidden_columns.back() = null_entry;
    hidden = arrow::RecordBatch::Make(hidden->schema(), 1, hidden_columns);
    auto hidden_request =
        vgi::wire::ResultBuilder(gen::TableFunctionProjectionRequestSchema())
            .set_binary("bind_call", vgi::wire::encode_ipc(call))
            .set_binary("bind_result", vgi::wire::encode_ipc(hidden))
            .set_binary_list("expressions",
                             {vgi::wire::base64_decode(vgi::generated_expression_metadata(
                                 "CAST(\"n\" AS BIGINT)", {"main"}))})
            .set_binary("result_schema", vgi::wire::encode_schema(
                                             arrow::schema({arrow::field("wide", arrow::int64())})))
            .finish();
    CHECK_THROWS(worker.call("table_function_projection",
                             vgi::wire::ResultBuilder(gen::TableFunctionProjectionParamsSchema())
                                 .set_binary("request", vgi::wire::encode_ipc(hidden_request))
                                 .finish()));
    function->claim = static_cast<vgi::BindRepeatability>(99);
    CHECK_THROWS(worker.call("bind", vgi::wire::ResultBuilder(gen::BindParamsSchema())
                                         .set_binary("request", vgi::wire::encode_ipc(call))
                                         .finish()));
}
