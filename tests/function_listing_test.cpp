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

#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
