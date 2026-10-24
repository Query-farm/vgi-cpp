// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// A built worker answers on the VGI wire name, on every transport it serves.
//
// Since vgi-rpc 0.46 every request names the protocol it addresses, and the
// server dispatches on the pair (protocol, method): the `vgi_rpc.protocol`
// metadata key on the raw transports, and also the path segment over HTTP
// (`{prefix}/{protocol}/{method}`). The name is therefore a cross-port wire
// contract -- `vgi.v2`, which the DuckDB extension sends -- and a worker that
// declares any other name answers ProtocolNotSupported to every real client.
//
// These tests drive the real example worker binary with vgi-rpc's own
// clients. Nothing here builds a response: the answers are the worker's.
// The constant being right and the *server* announcing it are different
// facts, which is why they talk to a running worker rather than reading the
// generated header back.

#include <catch2/catch_test_macros.hpp>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <set>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <vgi_rpc/client.h>
#include <vgi_rpc/http_client.h>
#include <vgi_rpc/metadata.h>

#include "methods.h"
#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"
#include "wire.h"

#ifndef VGI_EXAMPLE_WORKER_PATH
#error "VGI_EXAMPLE_WORKER_PATH must name the built vgi-example-worker"
#endif

namespace {

namespace gen = ::vgi::generated;

const std::string kWorker = VGI_EXAMPLE_WORKER_PATH;
const std::string kProtocol(gen::VGI_PROTOCOL_NAME);
const std::string kVersion(gen::VGI_PROTOCOL_VERSION);
// An incompatible major is a *different* name, and so a routing refusal rather
// than a handler failing somewhere inside a payload it cannot read.
const std::string kOtherMajor = "vgi.v1";

// The reference vgi.v2 protocol hash: the unpublished combined contract, 74 methods. The
// hash covers each method's type and its params, result and header schemas
// (field names, order, nullability, types), not docs or defaults, so a
// registration that drifts from the reference changes it. Every SDK reports
// this same value. This is the changing development contract; the public
// version transition remains gated on the complete combined protocol freeze.
const std::string kReferenceProtocolHash =
    "41f61f37c7ce42f63f3ca47f8861508f3f356d2422a5b8a0bab48ff92c0857a8";

// A worker serving a listening transport, owned for the length of one test.
//
// Started with fork/exec rather than a shell, and ready only once it prints
// the line that names where it is listening (`PORT:<n>`, `UNIX:<path>`): that
// line is printed after the bind, so it is the only race-free signal.
class ServedWorker {
public:
    ServedWorker(const std::vector<std::string>& args, const std::string& ready_prefix) {
        int out[2];
        REQUIRE(::pipe(out) == 0);
        pid_ = ::fork();
        REQUIRE(pid_ >= 0);
        if (pid_ == 0) {
            ::dup2(out[1], STDOUT_FILENO);
            ::close(out[0]);
            ::close(out[1]);
            std::vector<std::string> owned{kWorker};
            owned.insert(owned.end(), args.begin(), args.end());
            std::vector<char*> argv;
            for (auto& arg : owned) argv.push_back(arg.data());
            argv.push_back(nullptr);
            ::execv(kWorker.c_str(), argv.data());
            ::_exit(127);
        }
        ::close(out[1]);
        stdout_ = out[0];
        ready_ = read_ready_line(ready_prefix);
    }

    ~ServedWorker() {
        if (pid_ > 0) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, nullptr, 0);
        }
        if (stdout_ >= 0) ::close(stdout_);
    }

    ServedWorker(const ServedWorker&) = delete;
    ServedWorker& operator=(const ServedWorker&) = delete;

    // What followed the ready prefix: the port, or the socket path.
    const std::string& ready() const { return ready_; }

private:
    std::string read_ready_line(const std::string& prefix) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        std::string line;
        while (std::chrono::steady_clock::now() < deadline) {
            pollfd pfd{stdout_, POLLIN, 0};
            if (::poll(&pfd, 1, 100) <= 0) continue;
            char c = 0;
            const auto got = ::read(stdout_, &c, 1);
            REQUIRE(got == 1);  // EOF: the worker exited before binding
            if (c != '\n') {
                line.push_back(c);
                continue;
            }
            if (line.rfind(prefix, 0) == 0) return line.substr(prefix.size());
            line.clear();
        }
        FAIL("worker never printed a line starting with '" << prefix << "'");
        return {};
    }

    pid_t pid_ = -1;
    int stdout_ = -1;
    std::string ready_;
};

// A private directory for a unix socket, removed afterwards. Short, because a
// socket path is capped near 104 bytes and a CI workspace path is not.
class TempDir {
public:
    TempDir() {
        auto pattern = (std::filesystem::temp_directory_path() / "vgi-cpp-XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        path_ = pattern;
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// `catalog_catalogs` takes no parameters: one row, no columns.
std::shared_ptr<arrow::RecordBatch> no_params() {
    return arrow::RecordBatch::Make(arrow::schema({}), 1, arrow::ArrayVector{});
}

// The catalog names a `catalog_catalogs` answer lists. Reading it through
// means the call reached the VGI handler, not merely the routing layer.
std::set<std::string> catalog_names(const std::shared_ptr<arrow::RecordBatch>& response) {
    REQUIRE(response != nullptr);
    const auto payload = vgi::wire::decode_ipc(vgi::wire::get_binary(response, "result"));
    std::set<std::string> names;
    for (const auto& item : vgi::wire::get_binary_list(payload, "items")) {
        names.insert(vgi::wire::get_string(vgi::wire::decode_ipc(item), "name"));
    }
    return names;
}

// What `list_protocols` reports is what the routing key is matched against.
void require_hosts_vgi(const std::vector<vgi_rpc::HostedProtocol>& listing) {
    std::set<std::string> hosted;
    std::string hosted_list;
    for (const auto& protocol : listing) {
        hosted.insert(protocol.name);
        hosted_list += " " + protocol.name;
    }
    INFO("hosted protocols:" << hosted_list);
    REQUIRE(hosted.count(kProtocol) == 1);
    REQUIRE(hosted.count(vgi_rpc::kReflectionProtocolName) == 1);

    // The application protocol leads the server's order.
    REQUIRE_FALSE(listing.empty());
    const auto& application = listing.front();
    CHECK(application.name == kProtocol);
    CHECK(application.version == kVersion);
    // Byte-identical to the reference surface, on every transport.
    CHECK(kVersion == "2.2.0");
    CHECK(application.hash == kReferenceProtocolHash);
}

// Reflection describes exactly the VgiProtocol surface -- every method the
// generated table derives from vgi-python's VgiProtocol, and nothing else --
// and describes it as the reference does. Nullability and field order are part
// of an Arrow type, so they are part of the description and of its hash: a
// `result` column is nullable only for a `bytes | None` return, and the init
// header is the generated GlobalInitResponse, field for field.
void require_describes_the_protocol(const vgi_rpc::ServiceDescription& description) {
    CHECK(description.protocol_name == kProtocol);
    CHECK(description.protocol_version == kVersion);
    std::set<std::string> expected;
    for (const auto& spec : vgi::protocol_methods()) expected.insert(spec.name);
    std::set<std::string> described;
    for (const auto& [name, method] : description.methods) described.insert(name);
    CHECK(described == expected);

    for (const auto& spec : vgi::protocol_methods()) {
        if (spec.kind == vgi::MethodKind::Void || spec.kind == vgi::MethodKind::Stream) continue;
        const auto* method = description.method(spec.name);
        REQUIRE(method != nullptr);
        INFO(spec.name << " declares result " << method->result_schema->ToString());
        REQUIRE(method->result_schema->num_fields() == 1);
        CHECK(method->result_schema->field(0)->nullable() == spec.optional_result);
    }
    const auto* init = description.method("init");
    REQUIRE(init != nullptr);
    REQUIRE(init->header_schema != nullptr);
    CHECK(init->header_schema->Equals(*gen::GlobalInitResponseSchema()));
}

vgi_rpc::RpcClientOptions raw_options(const std::string& protocol) {
    vgi_rpc::RpcClientOptions options;
    options.protocol = protocol;
    options.protocol_version = kVersion;
    return options;
}

// A raw transport carries the name only as metadata, so a request naming a
// protocol the worker does not host is refused before any handler runs.
void require_raw_refuses(const std::function<vgi_rpc::RpcClient()>& connect) {
    auto client = connect();
    try {
        client.call_unary("catalog_catalogs", no_params());
        FAIL("a request naming " << kOtherMajor << " was served");
    } catch (const vgi_rpc::RpcException& refused) {
        CHECK(refused.error_kind() == vgi_rpc::ERROR_KIND_PROTOCOL_NOT_SUPPORTED);
    }
}

}  // namespace

TEST_CASE("stdio: the worker hosts the VGI wire name", "[protocol][stdio]") {
    auto client = vgi_rpc::RpcClient::spawn({kWorker}, raw_options(kProtocol));
    require_hosts_vgi(client.list_protocols());
    require_describes_the_protocol(client.describe_protocol(kProtocol));
    CHECK(catalog_names(client.call_unary("catalog_catalogs", no_params()).batch).count("example"));
}

TEST_CASE("stdio: another major is a different protocol", "[protocol][stdio]") {
    require_raw_refuses(
        [] { return vgi_rpc::RpcClient::spawn({kWorker}, raw_options(kOtherMajor)); });
}

TEST_CASE("unix: the worker hosts the VGI wire name", "[protocol][unix]") {
    TempDir dir;
    const auto socket = (dir.path() / "w.sock").string();
    ServedWorker worker({"--unix", socket}, "UNIX:");

    auto client = vgi_rpc::RpcClient::connect_unix(socket, raw_options(kProtocol));
    require_hosts_vgi(client.list_protocols());
    require_describes_the_protocol(client.describe_protocol(kProtocol));
    CHECK(catalog_names(client.call_unary("catalog_catalogs", no_params()).batch).count("example"));
    client.close();

    require_raw_refuses(
        [&] { return vgi_rpc::RpcClient::connect_unix(socket, raw_options(kOtherMajor)); });
}

TEST_CASE("http: the worker serves {prefix}/vgi.v2/{method}", "[protocol][http]") {
    ServedWorker worker({"--http", "0"}, "PORT:");
    const auto base_url = "http://127.0.0.1:" + worker.ready();

    const auto client = vgi_rpc::HttpClient::builder(base_url)
                            .protocol(kProtocol)
                            .protocol_version(kVersion)
                            .build();
    require_hosts_vgi(client.list_protocols());
    require_describes_the_protocol(client.describe_protocol(kProtocol));
    const auto response =
        client.call("catalog_catalogs", vgi_rpc::AnnotatedBatch::data(no_params()));
    CHECK(catalog_names(response.batch).count("example"));

    // The path segment is the name, so the other major is simply not a route
    // on this server: a 404 an edge device can read without an Arrow parser.
    const auto other = vgi_rpc::HttpClient::builder(base_url)
                           .protocol(kOtherMajor)
                           .protocol_version(kVersion)
                           .build();
    try {
        other.call("catalog_catalogs", vgi_rpc::AnnotatedBatch::data(no_params()));
        FAIL("a request routed to " << kOtherMajor << " was served");
    } catch (const vgi_rpc::HttpClientError& refused) {
        CHECK(refused.http_status() == 404);
    }
}

TEST_CASE("example constraint discovery preserves validated immutable snapshots",
          "[catalog][constraints][stdio]") {
    auto client = vgi_rpc::RpcClient::spawn({kWorker}, raw_options(kProtocol));
    const auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                             .set_string("name", "example")
                             .fill_defaults()
                             .finish();
    const auto attached = client.call_unary(
        "catalog_attach", vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                              .set_binary("request", vgi::wire::encode_ipc(request))
                              .finish());
    const auto attach = vgi::wire::get_binary(
        vgi::wire::decode_ipc(vgi::wire::get_binary(attached.batch, "result")),
        "attach_opaque_data");
    const auto read = [&](const std::string& name, int version = 0) {
        auto params = vgi::wire::ResultBuilder(gen::CatalogTableGetParamsSchema());
        params.set_binary("attach_opaque_data", attach)
            .set_string_list("schema_path", {"data"})
            .set_string("name", name);
        if (version)
            params.set_string("at_unit", "VERSION").set_string("at_value", std::to_string(version));
        const auto response =
            client.call_unary("catalog_table_get", params.fill_defaults().finish());
        const auto payload = vgi::wire::decode_ipc(vgi::wire::get_binary(response.batch, "result"));
        REQUIRE(payload != nullptr);
        const auto items = vgi::wire::get_binary_list(payload, "items");
        REQUIRE(items.size() == 1);
        return vgi::wire::decode_ipc(items.front());
    };
    for (const auto& [name, count] :
         std::vector<std::pair<std::string, size_t>>{{"departments", 5},
                                                     {"products", 2},
                                                     {"employees", 6},
                                                     {"projects", 5},
                                                     {"versioned_constraints", 5}}) {
        const auto table = read(name);
        REQUIRE(table != nullptr);
        REQUIRE(table->schema()->Equals(*gen::TableInfoSchema(), true));
        const auto columns = vgi::wire::get_schema(table, "columns");
        REQUIRE(columns != nullptr);
        const std::map<std::string, std::set<std::string>> required = {
            {"departments", {"id", "name"}},
            {"products", {"id"}},
            {"employees", {"id", "name", "email"}},
            {"projects", {"department_id", "project_code", "title"}},
            {"versioned_constraints", {"id", "name"}}};
        for (const auto& field : columns->fields())
            CHECK(field->nullable() == !required.at(name).contains(field->name()));
        const auto constraints = vgi::wire::get_binary_list(table, "constraints");
        REQUIRE(constraints.size() == count);
        std::set<std::string> keys;
        for (const auto& encoded : constraints) {
            const auto info = vgi::wire::decode_ipc(encoded);
            REQUIRE(info->schema()->Equals(*gen::ConstraintInfoSchema(), true));
            CHECK(vgi::wire::get_enum(info, "enforcement") == "ENFORCED");
            CHECK(vgi::wire::get_enum(info, "validation") == "ALL_ROWS");
            const auto ref = vgi::wire::get_ipc(info, "ref");
            CHECK(vgi::wire::get_string(ref, "owner_table") == name);
            CHECK(keys.insert(vgi::wire::get_string(ref, "name")).second);
            const auto definition = vgi::wire::get_ipc(info, "definition");
            if (vgi::wire::get_enum(definition, "kind") == "FOREIGN_KEY") {
                const auto target = vgi::wire::get_ipc(info, "referenced_constraint");
                CHECK(vgi::wire::get_string(target, "owner_table") == "departments");
                CHECK(vgi::wire::get_string(target, "name") == "fixture:departments:primary");
            }
        }
    }
    std::vector<std::string> previous;
    for (int version = 1; version <= 3; ++version) {
        const auto table = read("versioned_constraints", version);
        const auto constraints = vgi::wire::get_binary_list(table, "constraints");
        REQUIRE(constraints.size() == static_cast<size_t>(version == 1 ? 1 : version + 2));
        const auto columns = vgi::wire::get_schema(table, "columns");
        CHECK_FALSE(columns->GetFieldByName("id")->nullable());
        CHECK(columns->GetFieldByName("name")->nullable() == (version == 1));
        for (size_t i = 0; i < previous.size(); ++i) CHECK(previous[i] == constraints[i]);
        previous = constraints;
    }
}
