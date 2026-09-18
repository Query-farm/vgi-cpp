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
void require_hosts_vgi(const vgi_rpc::ProtocolListing& listing) {
    std::set<std::string> hosted;
    std::string hosted_list;
    for (const auto& protocol : listing.protocols) {
        hosted.insert(protocol.protocol);
        hosted_list += " " + protocol.protocol;
    }
    INFO("hosted protocols:" << hosted_list);
    REQUIRE(hosted.count(kProtocol) == 1);
    REQUIRE(hosted.count(vgi_rpc::kReflectionProtocolName) == 1);

    const auto* application = listing.application();
    REQUIRE(application != nullptr);
    CHECK(application->protocol == kProtocol);
    CHECK(application->protocol_version == kVersion);
    CHECK_FALSE(application->protocol_hash.empty());
}

// Reflection describes exactly the VgiProtocol surface -- every method the
// generated table derives from vgi-python's VgiProtocol, and nothing else.
void require_describes_the_protocol(const vgi_rpc::ServiceDescription& description) {
    CHECK(description.protocol_name == kProtocol);
    CHECK(description.protocol_version == kVersion);
    std::set<std::string> expected;
    for (const auto& spec : vgi::protocol_methods()) expected.insert(spec.name);
    std::set<std::string> described;
    for (const auto& [name, method] : description.methods) described.insert(name);
    CHECK(described == expected);
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
    require_describes_the_protocol(client.describe(kProtocol));
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
    require_describes_the_protocol(client.describe(kProtocol));
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
    require_describes_the_protocol(client.describe(kProtocol));
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
