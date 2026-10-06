// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Sealed grants close the identity loop end to end on a built worker.
//
// `vgi_rpc.Identity.v1`'s `issue_grant` mints a credential meant to be
// presented later, by unattended automation, as an ordinary bearer.  With grant
// keys configured (`--grant-key`, or `VGI_RPC_GRANT_KEYS`) the worker is the
// minter and accepts the grant back: a vgi.v2 call carrying
// `Authorization: Bearer <grant>` runs as the grant's owner.  These tests drive
// the real example worker binary over HTTP; the owner is read back from the
// worker's own access log, which records every call's resolved principal.

#include <catch2/catch_test_macros.hpp>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <nlohmann/json.hpp>

#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/client.h>
#include <vgi_rpc/grants.h>
#include <vgi_rpc/http_client.h>
#include <vgi_rpc/token_identity.h>

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
// Bytes 0x00..0x1f, standard base64.
const std::string kGrantKey = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
const std::string kOwner = "owner@example.com";

// A worker serving HTTP, owned for the length of one test.  Ready only once it
// prints `PORT:<n>`, which it does after the bind.
class ServedWorker {
public:
    explicit ServedWorker(const std::vector<std::string>& args) {
        int out[2];
        REQUIRE(::pipe(out) == 0);
        pid_ = ::fork();
        REQUIRE(pid_ >= 0);
        if (pid_ == 0) {
            ::dup2(out[1], STDOUT_FILENO);
            ::close(out[0]);
            ::close(out[1]);
            // Grants on only where a test asks: an inherited key would turn
            // them on for the "absent" case.
            ::unsetenv("VGI_RPC_GRANT_KEYS");
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
        port_ = read_port();
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

    std::string base_url() const { return "http://127.0.0.1:" + port_; }

private:
    std::string read_port() {
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
            if (line.rfind("PORT:", 0) == 0) return line.substr(5);
            line.clear();
        }
        FAIL("worker never printed PORT:<n>");
        return {};
    }

    pid_t pid_ = -1;
    int stdout_ = -1;
    std::string port_;
};

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

std::shared_ptr<arrow::RecordBatch> no_params() {
    return arrow::RecordBatch::Make(arrow::schema({}), 1, arrow::ArrayVector{});
}

std::set<std::string> catalog_names(const std::shared_ptr<arrow::RecordBatch>& response) {
    REQUIRE(response != nullptr);
    const auto payload = vgi::wire::decode_ipc(vgi::wire::get_binary(response, "result"));
    std::set<std::string> names;
    for (const auto& item : vgi::wire::get_binary_list(payload, "items")) {
        names.insert(vgi::wire::get_string(vgi::wire::decode_ipc(item), "name"));
    }
    return names;
}

std::string now_text() {
    return std::to_string(
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count());
}

// A bearer over plain HTTP is refused by the client unless allowed; these
// tests talk to a worker on loopback.
vgi_rpc::HttpClientConfig loopback_config() {
    vgi_rpc::HttpClientConfig config;
    config.allow_insecure_credentials = true;
    return config;
}

vgi_rpc::HttpClient identity_client(
    const std::string& base_url, const std::vector<std::pair<std::string, std::string>>& headers) {
    auto builder = vgi_rpc::HttpClient::builder(base_url)
                       .config(loopback_config())
                       .protocol(vgi_rpc::kIdentityProtocolName);
    for (const auto& [name, value] : headers) builder.header(name, value);
    return builder.build();
}

vgi_rpc::HttpClient vgi_client(const std::string& base_url, const std::string& bearer) {
    return vgi_rpc::HttpClient::builder(base_url)
        .config(loopback_config())
        .protocol(kProtocol)
        .protocol_version(kVersion)
        .header("Authorization", "Bearer " + bearer)
        .build();
}

std::shared_ptr<arrow::RecordBatch> issue_grant_request() {
    arrow::StringBuilder purpose;
    VGI_RPC_THROW_NOT_OK(purpose.Append("nightly"));
    auto item = arrow::field("item", arrow::utf8(), /*nullable=*/true);
    arrow::ListBuilder scopes(arrow::default_memory_pool(),
                              std::make_shared<arrow::StringBuilder>(), arrow::list(item));
    auto* values = static_cast<arrow::StringBuilder*>(scopes.value_builder());
    VGI_RPC_THROW_NOT_OK(scopes.Append());
    VGI_RPC_THROW_NOT_OK(values->Append("read"));
    arrow::Int64Builder ttl;
    VGI_RPC_THROW_NOT_OK(ttl.Append(600));
    auto schema = arrow::schema({arrow::field("purpose", arrow::utf8(), false),
                                 arrow::field("scopes", arrow::list(item), false),
                                 arrow::field("ttl_seconds", arrow::int64(), false)});
    return arrow::RecordBatch::Make(
        schema, 1,
        {vgi_rpc::unwrap(purpose.Finish()), vgi_rpc::unwrap(scopes.Finish()),
         vgi_rpc::unwrap(ttl.Finish())});
}

// Mint as a freshly authenticated caller: the fixture's header authentication
// carries the auth_time the freshness guard requires.
std::string mint(const std::string& base_url) {
    const auto client = identity_client(
        base_url, {{"X-Conformance-Principal", kOwner}, {"X-Conformance-Auth-Time", now_text()}});
    const auto response =
        client.call("issue_grant", vgi_rpc::AnnotatedBatch::data(issue_grant_request()));
    const auto grant = vgi::wire::decode_ipc(vgi::wire::get_binary(response.batch, "result"));
    return vgi::wire::get_string(grant, "token");
}

// The access-log record of the first `vgi.v2` call to `method`.
std::optional<nlohmann::json> logged_call(const std::filesystem::path& log,
                                          const std::string& method) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream in(log);
        std::string line;
        while (std::getline(in, line)) {
            const auto record = nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false);
            if (record.is_object() && record.value("protocol", "") == kProtocol &&
                record.value("method", "") == method) {
                return std::optional<nlohmann::json>(std::in_place, record);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return std::nullopt;
}

}  // namespace

TEST_CASE("http: a minted grant authenticates a vgi.v2 call as its owner", "[grants][http]") {
    TempDir dir;
    const auto log = dir.path() / "access.jsonl";
    ServedWorker worker({"--http", "0", "--conformance-principal-header", "--grant-key", kGrantKey,
                         "--access-log", log.string()});

    const std::string token = mint(worker.base_url());
    REQUIRE(token.rfind(vgi_rpc::kGrantTokenPrefix, 0) == 0);

    const auto response = vgi_client(worker.base_url(), token)
                              .call("catalog_catalogs", vgi_rpc::AnnotatedBatch::data(no_params()));
    CHECK(catalog_names(response.batch).count("example"));

    const auto record = logged_call(log, "catalog_catalogs");
    REQUIRE(record);
    CHECK(record->value("principal", "") == kOwner);
    CHECK(record->value("auth_domain", "") == "grant");
    CHECK(record->value("authenticated", false));
}

TEST_CASE("http: a grant cannot mint a grant", "[grants][http]") {
    ServedWorker worker(
        {"--http", "0", "--conformance-principal-header", "--grant-key", kGrantKey});
    const std::string token = mint(worker.base_url());
    const auto client = identity_client(worker.base_url(), {{"Authorization", "Bearer " + token}});
    try {
        client.call("issue_grant", vgi_rpc::AnnotatedBatch::data(issue_grant_request()));
        FAIL("a grant-authenticated caller minted a grant");
    } catch (const vgi_rpc::RpcRemoteError& refused) {
        CHECK(refused.error_kind() == "stale_auth");
    }
}

TEST_CASE("http: a tampered grant is refused 401", "[grants][http]") {
    ServedWorker worker(
        {"--http", "0", "--conformance-principal-header", "--grant-key", kGrantKey});
    std::string token = mint(worker.base_url());
    const size_t mid = token.size() - 10;
    token[mid] = token[mid] == 'A' ? 'B' : 'A';
    try {
        vgi_client(worker.base_url(), token)
            .call("catalog_catalogs", vgi_rpc::AnnotatedBatch::data(no_params()));
        FAIL("a tampered grant was accepted");
    } catch (const vgi_rpc::HttpClientError& refused) {
        CHECK(refused.http_status() == 401);
    }
}

TEST_CASE("http: without grant keys nothing changes", "[grants][http]") {
    ServedWorker worker({"--http", "0", "--conformance-principal-header"});
    const auto listing = vgi_rpc::HttpClient::builder(worker.base_url())
                             .protocol(kProtocol)
                             .protocol_version(kVersion)
                             .build()
                             .list_protocols();
    for (const auto& protocol : listing.protocols) {
        CHECK(protocol.protocol != vgi_rpc::kIdentityProtocolName);
    }
}

TEST_CASE("a malformed grant key refuses to start the worker", "[grants]") {
    const std::string command = "'" + kWorker + "' --http 0 --grant-key not-a-key >/dev/null 2>&1";
    const int status = std::system(command.c_str());
    REQUIRE(status != -1);
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) != 0);
}
