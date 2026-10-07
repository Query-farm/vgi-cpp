// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Sealed opaque values on the built example worker, over HTTP: the checks of
// vgi-python's opaque-data-sealing conformance group. A value attached as one
// principal and replayed as another, a value with one flipped byte, and the
// unsealed shape of a value are all the identical `not recognized`; a
// transaction replayed under a different attach is refused; and neither the
// secret attach option nor the raw value reaches the worker's log.

#include <catch2/catch_test_macros.hpp>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/http_client.h>

#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"
#include "wire.h"

#ifndef VGI_EXAMPLE_WORKER_PATH
#error "VGI_EXAMPLE_WORKER_PATH must name the built vgi-example-worker"
#endif

using namespace std::string_literals;

namespace {

namespace gen = ::vgi::generated;

const std::string kWorker = VGI_EXAMPLE_WORKER_PATH;
const std::string kProtocol(gen::VGI_PROTOCOL_NAME);
const std::string kVersion(gen::VGI_PROTOCOL_VERSION);
const std::string kSecret = "sk-explicit-must-not-leak";

// The worker over HTTP, its stderr (with VGI_TRACE on) captured to a file.
class ServedWorker {
public:
    explicit ServedWorker(std::string log) : log_(std::move(log)) {
        int out[2];
        REQUIRE(::pipe(out) == 0);
        pid_ = ::fork();
        REQUIRE(pid_ >= 0);
        if (pid_ == 0) {
            ::dup2(out[1], STDOUT_FILENO);
            const int err = ::open(log_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (err >= 0) ::dup2(err, STDERR_FILENO);
            ::close(out[0]);
            ::close(out[1]);
            ::setenv("VGI_TRACE", "1", 1);
            ::unsetenv("VGI_SIGNING_KEY");
            ::unsetenv("VGI_RPC_GRANT_KEYS");
            ::unsetenv("VGI_WORKER_CATALOG_NAME");
            std::vector<std::string> owned{kWorker, "--http", "0",
                                           "--conformance-principal-header"};
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
    std::string log() const {
        std::ifstream in(log_, std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        return text.str();
    }

private:
    std::string read_port() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        std::string line;
        while (std::chrono::steady_clock::now() < deadline) {
            pollfd pfd{stdout_, POLLIN, 0};
            if (::poll(&pfd, 1, 100) <= 0) continue;
            char c = 0;
            REQUIRE(::read(stdout_, &c, 1) == 1);
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

    std::string log_;
    pid_t pid_ = -1;
    int stdout_ = -1;
    std::string port_;
};

std::string now_text() {
    return std::to_string(
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count());
}

vgi_rpc::HttpClient as(const std::string& url, const std::string& principal) {
    auto builder = vgi_rpc::HttpClient::builder(url).protocol(kProtocol).protocol_version(kVersion);
    if (!principal.empty()) {
        builder.header("X-Conformance-Principal", principal);
        builder.header("X-Conformance-Auth-Time", now_text());
    }
    return builder.build();
}

std::shared_ptr<arrow::RecordBatch> payload(const vgi_rpc::AnnotatedBatch& response) {
    return vgi::wire::decode_ipc(vgi::wire::get_binary(response.batch, "result"));
}

std::string attach(const vgi_rpc::HttpClient& client) {
    arrow::StringBuilder api_key, region;
    VGI_RPC_THROW_NOT_OK(api_key.Append(kSecret));
    VGI_RPC_THROW_NOT_OK(region.Append("eu-west-1"));
    std::shared_ptr<arrow::Array> a = vgi_rpc::unwrap(api_key.Finish());
    std::shared_ptr<arrow::Array> r = vgi_rpc::unwrap(region.Finish());
    const auto options =
        arrow::RecordBatch::Make(arrow::schema({arrow::field("api_key", arrow::utf8()),
                                                arrow::field("region", arrow::utf8())}),
                                 1, {a, r});
    const auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                             .set_string("name", "attach_options_required")
                             .set_binary("options", vgi::wire::encode_ipc(options))
                             .fill_defaults()
                             .finish();
    const auto params = vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                            .set_binary("request", vgi::wire::encode_ipc(request))
                            .finish();
    return vgi::wire::get_binary(
        payload(client.call("catalog_attach", vgi_rpc::AnnotatedBatch::data(params))),
        "attach_opaque_data");
}

vgi_rpc::AnnotatedBatch schemas(const std::string& attach) {
    return vgi_rpc::AnnotatedBatch::data(vgi::wire::ResultBuilder(gen::CatalogSchemasParamsSchema())
                                             .set_binary("attach_opaque_data", attach)
                                             .fill_defaults()
                                             .finish());
}

std::string begin(const vgi_rpc::HttpClient& client, const std::string& attach) {
    const auto params = vgi::wire::ResultBuilder(gen::CatalogTransactionBeginParamsSchema())
                            .set_binary("attach_opaque_data", attach)
                            .fill_defaults()
                            .finish();
    return vgi::wire::get_binary(
        payload(client.call("catalog_transaction_begin", vgi_rpc::AnnotatedBatch::data(params))),
        "transaction_opaque_data");
}

vgi_rpc::AnnotatedBatch commit(const std::string& attach, const std::string& transaction) {
    return vgi_rpc::AnnotatedBatch::data(
        vgi::wire::ResultBuilder(gen::CatalogTransactionCommitParamsSchema())
            .set_binary("attach_opaque_data", attach)
            .set_binary("transaction_opaque_data", transaction)
            .fill_defaults()
            .finish());
}

// `type|code|message` of the refusal, or "<accepted>".
std::string refusal(const vgi_rpc::HttpClient& client, const std::string& method,
                    const vgi_rpc::AnnotatedBatch& request) {
    try {
        (void)client.call(method, request);
    } catch (const vgi_rpc::RpcRemoteError& error) {
        return error.exception_type() + "|" + error.error_code() + "|" + error.error_kind() + "|" +
               error.what() + "|" + std::to_string(error.error_details().size());
    }
    return "<accepted>";
}

std::string hex(const std::string& bytes) {
    return vgi_rpc::crypto::hex_encode(reinterpret_cast<const uint8_t*>(bytes.data()),
                                       bytes.size());
}

std::string short_hash(const std::string& bytes) {
    const auto text = hex(bytes);
    const auto digest =
        vgi_rpc::crypto::sha256(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    return vgi_rpc::crypto::hex_encode(digest.data(), digest.size()).substr(0, 12);
}

const std::string kAttachRejected =
    "ValueError|INVALID_ARGUMENT|opaque_data_not_recognized|attach_opaque_data not recognized|0";

}  // namespace

TEST_CASE("http: opaque values are sealed, bound and refused uniformly", "[opaque][http]") {
    const auto log = (std::filesystem::temp_directory_path() /
                      ("vgi-opaque-" + std::to_string(::getpid()) + ".log"))
                         .string();
    ServedWorker worker(log);
    const auto alice = as(worker.base_url(), "alice");
    const auto bob = as(worker.base_url(), "bob");

    const auto value = attach(alice);
    CHECK(value.find(kSecret) == std::string::npos);
    CHECK(value.find("attach_options_required") == std::string::npos);
    CHECK_NOTHROW(alice.call("catalog_schemas", schemas(value)));

    // Replayed as another principal, and as nobody.
    const auto replayed = refusal(bob, "catalog_schemas", schemas(value));
    CHECK(replayed == kAttachRejected);
    CHECK(refusal(as(worker.base_url(), ""), "catalog_schemas", schemas(value)) == kAttachRejected);
    // One flipped byte.
    std::string tampered = value;
    tampered[tampered.size() / 2] ^= 0x01;
    const auto flipped = refusal(alice, "catalog_schemas", schemas(tampered));
    CHECK(flipped == kAttachRejected);
    // The shape this worker produces unsealed: never read as plaintext.
    const auto plaintext =
        refusal(alice, "catalog_schemas",
                schemas("attach_options_required\0\0attach_options_required\0deadbeef\0"s));
    CHECK(plaintext == kAttachRejected);
    CHECK(replayed == flipped);
    CHECK(flipped == plaintext);

    // A transaction is bound to the attach it was begun under.
    const auto other = attach(alice);
    const auto transaction = begin(alice, value);
    CHECK_NOTHROW(alice.call("catalog_transaction_commit", commit(value, transaction)));
    const auto crossed = refusal(alice, "catalog_transaction_commit", commit(other, transaction));
    CHECK(crossed ==
          "ValueError|INVALID_ARGUMENT|opaque_data_not_recognized|transaction_opaque_data not "
          "recognized|0");
    CHECK(refusal(bob, "catalog_transaction_commit", commit(value, transaction)) ==
          kAttachRejected);

    // The log carries the short hash of the value, never the value, its hex,
    // or the secret it was made from.
    const auto text = worker.log();
    CHECK(text.find("attach=" + short_hash(value)) != std::string::npos);
    CHECK(text.find(kSecret) == std::string::npos);
    CHECK(text.find(value) == std::string::npos);
    CHECK(text.find(hex(value)) == std::string::npos);
    CHECK(text.find(hex(value).substr(0, 32)) == std::string::npos);
    std::filesystem::remove(log);
}
