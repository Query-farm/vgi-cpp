// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Sealing `attach_opaque_data` / `transaction_opaque_data`
// (vgi-python docs/protocol/vgi-opaque-data-sealing.md): the envelope and its
// AADs, the boundary that opens every value before a handler runs, and a real
// Dispatcher in-process -- unsealed (the OS-owned transports) and sealed.

#include <catch2/catch_test_macros.hpp>

#include <sys/socket.h>
#include <unistd.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/client.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/errors.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/wire.h>

#include "dispatcher.h"
#include "opaque_seal.h"
#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"
#include "wire.h"

using namespace std::string_literals;

namespace {

namespace gen = ::vgi::generated;
namespace opaque = ::vgi::opaque;

const std::string kProtocol(gen::VGI_PROTOCOL_NAME);
const std::string kVersion(gen::VGI_PROTOCOL_VERSION);
const std::string kSecret = "sk-must-never-appear-in-plaintext";

vgi_rpc::AuthContext principal(const std::string& name, const std::string& domain = "bearer") {
    vgi_rpc::AuthContext auth;
    auth.domain = domain;
    auth.authenticated = true;
    auth.principal = name;
    return auth;
}

opaque::Key key() {
    return vgi_rpc::crypto::random_key();
}

// The error a call raised, as a client would see it: type, code, message.
template <typename F>
std::string error_of(F&& call) {
    try {
        call();
    } catch (const vgi_rpc::KindedError& error) {
        return error.exception_type() + "|" + vgi_rpc::code_name(error.code()) + "|" +
               error.kind() + "|" + error.what() + "|" + std::to_string(error.details().size());
    }
    return "<accepted>";
}

const std::string kAttachRejected =
    "ValueError|INVALID_ARGUMENT|opaque_data_not_recognized|attach_opaque_data not recognized|0";
const std::string kTransactionRejected =
    "ValueError|INVALID_ARGUMENT|opaque_data_not_recognized|transaction_opaque_data not "
    "recognized|0";

// What a table function's bind was handed, for the last bind.
std::mutex g_seen_mutex;
std::shared_ptr<arrow::RecordBatch> g_seen_options;

class OptionsProbe final : public vgi::TableFunction {
public:
    std::string name() const override { return "options_probe"; }
    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "records the attach options its bind sees";
        return md;
    }
    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        std::lock_guard<std::mutex> lock(g_seen_mutex);
        g_seen_options = params.attach_options;
        return arrow::schema({arrow::field("x", arrow::int64())});
    }
    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams&) const override {
        return nullptr;
    }
};

// A catalog with one secret and one plain attach option, and the probe.
void declare(vgi::Dispatcher& dispatcher) {
    vgi::CatalogModel model;
    model.name = "vault";
    vgi::AttachOptionSpec api_key{"api_key", "credential", arrow::utf8(), nullptr, true};
    api_key.secret = true;
    vgi::AttachOptionSpec region{"region", "plain", arrow::utf8(), nullptr, false};
    model.attach_options = {api_key, region};
    dispatcher.set_catalog(std::move(model));
    dispatcher.register_table(std::make_shared<OptionsProbe>());
}

// A Dispatcher served on its own thread over a socket pair: an OS-owned
// transport, anonymous callers.
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
    // The remote error a call raised, as `type|code|message`.
    std::string refused(const std::string& method,
                        const std::shared_ptr<arrow::RecordBatch>& params) {
        try {
            (void)client_->call_unary(method, params);
        } catch (const vgi_rpc::RpcException& error) {
            return error.exception_type() + "|" + error.error_code() + "|" + error.error_kind() +
                   "|" + error.what() + "|" + std::to_string(error.error_details().size());
        }
        return "<accepted>";
    }

private:
    std::unique_ptr<vgi_rpc::Server> server_;
    int fds_[2] = {-1, -1};
    std::thread thread_;
    std::optional<vgi_rpc::RpcClient> client_;
};

std::string options_ipc() {
    arrow::StringBuilder api_key, region;
    REQUIRE(api_key.Append(kSecret).ok());
    REQUIRE(region.Append("eu-west-1").ok());
    std::shared_ptr<arrow::Array> a = vgi_rpc::unwrap(api_key.Finish());
    std::shared_ptr<arrow::Array> r = vgi_rpc::unwrap(region.Finish());
    return vgi::wire::encode_ipc(
        arrow::RecordBatch::Make(arrow::schema({arrow::field("api_key", arrow::utf8()),
                                                arrow::field("region", arrow::utf8())}),
                                 1, {a, r}));
}

std::string attach(Served& worker) {
    const auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                             .set_string("name", "vault")
                             .set_binary("options", options_ipc())
                             .fill_defaults()
                             .finish();
    const auto params = vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                            .set_binary("request", vgi::wire::encode_ipc(request))
                            .finish();
    return vgi::wire::get_binary(worker.call("catalog_attach", params), "attach_opaque_data");
}

std::shared_ptr<arrow::RecordBatch> schemas_params(const std::string& attach) {
    return vgi::wire::ResultBuilder(gen::CatalogSchemasParamsSchema())
        .set_binary("attach_opaque_data", attach)
        .fill_defaults()
        .finish();
}

std::shared_ptr<arrow::RecordBatch> bind_params(const std::string& attach) {
    const auto bind_call = vgi::wire::ResultBuilder(gen::BindRequestSchema())
                               .set_string("function_name", "options_probe")
                               .set_binary("arguments", "")
                               .set_enum("function_type", "table")
                               .set_binary("attach_opaque_data", attach)
                               .set_bool("resolved_secrets_provided", false)
                               .set_string_list("schema_path", {"main"})
                               .fill_defaults()
                               .finish();
    return vgi::wire::ResultBuilder(gen::BindParamsSchema())
        .set_binary("request", vgi::wire::encode_ipc(bind_call))
        .finish();
}

std::string seen(const std::string& option) {
    std::lock_guard<std::mutex> lock(g_seen_mutex);
    REQUIRE(g_seen_options != nullptr);
    return vgi::wire::get_string(g_seen_options, option);
}

}  // namespace

TEST_CASE("an attach value opens only for the caller it was sealed for", "[opaque]") {
    const auto k = key();
    const auto alice = principal("alice");
    const auto sealed = opaque::seal_attach(k, "plaintext", alice);
    CHECK(opaque::open_attach(k, sealed, alice) == "plaintext");

    CHECK(error_of([&] { (void)opaque::open_attach(k, sealed, principal("bob")); }) ==
          kAttachRejected);
    // Same principal, another authenticator: a different caller.
    CHECK(error_of([&] { (void)opaque::open_attach(k, sealed, principal("alice", "jwt")); }) ==
          kAttachRejected);
    CHECK(error_of([&] {
              (void)opaque::open_attach(k, sealed, vgi_rpc::AuthContext::anonymous());
          }) == kAttachRejected);
    std::string tampered = sealed;
    tampered[tampered.size() / 2] ^= 0x01;
    CHECK(error_of([&] { (void)opaque::open_attach(k, tampered, alice); }) == kAttachRejected);
    CHECK(error_of([&] { (void)opaque::open_attach(key(), sealed, alice); }) == kAttachRejected);
    CHECK(error_of([&] { (void)opaque::open_attach(k, "vault\0\0vault\0id\0"s, alice); }) ==
          kAttachRejected);
    CHECK(error_of([&] { (void)opaque::open_attach(k, "", alice); }) == kAttachRejected);

    const auto anonymous = vgi_rpc::AuthContext::anonymous();
    CHECK(opaque::open_attach(k, opaque::seal_attach(k, "p", anonymous), anonymous) == "p");
}

TEST_CASE("a transaction value is bound to its caller and its parent attach", "[opaque]") {
    const auto k = key();
    const auto alice = principal("alice");
    const auto attach_a = opaque::seal_attach(k, "a", alice);
    const auto attach_b = opaque::seal_attach(k, "b", alice);
    const auto txn = opaque::seal_transaction(k, "txn-1", alice, attach_a);
    CHECK(opaque::open_transaction(k, txn, alice, attach_a) == "txn-1");
    CHECK(error_of([&] { (void)opaque::open_transaction(k, txn, alice, attach_b); }) ==
          kTransactionRejected);
    CHECK(error_of([&] { (void)opaque::open_transaction(k, txn, principal("bob"), attach_a); }) ==
          kTransactionRejected);
}

TEST_CASE("the boundary opens nested values and refuses what it cannot open", "[opaque]") {
    const auto k = key();
    const auto alice = principal("alice");
    const auto sealed = opaque::seal_attach(k, "vault\0\0vault\0id\0"s, alice);
    const auto params = bind_params(sealed);

    opaque::Call call(k, alice);
    const auto opened = call.open(vgi_rpc::Request(params, nullptr));
    const auto bind_call = vgi::wire::get_ipc(opened.batch(), "request");
    CHECK(vgi::wire::get_binary(bind_call, "attach_opaque_data") == "vault\0\0vault\0id\0"s);
    REQUIRE(call.attach_as_sent());
    CHECK(*call.attach_as_sent() == sealed);

    opaque::Call bob(k, principal("bob"));
    CHECK(error_of([&] { (void)bob.open(vgi_rpc::Request(params, nullptr)); }) == kAttachRejected);
    // The unsealed shape of the value, on a sealing transport: refused.
    CHECK(error_of([&] {
              (void)bob.open(vgi_rpc::Request(bind_params("vault\0\0vault\0id\0"s), nullptr));
          }) == kAttachRejected);
}

TEST_CASE("the short hash is 12 hex of SHA-256 over the hex text, never the value", "[opaque]") {
    CHECK(opaque::short_hash("abc") == "86900f25bd2e");
    CHECK(opaque::random_id().size() == 32);
    CHECK(opaque::random_id() != opaque::random_id());
}

TEST_CASE("unsealed, an attach value never carries a secret option", "[opaque][dispatcher]") {
    vgi::Dispatcher dispatcher;
    declare(dispatcher);
    Served worker(dispatcher);

    const auto value = attach(worker);
    CHECK(value.find(kSecret) == std::string::npos);
    CHECK(vgi::wire::base64_decode(value.substr(value.rfind('\0') + 1)).find(kSecret) ==
          std::string::npos);
    // Kept server-side, and back for the function that needs it.
    (void)worker.call("bind", bind_params(value));
    CHECK(seen("api_key") == kSecret);
    CHECK(seen("region") == "eu-west-1");
}

TEST_CASE("sealed, the value is opaque and opens only as sent", "[opaque][dispatcher]") {
    vgi::Dispatcher dispatcher;
    declare(dispatcher);
    dispatcher.set_opaque_key(key());
    Served worker(dispatcher);

    const auto value = attach(worker);
    CHECK(value.find(kSecret) == std::string::npos);
    CHECK(value.find("vault") == std::string::npos);
    CHECK(static_cast<uint8_t>(value[0]) == opaque::kAttachEnvelopeVersion);
    (void)worker.call("bind", bind_params(value));
    CHECK(seen("api_key") == kSecret);
    (void)worker.call("catalog_schemas", schemas_params(value));

    std::string tampered = value;
    tampered[tampered.size() - 3] ^= 0x01;
    CHECK(worker.refused("catalog_schemas", schemas_params(tampered)) == kAttachRejected);
    CHECK(worker.refused("bind", bind_params(tampered)) == kAttachRejected);
    CHECK(worker.refused("catalog_schemas", schemas_params("vault\0\0vault\0id\0"s)) ==
          kAttachRejected);
    CHECK(worker.refused("catalog_schemas", schemas_params(std::string())) == kAttachRejected);
}
