// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Every result pointer a vgi worker publishes carries its digest.
//
// vgi-rpc lets a server publish a result pointer without
// `vgi_rpc.location.sha256` (`publish_external(..., include_sha256 = false)`,
// or an `ExternalRef` built without one), and its conformance suite keeps that
// path tested. The vgi extension, since protocol 3.4, refuses such a pointer.
// So this SDK must never emit one: its only route to a pointer is vgi-rpc's
// HTTP externalizer, which always attaches the digest, and nothing here calls
// `publish_external` or `Result::from_external_ref`.
//
// The test holds that as a statement about a running server: a real Dispatcher
// behind vgi-rpc's HTTP transport, with externalization configured to fire on
// every response, against an in-process storage service that keeps what was
// uploaded. Each answer must be a pointer whose digest is present and is the
// SHA-256 of the bytes the storage service received.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arrow/io/memory.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/http_config.h>
#include <vgi_rpc/metadata.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/wire.h>

#include "dispatcher.h"
#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"
#include "wire.h"

namespace {

namespace gen = ::vgi::generated;
namespace keys = ::vgi_rpc::keys;

const std::string kProtocol(gen::VGI_PROTOCOL_NAME);
const std::string kVersion(gen::VGI_PROTOCOL_VERSION);

std::string sha256_hex(const std::string& bytes) {
    const auto digest =
        vgi_rpc::crypto::sha256(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    return vgi_rpc::crypto::hex_encode(digest.data(), digest.size());
}

// vgi_rpc.conformance.fake_storage's contract, reduced to what an upload
// needs: POST /alloc names an object URL, PUT stores the bytes there.
class Storage {
public:
    Storage() {
        server_.Post("/alloc", [this](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::string url = base_url() + "/obj/" + std::to_string(next_++);
            res.set_content(nlohmann::json{{"object_url", url}}.dump(), "application/json");
        });
        server_.Put(R"(/obj/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(mutex_);
            objects_[base_url() + req.path] = req.body;
            res.status = 204;
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        REQUIRE(port_ > 0);
        thread_ = std::thread([this] { server_.listen_after_bind(); });
    }

    ~Storage() {
        server_.stop();
        thread_.join();
    }

    std::string base_url() const { return "http://127.0.0.1:" + std::to_string(port_); }

    std::map<std::string, std::string> objects() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return objects_;
    }

private:
    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
    mutable std::mutex mutex_;
    size_t next_ = 0;
    std::map<std::string, std::string> objects_;
};

// A Dispatcher behind vgi-rpc's HTTP transport on a background thread.
class Served {
public:
    Served(vgi::Dispatcher& dispatcher, vgi_rpc::HttpConfig cfg) {
        vgi_rpc::ServerBuilder builder;
        builder.protocol(kProtocol).protocol_version(kVersion);
        dispatcher.install(builder);
        server_ = builder.build();
        cfg.host = "127.0.0.1";
        cfg.port = 0;
        cfg.prefix = "/vgi";
        cfg.on_listen = [this](int port) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                port_ = port;
            }
            ready_.notify_all();
        };
        thread_ = std::thread([this, cfg] { server_->serve_http(cfg); });
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait_for(lock, std::chrono::seconds(10), [this] { return port_ != 0; });
        REQUIRE(port_ != 0);
    }

    ~Served() {
        // The accept loop lives inside serve_http, which has no stop: detach
        // and leak rather than free the server underneath it.
        thread_.detach();
        (void)server_.release();
    }

    // POST one unary vgi.v3 call and return the raw response body.
    std::string post(const std::string& method,
                     const std::shared_ptr<arrow::RecordBatch>& params) const {
        auto md = std::make_shared<arrow::KeyValueMetadata>();
        md->Append(keys::METHOD, method);
        md->Append(keys::REQUEST_VERSION, vgi_rpc::REQUEST_VERSION_VALUE);
        md->Append(keys::PROTOCOL, kProtocol);
        md->Append(keys::PROTOCOL_VERSION, kVersion);
        auto sink = vgi_rpc::unwrap(arrow::io::BufferOutputStream::Create());
        vgi_rpc::write_ipc_stream(sink, params->schema(),
                                  {vgi_rpc::AnnotatedBatch::with_metadata(params, md)});
        const auto body = vgi_rpc::unwrap(sink->Finish());

        httplib::Client client("127.0.0.1", port_);
        client.set_read_timeout(10, 0);
        const httplib::Headers headers{{"Accept-Encoding", "identity"},
                                       {"X-VGI-Accept-Encoding", "identity"}};
        auto res = client.Post("/vgi/" + kProtocol + "/" + method, headers, body->ToString(),
                               "application/vnd.apache.arrow.stream");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        REQUIRE_FALSE(res->has_header("X-VGI-RPC-Error"));
        return res->body;
    }

private:
    std::unique_ptr<vgi_rpc::Server> server_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable ready_;
    int port_ = 0;
};

// The response's result slot must be a pointer naming an object the storage
// service holds, with the SHA-256 of those exact bytes. Returns the payload
// the pointer resolves to.
std::shared_ptr<arrow::RecordBatch> require_digested_pointer(const std::string& body,
                                                             const Storage& storage) {
    auto reader = std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(body));
    const auto contents = vgi_rpc::read_ipc_stream(reader);
    REQUIRE(contents.has_value());
    REQUIRE_FALSE(contents->batches.empty());
    const auto& slot = contents->batches.back();
    REQUIRE(slot.batch->num_rows() == 0);
    REQUIRE(slot.custom_metadata);

    const std::string url = vgi_rpc::get_metadata_value(slot.custom_metadata, keys::LOCATION);
    REQUIRE_FALSE(url.empty());
    INFO("pointer " << url);
    REQUIRE(slot.custom_metadata->FindKey(keys::LOCATION_SHA256) >= 0);
    const std::string digest =
        vgi_rpc::get_metadata_value(slot.custom_metadata, keys::LOCATION_SHA256);

    const auto objects = storage.objects();
    const auto stored = objects.find(url);
    REQUIRE(stored != objects.end());
    CHECK(digest == sha256_hex(stored->second));

    auto object =
        std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(stored->second));
    const auto resolved = vgi_rpc::read_ipc_stream(object);
    REQUIRE(resolved.has_value());
    REQUIRE(resolved->batches.size() == 1);
    return resolved->batches.front().batch;
}

std::shared_ptr<arrow::RecordBatch> no_params() {
    return arrow::RecordBatch::Make(arrow::schema({}), 1, arrow::ArrayVector{});
}

}  // namespace

TEST_CASE("every result pointer the worker publishes carries its sha256", "[external][http]") {
    Storage storage;
    vgi::Dispatcher dispatcher;
    dispatcher.catalog().name = "digested";

    // Externalize every response, uncompressed so the stored bytes are the
    // bytes the digest covers. The threshold doubles as the inline-request
    // ceiling unless one is set.
    vgi_rpc::HttpConfig cfg;
    cfg.external_storage_url = storage.base_url();
    cfg.externalize_threshold = 0;
    cfg.max_request_bytes = 1 << 20;
    Served worker(dispatcher, cfg);

    const auto catalogs =
        require_digested_pointer(worker.post("catalog_catalogs", no_params()), storage);
    const auto listed = vgi::wire::get_binary_list(
        vgi::wire::decode_ipc(vgi::wire::get_binary(catalogs, "result")), "items");
    REQUIRE(listed.size() == 1);
    CHECK(vgi::wire::get_string(vgi::wire::decode_ipc(listed[0]), "name") == "digested");

    const auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                             .set_string("name", "digested")
                             .fill_defaults()
                             .finish();
    const auto attached = require_digested_pointer(
        worker.post("catalog_attach", vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
                                          .set_binary("request", vgi::wire::encode_ipc(request))
                                          .finish()),
        storage);
    CHECK_FALSE(
        vgi::wire::get_binary(vgi::wire::decode_ipc(vgi::wire::get_binary(attached, "result")),
                              "attach_opaque_data")
            .empty());

    // Every object uploaded was named by a pointer checked above.
    CHECK(storage.objects().size() == 2);
}
