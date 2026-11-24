// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Sealed framework payload v1 (src/storage_seal.h): vgi-python's cross-SDK
// vector byte for byte, every refusal, the host-local key file, and the
// framework's own storage writes holding no plaintext secret.

#include <catch2/catch_test_macros.hpp>

#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

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
#include "storage_seal.h"
#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"
#include "vgi/storage.h"
#include "wire.h"

namespace seal = ::vgi::storage_seal;

namespace {

namespace gen = ::vgi::generated;

std::string hex(const std::string& bytes) {
    return vgi_rpc::crypto::hex_encode(reinterpret_cast<const uint8_t*>(bytes.data()),
                                       bytes.size());
}

std::string unhex(const std::string& text) {
    const auto bytes = vgi_rpc::crypto::hex_decode(text);
    REQUIRE(bytes.has_value());
    return std::string(bytes->begin(), bytes->end());
}

std::string range_bytes(size_t n) {
    std::string out;
    for (size_t i = 0; i < n; ++i) out.push_back(static_cast<char>(i));
    return out;
}

// vgi-python tests/test_storage_seal.py VECTOR (Go/Java/TS/Rust check the same).
const std::string kVectorIkm = "vgi-test-signing-key-0123456789abcdef";
const std::string kVectorPlaintext = "vgi sealed framework payload v1";
const std::string kVectorDerivedKeyHex =
    "f5f759ad2009e91cf4b963a20b524a800224d2ac4e01472396dc1a2dab78bea9";
const std::string kVectorAadHex =
    "7667692e73746f726167652e7061796c6f61642e7631000000246174742d30303131323233333434353536363737"
    "3838393961616262636364646565666600000010000102030405060708090a0b0c0d0e0f000000135f7667692f62"
    "7566666572696e675f696e697400000008ffffffffffffffff";
const std::string kVectorWireHex =
    "01000102030405060708090a0bc79698647483a87eb533cf7332d1c527275cc7470dc6a3bdf1f72f624a97e24641"
    "6d4de55281bd0eee6d6491796a40";

seal::Slot vector_slot() {
    return {"att-00112233445566778899aabbccddeeff", range_bytes(16), seal::kBufferingInitNs,
            seal::pack_int_key(-1)};
}

std::array<uint8_t, seal::kNonceBytes> vector_nonce() {
    std::array<uint8_t, seal::kNonceBytes> nonce{};
    for (size_t i = 0; i < nonce.size(); ++i) nonce[i] = static_cast<uint8_t>(i);
    return nonce;
}

const std::string kSecret = "sk-dummy-secret-must-never-be-stored-in-plaintext";

// A private key file per test process, and no signing key: the subprocess
// and unix case, through the host-local key.
struct HostLocalKey {
    std::filesystem::path dir;
    std::filesystem::path file;
    HostLocalKey() {
        // A fresh path each time: the SDK caches a key per path.
        static int counter = 0;
        dir = std::filesystem::temp_directory_path() /
              ("vgi-seal-test-" + std::to_string(::getpid()) + "-" + std::to_string(++counter));
        std::filesystem::create_directories(dir);
        ::chmod(dir.c_str(), 0700);
        file = dir / "attach-secret-options.key";
        ::setenv("VGI_ATTACH_SECRET_KEY_FILE", file.c_str(), 1);
        ::unsetenv("VGI_SIGNING_KEY");
    }
    ~HostLocalKey() {
        ::unsetenv("VGI_ATTACH_SECRET_KEY_FILE");
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};

template <typename F>
bool refuses(F&& call) {
    try {
        call();
    } catch (const seal::StoragePayloadError&) {
        return true;
    }
    return false;
}

}  // namespace

TEST_CASE("the cross-SDK vector seals to vgi-python's bytes and opens back", "[storage_seal]") {
    CHECK(seal::pack_int_key(-1) == std::string(8, '\xff'));
    CHECK(hex(seal::pack_int_key(-2)) == "feffffffffffffff");
    CHECK(hex(seal::pack_int_key(258)) == "0201000000000000");

    const auto key = seal::derive_key(kVectorIkm);
    CHECK(vgi_rpc::crypto::hex_encode(key.data(), key.size()) == kVectorDerivedKeyHex);
    CHECK(hex(seal::aad(vector_slot())) == kVectorAadHex);
    CHECK(hex(seal::seal_with(kVectorPlaintext, kVectorIkm, vector_slot(), vector_nonce())) ==
          kVectorWireHex);
    CHECK(seal::open_with(unhex(kVectorWireHex), kVectorIkm, vector_slot()) == kVectorPlaintext);
}

TEST_CASE("a payload opens only unmodified, in its slot, under its key", "[storage_seal]") {
    const auto ikm = kVectorIkm;
    const auto slot = vector_slot();
    const auto sealed = seal::seal_with(kSecret, ikm, slot);
    CHECK(sealed.find(kSecret) == std::string::npos);
    CHECK(sealed.size() == 1 + seal::kNonceBytes + kSecret.size() + seal::kTagBytes);
    CHECK(seal::open_with(sealed, ikm, slot) == kSecret);
    // Random nonces: the same payload never seals to the same bytes.
    CHECK(seal::seal_with(kSecret, ikm, slot) != sealed);

    for (size_t i = 0; i < sealed.size(); ++i) {
        std::string tampered = sealed;
        tampered[i] = static_cast<char>(tampered[i] ^ 0x01);
        CHECK(refuses([&] { (void)seal::open_with(tampered, ikm, slot); }));
    }
    auto moved = slot;
    moved.shard_key = "att-other";
    CHECK(refuses([&] { (void)seal::open_with(sealed, ikm, moved); }));
    moved = slot;
    moved.scope_id = range_bytes(15);
    CHECK(refuses([&] { (void)seal::open_with(sealed, ikm, moved); }));
    moved = slot;
    moved.ns = seal::kAggregateStateNs;
    CHECK(refuses([&] { (void)seal::open_with(sealed, ikm, moved); }));
    moved = slot;
    moved.key = seal::pack_int_key(-2);
    CHECK(refuses([&] { (void)seal::open_with(sealed, ikm, moved); }));
    CHECK(refuses([&] { (void)seal::open_with(sealed, ikm + "x", slot); }));
    CHECK(refuses([&] { (void)seal::open_with(sealed.substr(0, 20), ikm, slot); }));
    CHECK(refuses([&] { (void)seal::open_with(std::string(), ikm, slot); }));
    // An unsealed value an older release wrote is refused, never read.
    CHECK(refuses([&] { (void)seal::open_with("plain bind call bytes " + kSecret, ikm, slot); }));
}

TEST_CASE("VGI_SIGNING_KEY is the key material exactly as configured", "[storage_seal]") {
    HostLocalKey isolated;
    ::setenv("VGI_SIGNING_KEY", kVectorIkm.c_str(), 1);
    CHECK(seal::resolve_ikm() == kVectorIkm);
    CHECK(seal::open(unhex(kVectorWireHex), vector_slot()) == kVectorPlaintext);
    ::unsetenv("VGI_SIGNING_KEY");
    // Not created: the signing key was enough.
    CHECK_FALSE(std::filesystem::exists(isolated.file));
}

TEST_CASE("without a signing key, the private host-local key file", "[storage_seal]") {
    HostLocalKey isolated;
    CHECK(seal::host_local_key_path() == isolated.file.string());
    const auto sealed = seal::seal(kSecret, vector_slot());
    REQUIRE(std::filesystem::exists(isolated.file));
    CHECK(std::filesystem::file_size(isolated.file) == 32);
    struct stat st{};
    REQUIRE(::stat(isolated.file.c_str(), &st) == 0);
    CHECK((st.st_mode & 0777) == 0600);
    CHECK(seal::open(sealed, vector_slot()) == kSecret);
    // Another process reads the same file and so opens the same payload.
    std::string ikm(32, '\0');
    {
        std::FILE* f = std::fopen(isolated.file.c_str(), "rb");
        REQUIRE(f != nullptr);
        REQUIRE(std::fread(ikm.data(), 1, 32, f) == 32);
        std::fclose(f);
    }
    CHECK(seal::open_with(sealed, ikm, vector_slot()) == kSecret);
}

TEST_CASE("an unusable host-local key file is an error, not a plaintext fallback",
          "[storage_seal]") {
    HostLocalKey isolated;
    {
        std::FILE* f = std::fopen(isolated.file.c_str(), "wb");
        REQUIRE(f != nullptr);
        std::fputs("short", f);
        std::fclose(f);
        ::chmod(isolated.file.c_str(), 0600);
    }
    CHECK(refuses([&] { (void)seal::seal(kSecret, vector_slot()); }));
    CHECK(refuses([&] { (void)seal::open(std::string(40, '\x01'), vector_slot()); }));
}

TEST_CASE("framework storage holds the sealed envelope, never the plaintext",
          "[storage_seal][storage]") {
    HostLocalKey isolated;
    auto storage = vgi::default_storage();
    const auto random = vgi_rpc::crypto::random_bytes(8);
    const std::string execution = "seal-test-" + hex(std::string(random.begin(), random.end()));
    const seal::Slot slot{"", execution, seal::kBufferingInitNs, seal::pack_int_key(-1)};

    seal::sealed_put(*storage, execution, "bind.call", slot, "bind call with " + kSecret);
    const auto raw = storage->kv_get(execution, "bind.call");
    REQUIRE(raw.has_value());
    CHECK(raw->find(kSecret) == std::string::npos);
    CHECK(static_cast<uint8_t>((*raw)[0]) == seal::kVersion);
    CHECK(seal::sealed_get(*storage, execution, "bind.call", slot) == "bind call with " + kSecret);
    CHECK_FALSE(seal::sealed_get(*storage, execution, "absent", slot).has_value());

    std::string tampered = *raw;
    tampered[tampered.size() / 2] = static_cast<char>(tampered[tampered.size() / 2] ^ 0x01);
    storage->kv_put(execution, "bind.call", tampered);
    CHECK(refuses([&] { (void)seal::sealed_get(*storage, execution, "bind.call", slot); }));
    // Copied to another slot, a genuine envelope does not open either.
    storage->kv_put(execution, "bind.schema", *raw);
    CHECK(refuses([&] {
        (void)seal::sealed_get(*storage, execution, "bind.schema",
                               {"", execution, seal::kBufferingInitNs, seal::pack_int_key(-2)});
    }));
    storage->clear(execution);
}

namespace {

const std::string kProtocol(gen::VGI_PROTOCOL_NAME);
const std::string kVersion(gen::VGI_PROTOCOL_VERSION);

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

// An OS-owned transport (no opaque key): the secret attach option is kept
// server-side, in framework storage.
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
    bool refused(const std::string& method, const std::shared_ptr<arrow::RecordBatch>& params) {
        try {
            (void)client_->call_unary(method, params);
        } catch (const vgi_rpc::RpcException&) {
            return true;
        }
        return false;
    }

private:
    std::unique_ptr<vgi_rpc::Server> server_;
    int fds_[2] = {-1, -1};
    std::thread thread_;
    std::optional<vgi_rpc::RpcClient> client_;
};

std::shared_ptr<arrow::RecordBatch> attach_params() {
    arrow::StringBuilder api_key, region;
    REQUIRE(api_key.Append(kSecret).ok());
    REQUIRE(region.Append("eu-west-1").ok());
    std::shared_ptr<arrow::Array> a = vgi_rpc::unwrap(api_key.Finish());
    std::shared_ptr<arrow::Array> r = vgi_rpc::unwrap(region.Finish());
    const auto options = vgi::wire::encode_ipc(
        arrow::RecordBatch::Make(arrow::schema({arrow::field("api_key", arrow::utf8()),
                                                arrow::field("region", arrow::utf8())}),
                                 1, {a, r}));
    const auto request = vgi::wire::ResultBuilder(gen::CatalogAttachRequestSchema())
                             .set_string("name", "vault")
                             .set_binary("options", options)
                             .fill_defaults()
                             .finish();
    return vgi::wire::ResultBuilder(gen::CatalogAttachParamsSchema())
        .set_binary("request", vgi::wire::encode_ipc(request))
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

}  // namespace

TEST_CASE("a secret attach option kept server-side is stored sealed",
          "[storage_seal][dispatcher]") {
    HostLocalKey isolated;
    vgi::Dispatcher dispatcher;
    vgi::CatalogModel model;
    model.name = "vault";
    vgi::AttachOptionSpec api_key{"api_key", "credential", arrow::utf8(), nullptr, true};
    api_key.secret = true;
    vgi::AttachOptionSpec region{"region", "plain", arrow::utf8(), nullptr, false};
    model.attach_options = {api_key, region};
    dispatcher.set_catalog(std::move(model));
    dispatcher.register_table(std::make_shared<OptionsProbe>());
    Served worker(dispatcher);

    const auto value =
        vgi::wire::get_binary(worker.call("catalog_attach", attach_params()), "attach_opaque_data");
    // catalog \0 data_version \0 alias \0 id \0 options
    std::vector<std::string> fields;
    for (size_t start = 0;;) {
        const auto separator = value.find('\0', start);
        fields.push_back(value.substr(start, separator - start));
        if (separator == std::string::npos) break;
        start = separator + 1;
    }
    REQUIRE(fields.size() == 5);
    const auto& attachment_id = fields[3];

    auto storage = vgi::default_storage();
    const auto raw = storage->kv_get("vgi.attach_secrets", attachment_id);
    REQUIRE(raw.has_value());
    CHECK(raw->find(kSecret) == std::string::npos);
    CHECK(static_cast<uint8_t>((*raw)[0]) == seal::kVersion);

    (void)worker.call("bind", bind_params(value));
    {
        std::lock_guard<std::mutex> lock(g_seen_mutex);
        REQUIRE(g_seen_options != nullptr);
        CHECK(vgi::wire::get_string(g_seen_options, "api_key") == kSecret);
    }

    std::string tampered = *raw;
    tampered[tampered.size() - 1] = static_cast<char>(tampered[tampered.size() - 1] ^ 0x01);
    storage->kv_put("vgi.attach_secrets", attachment_id, tampered);
    CHECK(worker.refused("bind", bind_params(value)));
    storage->kv_del("vgi.attach_secrets", attachment_id);
}
