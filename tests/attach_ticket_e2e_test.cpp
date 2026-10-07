// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Attach tickets end to end on the built example worker, over HTTP.
//
// A freshly logged-in user attaches `ticket_probe` with a region and a secret
// api_key, seals that attach (`vgi.attach_tickets.v1` `seal_attach`) and mints
// a grant (`vgi_rpc.Identity.v1` `issue_grant`). A second client, authenticated
// by nothing but `Bearer <grant>`, then attaches with the single option
// `vgi_attach_ticket` and gets the user's attach back -- the same options the
// probe reports from. Another principal's grant with that ticket is refused.

#include <catch2/catch_test_macros.hpp>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/util/base64.h>

#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/client.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/http_client.h>
#include <vgi_rpc/token_identity.h>

#include "opaque_seal.h"
#include "vgi/attach_ticket.h"
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
const std::string kGrantKey = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
const std::string kSigningKey = "attach-ticket-e2e-signing-key";
// vgi-python's protocol hash for vgi.attach_tickets.v1: equal hashes mean an
// identical wire surface in any port.
const std::string kTicketsHash = "241fffa801dd073c76fa933b9ad5ac790a95b81fee02e73a8332da4d3526b4e0";

using Env = std::map<std::string, std::string>;

class ServedWorker {
public:
    ServedWorker(const std::vector<std::string>& args, const Env& env) {
        int out[2];
        REQUIRE(::pipe(out) == 0);
        pid_ = ::fork();
        REQUIRE(pid_ >= 0);
        if (pid_ == 0) {
            ::dup2(out[1], STDOUT_FILENO);
            ::close(out[0]);
            ::close(out[1]);
            for (const char* name : {"VGI_RPC_GRANT_KEYS", "VGI_SIGNING_KEY", "VGI_BEARER_TOKENS",
                                     "VGI_FIXTURE_TEST_BEARERS", "VGI_WORKER_CATALOG_NAME",
                                     "VGI_RPC_GRANT_MAX_TTL_SECONDS"}) {
                ::unsetenv(name);
            }
            for (const auto& [name, value] : env) ::setenv(name.c_str(), value.c_str(), 1);
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

    pid_t pid_ = -1;
    int stdout_ = -1;
    std::string port_;
};

using Headers = std::vector<std::pair<std::string, std::string>>;

std::string now_text() {
    return std::to_string(
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count());
}

// A freshly logged-in user, through the fixture's header authentication.
Headers fresh_login(const std::string& principal) {
    return {{"X-Conformance-Principal", principal}, {"X-Conformance-Auth-Time", now_text()}};
}

Headers bearer(const std::string& token) {
    return {{"Authorization", "Bearer " + token}};
}

vgi_rpc::HttpClient client(const std::string& base_url, const std::string& protocol,
                           const std::string& version, const Headers& headers) {
    vgi_rpc::HttpClientConfig config;
    config.allow_insecure_credentials = true;  // loopback
    auto builder = vgi_rpc::HttpClient::builder(base_url).config(config).protocol(protocol);
    if (!version.empty()) builder.protocol_version(version);
    for (const auto& [name, value] : headers) builder.header(name, value);
    return builder.build();
}

std::shared_ptr<arrow::Array> one_text(const std::string& value) {
    arrow::StringBuilder builder;
    VGI_RPC_THROW_NOT_OK(builder.Append(value));
    return vgi_rpc::unwrap(builder.Finish());
}

std::shared_ptr<arrow::Array> one_binary(const std::string& value) {
    arrow::BinaryBuilder builder;
    VGI_RPC_THROW_NOT_OK(builder.Append(value));
    return vgi_rpc::unwrap(builder.Finish());
}

std::string options_ipc(const std::map<std::string, std::string>& options) {
    arrow::FieldVector fields;
    std::vector<std::shared_ptr<arrow::Array>> values;
    for (const auto& [name, value] : options) {
        fields.push_back(arrow::field(name, arrow::utf8(), /*nullable=*/true));
        values.push_back(one_text(value));
    }
    return vgi::wire::encode_ipc(arrow::RecordBatch::Make(arrow::schema(fields), 1, values));
}

// `{request: binary}` wrapping one dataclass record.
vgi_rpc::AnnotatedBatch wrapped(const std::shared_ptr<arrow::RecordBatch>& request) {
    std::shared_ptr<arrow::Array> column = one_binary(vgi::wire::encode_ipc(request));
    return vgi_rpc::AnnotatedBatch::data(arrow::RecordBatch::Make(
        arrow::schema({arrow::field("request", arrow::binary(), /*nullable=*/false)}), 1,
        {column}));
}

std::shared_ptr<arrow::RecordBatch> payload_of(const vgi_rpc::AnnotatedBatch& response) {
    return vgi::wire::decode_ipc(vgi::wire::get_binary(response.batch, "result"));
}

// The attachment the worker sealed into `attach_opaque_data`: its catalog and
// its merged options -- exactly what `main.ticket_probe` reads its row from.
struct Attached {
    std::string catalog;
    std::string region;
    std::string api_key;
};

// Who a call ran as, which is what its attach value is sealed for.
vgi_rpc::AuthContext as(const std::string& domain, const std::string& principal) {
    vgi_rpc::AuthContext auth;
    auth.domain = domain;
    auth.authenticated = !principal.empty();
    auth.principal = principal;
    return auth;
}

// The attach value is sealed for the caller under the deployment's signing
// key (vgi-opaque-data-sealing); this test holds that key, so it opens it.
Attached attach(const vgi_rpc::HttpClient& vgi, const vgi_rpc::AuthContext& caller,
                const std::map<std::string, std::string>& options,
                const std::string& name = "ticket_probe") {
    std::shared_ptr<arrow::Array> name_array = one_text(name);
    std::shared_ptr<arrow::Array> options_array = one_binary(options_ipc(options));
    std::shared_ptr<arrow::Array> no_version =
        vgi_rpc::unwrap(arrow::MakeArrayOfNull(arrow::utf8(), 1));
    std::shared_ptr<arrow::Array> no_implementation =
        vgi_rpc::unwrap(arrow::MakeArrayOfNull(arrow::utf8(), 1));
    std::shared_ptr<arrow::Array> no_capabilities =
        vgi_rpc::unwrap(arrow::MakeArrayOfNull(arrow::binary(), 1));
    const auto request = arrow::RecordBatch::Make(
        gen::CatalogAttachRequestSchema(), 1,
        {name_array, options_array, no_version, no_implementation, no_capabilities});
    const auto result = payload_of(vgi.call("catalog_attach", wrapped(request)));
    const auto sealed =
        vgi::opaque::open_attach(vgi::attach_ticket_key(kSigningKey),
                                 vgi::wire::get_binary(result, "attach_opaque_data"), caller);

    std::vector<std::string> fields;
    for (size_t start = 0;;) {
        const auto separator = sealed.find('\0', start);
        fields.push_back(sealed.substr(start, separator - start));
        if (separator == std::string::npos) break;
        start = separator + 1;
    }
    REQUIRE(fields.size() == 5);
    const auto merged = vgi::wire::decode_ipc(arrow::util::base64_decode(fields[4]));
    return {fields[0], vgi::wire::get_string(merged, "region"),
            vgi::wire::get_string(merged, "api_key")};
}

// The row `main.probe` reports for an attachment.
std::pair<std::string, std::string> probe_row(const Attached& attached) {
    const auto digest = vgi_rpc::crypto::sha256(
        reinterpret_cast<const uint8_t*>(attached.api_key.data()), attached.api_key.size());
    return {attached.region,
            vgi_rpc::crypto::hex_encode(digest.data(), digest.size()).substr(0, 12)};
}

struct Sealed {
    std::string ticket;
    double expires_at;
};

Sealed seal(const vgi_rpc::HttpClient& tickets, const std::map<std::string, std::string>& options,
            int64_t ttl = 0) {
    std::shared_ptr<arrow::Array> catalog = one_text("ticket_probe");
    std::shared_ptr<arrow::Array> opts = one_binary(options_ipc(options));
    std::shared_ptr<arrow::Array> data_version = one_text("");
    std::shared_ptr<arrow::Array> implementation = one_text("");
    arrow::Int64Builder ttl_builder;
    VGI_RPC_THROW_NOT_OK(ttl_builder.Append(ttl));
    std::shared_ptr<arrow::Array> ttl_array = vgi_rpc::unwrap(ttl_builder.Finish());
    const auto schema = arrow::schema({arrow::field("catalog_name", arrow::utf8(), false),
                                       arrow::field("options", arrow::binary(), true),
                                       arrow::field("data_version_spec", arrow::utf8(), false),
                                       arrow::field("implementation_version", arrow::utf8(), false),
                                       arrow::field("ttl_seconds", arrow::int64(), false)});
    const auto request = arrow::RecordBatch::Make(
        schema, 1, {catalog, opts, data_version, implementation, ttl_array});
    const auto ticket = payload_of(tickets.call("seal_attach", wrapped(request)));
    const auto expires =
        std::static_pointer_cast<arrow::DoubleArray>(ticket->GetColumnByName("expires_at"));
    return {vgi::wire::get_string(ticket, "ticket"), expires->Value(0)};
}

std::string issue_grant(const vgi_rpc::HttpClient& identity) {
    auto item = arrow::field("item", arrow::utf8(), /*nullable=*/true);
    arrow::ListBuilder scopes(arrow::default_memory_pool(),
                              std::make_shared<arrow::StringBuilder>(), arrow::list(item));
    VGI_RPC_THROW_NOT_OK(scopes.Append());
    std::shared_ptr<arrow::Array> purpose = one_text("vgi.unattended");
    std::shared_ptr<arrow::Array> scope_array = vgi_rpc::unwrap(scopes.Finish());
    arrow::Int64Builder ttl;
    VGI_RPC_THROW_NOT_OK(ttl.Append(600));
    std::shared_ptr<arrow::Array> ttl_array = vgi_rpc::unwrap(ttl.Finish());
    const auto schema = arrow::schema({arrow::field("purpose", arrow::utf8(), false),
                                       arrow::field("scopes", arrow::list(item), false),
                                       arrow::field("ttl_seconds", arrow::int64(), false)});
    const auto response =
        identity.call("issue_grant", vgi_rpc::AnnotatedBatch::data(arrow::RecordBatch::Make(
                                         schema, 1, {purpose, scope_array, ttl_array})));
    return vgi::wire::get_string(payload_of(response), "token");
}

struct Session {
    vgi_rpc::HttpClient vgi;
    vgi_rpc::HttpClient tickets;
    vgi_rpc::HttpClient identity;
};

Session session(const std::string& url, const Headers& headers) {
    return {
        client(url, kProtocol, kVersion, headers),
        client(url, vgi::kAttachTicketsProtocolName, vgi::kAttachTicketsProtocolVersion, headers),
        client(url, vgi_rpc::kIdentityProtocolName, "", headers)};
}

ServedWorker tickets_worker(Env extra = {}) {
    Env env{{"VGI_SIGNING_KEY", kSigningKey}};
    env.insert(extra.begin(), extra.end());
    return ServedWorker({"--http", "0", "--conformance-principal-header", "--grant-key", kGrantKey},
                        env);
}

std::optional<vgi_rpc::HostedProtocol> hosted(const std::string& url, const std::string& name) {
    for (const auto& protocol : client(url, kProtocol, kVersion, {}).list_protocols()) {
        if (protocol.name == name) return protocol;
    }
    return std::nullopt;
}

template <typename F>
vgi_rpc::RpcRemoteError refused(F&& call) {
    try {
        call();
    } catch (const vgi_rpc::RpcRemoteError& error) {
        return error;
    }
    FAIL("the call was not refused");
    throw std::logic_error("unreachable");
}

}  // namespace

TEST_CASE("http: grant plus ticket reattaches as the user", "[attach_ticket][http]") {
    auto worker = tickets_worker();
    const std::map<std::string, std::string> options{{"region", "eu-west-2"},
                                                     {"api_key", "sk-live-secret"}};
    const auto user = session(worker.base_url(), fresh_login("alice@example.com"));
    const auto original = attach(user.vgi, as("conformance", "alice@example.com"), options);
    CHECK(original.catalog == "ticket_probe");
    CHECK(probe_row(original) == std::pair<std::string, std::string>{"eu-west-2", "a03d43d254a4"});

    const auto sealed = seal(user.tickets, options);
    const auto grant = issue_grant(user.identity);
    CHECK(sealed.ticket.rfind(vgi::kAttachTicketPrefix, 0) == 0);
    CHECK(sealed.ticket.find("sk-live-secret") == std::string::npos);
    // Capped at the grant maximum (7 days), so the two halves expire together.
    const double now =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    CHECK(sealed.expires_at <= now + 7 * 24 * 3600 + 5);
    CHECK(sealed.expires_at > now + 7 * 24 * 3600 - 60);

    // The runner holds only the grant, and names some other catalog -- a
    // memory catalog, which the router would hand the request to by name: the
    // ticket is redeemed before routing, so the sealed one is what attaches.
    const auto runner = session(worker.base_url(), bearer(grant));
    const auto reattached = attach(runner.vgi, as("grant", "alice@example.com"),
                                   {{"vgi_attach_ticket", sealed.ticket}}, "contents_memory");
    CHECK(reattached.catalog == "ticket_probe");
    CHECK(probe_row(reattached) == probe_row(original));
}

TEST_CASE("http: another user's grant cannot redeem the ticket", "[attach_ticket][http]") {
    auto worker = tickets_worker();
    const auto alice = session(worker.base_url(), fresh_login("alice"));
    const auto sealed = seal(alice.tickets, {{"api_key", "k"}});
    const auto bobs_grant = issue_grant(session(worker.base_url(), fresh_login("bob")).identity);
    const auto runner = session(worker.base_url(), bearer(bobs_grant));
    const auto error = refused([&] {
        (void)attach(runner.vgi, as("grant", "bob"), {{"vgi_attach_ticket", sealed.ticket}});
    });
    CHECK(error.error_kind() == "attach_ticket_invalid");
    CHECK(error.error_code() == "INVALID_ARGUMENT");
    CHECK(std::string(error.what()).find(sealed.ticket) == std::string::npos);
}

TEST_CASE("http: no option may ride beside the ticket", "[attach_ticket][http]") {
    auto worker = tickets_worker();
    const auto alice = session(worker.base_url(), fresh_login("alice"));
    const auto sealed = seal(alice.tickets, {{"api_key", "k"}});
    const auto error = refused([&] {
        (void)attach(alice.vgi, as("conformance", "alice"),
                     {{"vgi_attach_ticket", sealed.ticket}, {"region", "us-west-1"}});
    });
    CHECK(error.error_kind() == "invalid_request");
}

TEST_CASE("http: seal_attach refuses the anonymous and validates options",
          "[attach_ticket][http]") {
    auto worker = tickets_worker();
    const auto anonymous = session(worker.base_url(), {});
    auto error = refused([&] { (void)seal(anonymous.tickets, {{"api_key", "k"}}); });
    CHECK(error.error_kind() == "action_denied");
    CHECK(error.error_code() == "PERMISSION_DENIED");

    const auto alice = session(worker.base_url(), fresh_login("alice"));
    error = refused([&] { (void)seal(alice.tickets, {{"api_key", "k"}, {"x", "1"}}); });
    CHECK(error.error_kind() == "invalid_request");
    CHECK(error.error_code() == "INVALID_ARGUMENT");
    error = refused([&] { (void)seal(alice.tickets, {{"region", "r"}}); });  // api_key required
    CHECK(error.error_kind() == "invalid_request");
    error = refused([&] { (void)seal(alice.tickets, {{"api_key", "k"}}, -1); });
    CHECK(error.error_kind() == "invalid_request");
}

TEST_CASE("http: vgi.attach_tickets.v1 is hosted only with a key and grants",
          "[attach_ticket][http]") {
    {
        auto worker = tickets_worker();
        const auto protocol = hosted(worker.base_url(), vgi::kAttachTicketsProtocolName);
        REQUIRE(protocol);
        CHECK(protocol->version == vgi::kAttachTicketsProtocolVersion);
        CHECK(protocol->hash == kTicketsHash);
    }
    {
        // No explicit signing key.
        ServedWorker worker(
            {"--http", "0", "--conformance-principal-header", "--grant-key", kGrantKey}, {});
        CHECK_FALSE(hosted(worker.base_url(), vgi::kAttachTicketsProtocolName));
    }
    {
        // No way to issue grants.
        ServedWorker worker({"--http", "0", "--conformance-principal-header"},
                            {{"VGI_SIGNING_KEY", kSigningKey}});
        CHECK_FALSE(hosted(worker.base_url(), vgi::kAttachTicketsProtocolName));
    }
}

TEST_CASE("stdio: no attach tickets off HTTP", "[attach_ticket][stdio]") {
    ::setenv("VGI_SIGNING_KEY", kSigningKey.c_str(), 1);
    ::setenv("VGI_RPC_GRANT_KEYS", kGrantKey.c_str(), 1);
    vgi_rpc::RpcClientOptions options;
    options.protocol = kProtocol;
    options.protocol_version = kVersion;
    auto stdio = vgi_rpc::RpcClient::spawn({kWorker}, options);
    for (const auto& protocol : stdio.list_protocols()) {
        CHECK(protocol.name != vgi::kAttachTicketsProtocolName);
    }
    // And a ticket presented there is invalid: there is no key to open it under.
    std::shared_ptr<arrow::Array> name = one_text("ticket_probe");
    std::shared_ptr<arrow::Array> opts =
        one_binary(options_ipc({{"vgi_attach_ticket", "vgia1.AAAA"}}));
    std::shared_ptr<arrow::Array> none = vgi_rpc::unwrap(arrow::MakeArrayOfNull(arrow::utf8(), 1));
    std::shared_ptr<arrow::Array> none_binary =
        vgi_rpc::unwrap(arrow::MakeArrayOfNull(arrow::binary(), 1));
    const auto request = arrow::RecordBatch::Make(gen::CatalogAttachRequestSchema(), 1,
                                                  {name, opts, none, none, none_binary});
    try {
        (void)stdio.call_unary("catalog_attach", wrapped(request).batch);
        FAIL("a ticket was redeemed off HTTP");
    } catch (const vgi_rpc::RpcException& error) {
        CHECK(error.error_kind() == "attach_ticket_invalid");
    }
    stdio.close();
    ::unsetenv("VGI_SIGNING_KEY");
    ::unsetenv("VGI_RPC_GRANT_KEYS");
}

TEST_CASE("http: the fixture's test bearers are fresh logins", "[attach_ticket][http]") {
    ServedWorker worker({"--http", "0", "--grant-key", kGrantKey},
                        {{"VGI_SIGNING_KEY", kSigningKey}, {"VGI_FIXTURE_TEST_BEARERS", "1"}});
    const auto alice = session(worker.base_url(), bearer("vgi-test-alice"));
    const std::map<std::string, std::string> options{{"api_key", "sk-test-0123456789"}};
    CHECK(probe_row(attach(alice.vgi, as("bearer", "alice"), options)) ==
          std::pair<std::string, std::string>{"us-east-1", "0d3b56072291"});
    const auto sealed = seal(alice.tickets, options);
    const auto grant = issue_grant(alice.identity);
    const auto runner = session(worker.base_url(), bearer(grant));
    CHECK(probe_row(
              attach(runner.vgi, as("grant", "alice"), {{"vgi_attach_ticket", sealed.ticket}})) ==
          std::pair<std::string, std::string>{"us-east-1", "0d3b56072291"});
    const auto bob = session(worker.base_url(), bearer("vgi-test-bob"));
    CHECK(refused([&] {
              (void)attach(bob.vgi, as("bearer", "bob"), {{"vgi_attach_ticket", sealed.ticket}});
          }).error_kind() == "attach_ticket_invalid");
}
