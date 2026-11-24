// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Orchard hosted mode end to end: the built example worker with
// `VGI_ORCHARD_HOSTED=1` against a JWKS served from this process. The port of
// vgi-python's `tests/test_orchard_hosted.py`; the full SDK-neutral contract
// is the extension repo's `test/hosted/` harness, which CI also runs.

#include <catch2/catch_test_macros.hpp>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arrow/record_batch.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <vgi_rpc/http_client.h>

#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"
#include "pass_minter.h"
#include "wire.h"

#ifndef VGI_EXAMPLE_WORKER_PATH
#error "VGI_EXAMPLE_WORKER_PATH must name the built vgi-example-worker"
#endif

extern char** environ;

namespace {

namespace gen = ::vgi::generated;
using pass_minter::SigningKey;

const std::string kWorker = VGI_EXAMPLE_WORKER_PATH;
const std::string kIssuer = "https://auth.orchard.test";
const std::string kAudience = "https://desk.orchard.test/d/dep_123";
const std::string kChallenge = "Bearer error=\"invalid_token\"";
const std::string kRpcPath = "/" + std::string(gen::VGI_PROTOCOL_NAME) + "/catalog_attach";

// Inherited variables that would configure another way in.
const std::vector<std::string> kScrubbed = {
    "VGI_ORCHARD_HOSTED",     "VGI_JWT_ISSUER",           "VGI_JWT_AUDIENCE",
    "VGI_JWT_JWKS_URI",       "VGI_JWT_ALGORITHMS",       "VGI_SIGNING_KEY",
    "VGI_SIGNING_KEY_MINTED", "VGI_BEARER_TOKENS",        "VGI_OPTIONAL_BEARER_TOKENS",
    "VGI_RPC_GRANT_KEYS",     "VGI_OAUTH_RESOURCE",       "VGI_PROXY_PROOF_MODE",
    "VGI_FIXTURE_IDENTITY",   "VGI_FIXTURE_TEST_BEARERS", "VGI_INTROSPECT_PRINCIPALS"};

using Env = std::map<std::string, std::optional<std::string>>;

int64_t wall_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

nlohmann::json claims() {
    const int64_t now = wall_now();
    return {{"iss", kIssuer},
            {"aud", kAudience},
            {"sub", "user_01M7KNJYC18T"},
            {"kind", "user"},
            {"email", "ada@example.com"},
            {"email_verified", true},
            {"name", "Ada"},
            {"org_id", "org_01M7"},
            {"client_id", "vgi-duckdb"},
            {"iat", now},
            {"nbf", now},
            {"exp", now + 900},
            {"jti", "jti-" + std::to_string(now)}};
}

// The process-wide signing key and the JWKS that publishes it.
const SigningKey& desk_key() {
    static const SigningKey key = SigningKey::ec("desk-es256-1");
    return key;
}

class JwksServer {
public:
    JwksServer() {
        const std::string body = nlohmann::json{{"keys", {desk_key().jwk}}}.dump();
        server_.Get("/jwks", [body](const httplib::Request&, httplib::Response& res) {
            res.set_content(body, "application/json");
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        REQUIRE(port_ > 0);
        thread_ = std::thread([this] { server_.listen_after_bind(); });
    }
    ~JwksServer() {
        server_.stop();
        thread_.join();
    }
    std::string uri() const { return "http://127.0.0.1:" + std::to_string(port_) + "/jwks"; }

private:
    httplib::Server server_;
    int port_ = 0;
    std::thread thread_;
};

Env hosted_env(const std::string& jwks_uri) {
    return {{"VGI_ORCHARD_HOSTED", "1"},
            {"VGI_JWT_ISSUER", kIssuer},
            {"VGI_JWT_AUDIENCE", kAudience},
            {"VGI_JWT_JWKS_URI", jwks_uri},
            {"VGI_SIGNING_KEY", "orchard-hosted-test-signing-key-0001"}};
}

Env with(Env env, const Env& overrides) {
    for (const auto& [name, value] : overrides) env[name] = value;
    return env;
}

std::vector<std::string> child_environment(const Env& env) {
    std::vector<std::string> out;
    for (char** e = environ; *e != nullptr; ++e) {
        const std::string entry(*e);
        const std::string name = entry.substr(0, entry.find('='));
        if (std::find(kScrubbed.begin(), kScrubbed.end(), name) != kScrubbed.end()) continue;
        if (env.count(name)) continue;
        out.push_back(entry);
    }
    for (const auto& [name, value] : env) {
        if (value) out.push_back(name + "=" + *value);
    }
    return out;
}

// The example worker with `args` under `env`; stderr goes to a file.
class WorkerProcess {
public:
    WorkerProcess(const std::vector<std::string>& args, const Env& env) {
        stderr_path_ =
            std::filesystem::temp_directory_path() / ("vgi-hosted-" + std::to_string(::getpid()) +
                                                      "-" + std::to_string(counter_++) + ".stderr");
        int out[2];
        REQUIRE(::pipe(out) == 0);
        std::vector<std::string> owned{kWorker};
        owned.insert(owned.end(), args.begin(), args.end());
        std::vector<std::string> envs = child_environment(env);
        pid_ = ::fork();
        REQUIRE(pid_ >= 0);
        if (pid_ == 0) {
            ::dup2(out[1], STDOUT_FILENO);
            const int err = ::open(stderr_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (err >= 0) ::dup2(err, STDERR_FILENO);
            ::close(out[0]);
            ::close(out[1]);
            std::vector<char*> argv;
            for (auto& arg : owned) argv.push_back(arg.data());
            argv.push_back(nullptr);
            std::vector<char*> envp;
            for (auto& e : envs) envp.push_back(e.data());
            envp.push_back(nullptr);
            ::execve(kWorker.c_str(), argv.data(), envp.data());
            ::_exit(127);
        }
        ::close(out[1]);
        stdout_ = out[0];
    }

    ~WorkerProcess() {
        if (pid_ > 0 && !reaped_) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, nullptr, 0);
        }
        if (stdout_ >= 0) ::close(stdout_);
        std::error_code ignored;
        std::filesystem::remove(stderr_path_, ignored);
    }

    // The port from `PORT:<n>`, or nothing when the worker exited first.
    std::optional<int> port() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        std::string line;
        while (std::chrono::steady_clock::now() < deadline) {
            pollfd pfd{stdout_, POLLIN, 0};
            if (::poll(&pfd, 1, 100) <= 0) continue;
            char c = 0;
            if (::read(stdout_, &c, 1) != 1) return std::nullopt;
            if (c != '\n') {
                line.push_back(c);
                continue;
            }
            if (line.rfind("PORT:", 0) == 0) return std::stoi(line.substr(5));
            line.clear();
        }
        return std::nullopt;
    }

    // The exit status, waiting up to 30 s.
    std::optional<int> exit_status() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
            int status = 0;
            const pid_t done = ::waitpid(pid_, &status, WNOHANG);
            if (done == pid_) {
                reaped_ = true;
                return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return std::nullopt;
    }

    std::string stderr_text() const {
        std::ifstream in(stderr_path_);
        std::stringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
    }

private:
    static inline int counter_ = 0;
    pid_t pid_ = -1;
    bool reaped_ = false;
    int stdout_ = -1;
    std::filesystem::path stderr_path_;
};

// What a refused start says on stderr; fails the test when it started.
std::string refused(const std::vector<std::string>& args, const Env& env) {
    WorkerProcess worker(args, env);
    const auto status = worker.exit_status();
    REQUIRE(status.has_value());
    CHECK(*status != 0);
    return worker.stderr_text();
}

httplib::Result send(int port, const std::string& method, const std::string& path,
                     const std::optional<std::string>& token) {
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(10, 0);
    httplib::Headers headers{{"Origin", "https://shell.example.com"}};
    if (token) headers.emplace("Authorization", "Bearer " + *token);
    if (method == "HEAD") return client.Head(path, headers);
    if (method == "OPTIONS") return client.Options(path, headers);
    if (method == "POST")
        return client.Post(path, headers, "", "application/vnd.apache.arrow.stream");
    return client.Get(path, headers);
}

void check_401(const httplib::Result& res, const std::string& reason) {
    REQUIRE(res);
    CHECK(res->status == 401);
    CHECK(res->get_header_value("WWW-Authenticate") == kChallenge);
    CHECK(res->get_header_value("VGI-Auth-Reason") == reason);
    CHECK(res->has_header("VGI-Supported-Encodings"));
    // Hosted 401s advertise the protocol windows too (PROTOCOL_VERSIONS_V1.md
    // §4.1), and expose them to a browser.
    CHECK(res->get_header_value("VGI-Protocol-Versions").rfind("vgi.v3=", 0) == 0);
    CHECK(res->get_header_value("Access-Control-Expose-Headers").find("vgi-protocol-versions") !=
          std::string::npos);
    CHECK(res->get_header_value("Access-Control-Allow-Origin") == "*");
}

std::string tamper(const std::string& token) {
    const auto first = token.find('.');
    const auto second = token.find('.', first + 1);
    auto forged = claims();
    forged["org_id"] = "org_EVIL";
    return token.substr(0, first) + "." + pass_minter::b64u(forged.dump()) + token.substr(second);
}

}  // namespace

TEST_CASE("a hosted worker serves Pass holders only", "[hosted][http]") {
    JwksServer jwks;
    WorkerProcess worker({"--http", "0"}, hosted_env(jwks.uri()));
    const auto port = worker.port();
    REQUIRE(port.has_value());

    const auto head = send(*port, "HEAD", "/health", std::nullopt);
    REQUIRE(head);
    CHECK(head->status == 200);
    CHECK(head->has_header("VGI-Supported-Encodings"));
    CHECK(head->get_header_value("VGI-Protocol-Versions").rfind("vgi.v3=", 0) == 0);

    for (const auto& [method, path] : std::vector<std::pair<std::string, std::string>>{
             {"GET", "/health"}, {"OPTIONS", "/health"}, {"GET", "/"}, {"POST", kRpcPath}}) {
        INFO(method << " " << path);
        check_401(send(*port, method, path, std::nullopt), "missing_credential");
    }

    std::vector<std::string> used;
    const auto use = [&used](std::string token) {
        used.push_back(token);
        return token;
    };
    auto expired = claims();
    expired["exp"] = wall_now() - 120;
    auto wrong_aud = claims();
    wrong_aud["aud"] = "https://desk.orchard.test/d/dep_other";
    check_401(send(*port, "POST", kRpcPath, use(pass_minter::mint_pass(desk_key(), expired))),
              "expired_credential");
    check_401(send(*port, "POST", kRpcPath, use(pass_minter::mint_pass(desk_key(), wrong_aud))),
              "invalid_credential");
    check_401(
        send(*port, "POST", kRpcPath, use(tamper(pass_minter::mint_pass(desk_key(), claims())))),
        "invalid_credential");
    check_401(send(*port, "POST", kRpcPath,
                   use(pass_minter::mint_pass(
                       desk_key(), claims(),
                       {{"alg", "ES256"}, {"typ", "JWT"}, {"kid", "desk-es256-1"}}))),
              "invalid_credential");

    // A valid Pass reaches the worker: GET /health, and a vgi.v3 call.
    const std::string token = use(pass_minter::mint_pass(desk_key(), claims()));
    const auto health = send(*port, "GET", "/health", token);
    REQUIRE(health);
    CHECK(health->status == 200);
    vgi_rpc::HttpClientConfig config;
    config.allow_insecure_credentials = true;
    const auto client = vgi_rpc::HttpClient::builder("http://127.0.0.1:" + std::to_string(*port))
                            .config(config)
                            .protocol(std::string(gen::VGI_PROTOCOL_NAME))
                            .protocol_version(std::string(gen::VGI_PROTOCOL_VERSION))
                            .header("Authorization", "Bearer " + token)
                            .build();
    const auto response =
        client.call("catalog_catalogs", vgi_rpc::AnnotatedBatch::data(arrow::RecordBatch::Make(
                                            arrow::schema({}), 1, arrow::ArrayVector{})));
    CHECK(response.batch != nullptr);

    // Nothing of any token reaches the logs.
    const std::string log = worker.stderr_text();
    CHECK(log.find("Pass rejected") != std::string::npos);
    for (const auto& t : used) {
        CHECK(log.find(t) == std::string::npos);
        CHECK(log.find(t.substr(t.rfind('.') + 1)) == std::string::npos);
    }
    CHECK(log.find("Bearer ") == std::string::npos);
}

TEST_CASE("every misconfiguration refuses to start", "[hosted][startup]") {
    const std::string jwks = "http://127.0.0.1:9/jwks";  // never contacted
    const Env base = hosted_env(jwks);
    const std::vector<std::string> http = {"--http", "0"};
    const std::string key = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";

    for (const char* name : {"VGI_JWT_ISSUER", "VGI_JWT_AUDIENCE", "VGI_JWT_JWKS_URI"}) {
        INFO(name);
        CHECK(refused(http, with(base, {{name, std::nullopt}})).find(name) != std::string::npos);
        CHECK(refused(http, with(base, {{name, "  "}})).find(name) != std::string::npos);
    }
    const auto comma = refused(http, with(base, {{"VGI_JWT_AUDIENCE", kAudience + ",https://x"}}));
    CHECK(comma.find("VGI_JWT_AUDIENCE") != std::string::npos);
    CHECK(comma.find("exactly one") != std::string::npos);

    CHECK(refused(http, with(base, {{"VGI_SIGNING_KEY", std::nullopt}})).find("VGI_SIGNING_KEY") !=
          std::string::npos);
    const auto short_key = refused(http, with(base, {{"VGI_SIGNING_KEY", "short-key"}}));
    CHECK(short_key.find("32 bytes") != std::string::npos);
    CHECK(refused(http, with(base, {{"VGI_SIGNING_KEY_MINTED", "1"}})).find("VGI_SIGNING_KEY") !=
          std::string::npos);

    for (const auto& [name, value] : std::vector<std::pair<std::string, std::string>>{
             {"VGI_BEARER_TOKENS", "tok=alice"},
             {"VGI_OPTIONAL_BEARER_TOKENS", "tok=alice"},
             {"VGI_RPC_GRANT_KEYS", key},
             {"VGI_OAUTH_RESOURCE", "https://x.test"},
             {"VGI_PROXY_PROOF_MODE", "require"},
             {"VGI_JWT_ALGORITHMS", "ES256,HS256"},
             {"VGI_ORCHARD_HOSTED", "yes please"}}) {
        INFO(name);
        CHECK(refused(http, with(base, {{name, value}})).find(name) != std::string::npos);
    }

    CHECK(refused({"--http", "0", "--grant-key", key}, base).find("grant") != std::string::npos);
    // The fixture's identity hooks (resolve_token, mint_grant) and its
    // configure_http hook (a custom bearer authenticator) are other ways in.
    CHECK(refused({"--http", "0", "--conformance-identity"}, base).find("resolve_token") !=
          std::string::npos);
    CHECK(refused(http, with(base, {{"VGI_FIXTURE_TEST_BEARERS", "1"}})).find("configure_http") !=
          std::string::npos);

    // Transports that carry no Pass.
    CHECK(refused({}, base).find("only --http") != std::string::npos);
    CHECK(refused({"--unix", "/tmp/vgi-hosted-refuse.sock"}, base).find("only --http") !=
          std::string::npos);
    CHECK(refused({"--iroh-raw-upstream", "0", "--iroh-issuer", "x"}, base).find("only --http") !=
          std::string::npos);
    CHECK(refused({"--http", "0", "--iroh-issuer", "x"}, base).find("only --http") !=
          std::string::npos);
}
