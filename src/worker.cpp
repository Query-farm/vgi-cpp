// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "vgi/worker.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/compute/initialize.h>
#include <vgi_rpc/http_config.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/identity.h>
#include <vgi_rpc/iroh_identity.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/token_identity.h>

#include "dispatcher.h"
#include "landing.h"
#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"

namespace vgi {

namespace {
// The generated headers carry the namespace of their original consumer, the
// DuckDB extension.  Alias rather than post-process generated output.
namespace gen = ::vgi::generated;

std::pair<std::string, int> parse_tcp_bind(const std::string& value, const char* flag) {
    std::string host = "127.0.0.1";
    std::string port_text = value;
    const auto split = value.rfind(':');
    if (split != std::string::npos) {
        host = value.substr(0, split);
        port_text = value.substr(split + 1);
        if (host.empty()) host = "127.0.0.1";
        if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
            host = host.substr(1, host.size() - 2);
        }
    }
    char* end = nullptr;
    const long parsed = std::strtol(port_text.c_str(), &end, 10);
    if (end == port_text.c_str() || *end != '\0' || parsed < 0 || parsed > 65535) {
        throw std::invalid_argument(std::string(flag) + " needs [HOST:]PORT in 0..65535");
    }
    return {host, static_cast<int>(parsed)};
}

bool is_loopback_bind(const std::string& host) {
    return host == "127.0.0.1" || host == "::1" || host == "localhost";
}

std::map<std::string, std::string> bearer_tokens_from_env() {
    std::map<std::string, std::string> tokens;
    const char* configured = std::getenv("VGI_BEARER_TOKENS");
    if (!configured || !*configured) return tokens;
    std::string entries(configured);
    size_t begin = 0;
    while (begin <= entries.size()) {
        const auto end = entries.find(',', begin);
        const auto entry = entries.substr(begin, end == std::string::npos ? end : end - begin);
        const auto separator = entry.find('=');
        if (separator == std::string::npos || separator == 0 || separator + 1 == entry.size()) {
            throw std::invalid_argument(
                "VGI_BEARER_TOKENS must contain comma-separated token=principal entries");
        }
        tokens.emplace(entry.substr(0, separator), entry.substr(separator + 1));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return tokens;
}

void configure_bearer_auth(vgi_rpc::HttpConfig& config,
                           const std::map<std::string, std::string>& tokens) {
    if (tokens.empty()) return;
    config.peer_identity_providers.push_back(
        [tokens](const vgi_rpc::PeerResolutionContext& context) {
            const auto authorization = context.header("authorization");
            constexpr const char* kPrefix = "Bearer ";
            if (!authorization || authorization->rfind(kPrefix, 0) != 0) {
                throw vgi_rpc::PeerIdentityRejected("missing bearer credential");
            }
            const auto found =
                tokens.find(authorization->substr(std::char_traits<char>::length(kPrefix)));
            if (found == tokens.end()) {
                throw vgi_rpc::PeerIdentityRejected("invalid bearer credential");
            }
            return vgi_rpc::PeerIdentityResult::available(vgi_rpc::PeerIdentity(
                "bearer", "authorization", vgi_rpc::IdentityAssurance::CONFIGURED_PROXY,
                "vgi-worker", "http", vgi_rpc::PeerSubjectKind::USER, found->second,
                vgi_rpc::SubjectStability::STABLE, /*subject_verified=*/true));
        });
    config.peer_authentication_policy = vgi_rpc::peer_identity_primary("bearer");
}

void configure_signing_key(vgi_rpc::HttpConfig& config) {
    const char* configured = std::getenv("VGI_SIGNING_KEY");
    if (!configured || !*configured) return;

    const std::string raw(configured);
    if (raw.size() == config.token_key.size()) {
        std::copy(raw.begin(), raw.end(), config.token_key.begin());
        return;
    }
    vgi_rpc::crypto::Sha256 hash;
    hash.update(raw);
    config.token_key = hash.digest();
}
}  // namespace

Worker::Worker() : disp_(std::make_unique<Dispatcher>()) {}
Worker::~Worker() = default;
Worker::Worker(Worker&&) noexcept = default;
Worker& Worker::operator=(Worker&&) noexcept = default;

void Worker::set_catalog(CatalogModel catalog) {
    disp_->set_catalog(std::move(catalog));
}

CatalogModel& Worker::catalog() {
    return disp_->catalog();
}

CatalogModel& Worker::catalog(const std::string& name) {
    return disp_->catalog(name);
}

void Worker::hide_function(std::string name) {
    disp_->hide_function(std::move(name));
}

void Worker::set_server_id(std::string id) {
    server_id_ = std::move(id);
}

void Worker::register_scalar(std::shared_ptr<ScalarFunction> fn) {
    disp_->register_scalar(std::move(fn));
}

void Worker::register_scalar_in(std::string catalog, std::string schema,
                                std::shared_ptr<ScalarFunction> fn) {
    register_scalar_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Worker::register_scalar_in(std::string catalog, SchemaPath schema_path,
                                std::shared_ptr<ScalarFunction> fn) {
    disp_->register_scalar_in(std::move(catalog), std::move(schema_path), std::move(fn));
}

void Worker::register_table(std::shared_ptr<TableFunction> fn) {
    disp_->register_table(std::move(fn));
}

void Worker::register_table_in(std::string catalog, std::string schema,
                               std::shared_ptr<TableFunction> fn) {
    register_table_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Worker::register_table_in(std::string catalog, SchemaPath schema_path,
                               std::shared_ptr<TableFunction> fn) {
    disp_->register_table_in(std::move(catalog), std::move(schema_path), std::move(fn));
}

void Worker::register_copy_to(std::shared_ptr<CopyToFunction> writer) {
    disp_->register_copy_to(std::move(writer));
}

void Worker::register_copy_from(std::shared_ptr<CopyFromFunction> reader) {
    disp_->register_copy_from(std::move(reader));
}

void Worker::register_buffering(std::shared_ptr<TableBufferingFunction> fn) {
    disp_->register_buffering(std::move(fn));
}

void Worker::register_buffering_in(std::string catalog, std::string schema,
                                   std::shared_ptr<TableBufferingFunction> fn) {
    register_buffering_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Worker::register_buffering_in(std::string catalog, SchemaPath schema_path,
                                   std::shared_ptr<TableBufferingFunction> fn) {
    disp_->register_buffering_in(std::move(catalog), std::move(schema_path), std::move(fn));
}

void Worker::register_aggregate(std::shared_ptr<AggregateFunction> fn) {
    disp_->register_aggregate(std::move(fn));
}

void Worker::register_aggregate_in(std::string catalog, std::string schema,
                                   std::shared_ptr<AggregateFunction> fn) {
    register_aggregate_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Worker::register_aggregate_in(std::string catalog, SchemaPath schema_path,
                                   std::shared_ptr<AggregateFunction> fn) {
    disp_->register_aggregate_in(std::move(catalog), std::move(schema_path), std::move(fn));
}

void Worker::register_table_in_out(std::shared_ptr<TableInOutFunction> fn) {
    disp_->register_table_in_out(std::move(fn));
}

void Worker::register_table_in_out_in(std::string catalog, std::string schema,
                                      std::shared_ptr<TableInOutFunction> fn) {
    register_table_in_out_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Worker::register_table_in_out_in(std::string catalog, SchemaPath schema_path,
                                      std::shared_ptr<TableInOutFunction> fn) {
    disp_->register_table_in_out_in(std::move(catalog), std::move(schema_path), std::move(fn));
}

void Worker::set_hosted_protocols(HostedProtocolsHook hook) {
    hosted_protocols_ = std::move(hook);
}

void Worker::set_resolve_token(vgi_rpc::ResolveTokenHook hook) {
    resolve_token_ = std::move(hook);
}

void Worker::set_mint_grant(vgi_rpc::MintGrantHook hook) {
    mint_grant_ = std::move(hook);
}

void Worker::set_introspect_principals(std::vector<std::string> principals) {
    introspect_principals_ = std::move(principals);
}

void Worker::configure_http(std::function<void(vgi_rpc::HttpConfig&)> hook) {
    configure_http_ = std::move(hook);
}

namespace {

std::vector<std::string> split_principals(const std::string& raw) {
    std::vector<std::string> out;
    size_t begin = 0;
    while (begin <= raw.size()) {
        const auto end = raw.find(',', begin);
        auto item = raw.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const auto first = item.find_first_not_of(" \t");
        const auto last = item.find_last_not_of(" \t");
        if (first != std::string::npos) out.push_back(item.substr(first, last - first + 1));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return out;
}

bool valid_routing_key(const std::string& name) {
    if (name.empty() || name.size() > 255) return false;
    const auto lead = static_cast<unsigned char>(name[0]);
    if (!(std::isalpha(lead) || lead == '_')) return false;
    return std::all_of(name.begin() + 1, name.end(),
                       [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '.'; });
}

}  // namespace

std::unique_ptr<vgi_rpc::Server> Worker::build_server(Transport transport,
                                                      const std::vector<std::string>& args) {
    // The `bad_protocol` fixture advertises an incompatible version through
    // this override, so the engine's ATTACH fails with a clear mismatch rather
    // than somewhere later in the query.
    const char* override_version = std::getenv("VGI_PROTOCOL_VERSION_OVERRIDE");
    vgi_rpc::ServerBuilder builder;
    // The wire name is the routing key every request must carry: dispatch
    // resolves the pair (protocol, method), and over HTTP the name is also the
    // path segment (`{prefix}/vgi.v2/{method}`). It is generated from
    // vgi-python rather than spelled here, because a worker declaring any
    // other name answers ProtocolNotSupported to every client that exists.
    // Only the version is overridable: `bad_protocol` must reach the version
    // gate it exists to test, which a renamed protocol never would.
    const std::string primary_name(gen::VGI_PROTOCOL_NAME);
    builder.protocol(primary_name)
        .protocol_version(override_version && *override_version
                              ? std::string(override_version)
                              : std::string(gen::VGI_PROTOCOL_VERSION));
    if (!server_id_.empty()) builder.server_id(server_id_);
    disp_->install(builder);

    // The worker's hosted protocols: the same set on every transport, so the
    // hook is consulted here, in the one builder, and nowhere else.  vgi-rpc
    // validates these too, but its messages name `add_protocol` rather than
    // the worker hook that supplied the protocol; checking first lets the
    // error say what to fix.
    if (hosted_protocols_) {
        const std::string owner = "Worker::set_hosted_protocols hook";
        std::vector<vgi_rpc::ProtocolBuilder> hosted = hosted_protocols_();
        std::set<std::string> seen;
        for (size_t i = 0; i < hosted.size(); ++i) {
            const std::string& name = hosted[i].name();
            const std::string where = owner + ", entry " + std::to_string(i);
            if (name.rfind(vgi_rpc::kReservedProtocolPrefix, 0) == 0) {
                throw std::invalid_argument(
                    where + ": '" + name + "' claims the reserved '" +
                    vgi_rpc::kReservedProtocolPrefix +
                    "' prefix. Framework protocols are not supplied through this hook: "
                    "reflection is hosted automatically, and vgi_rpc.Identity.v1 is enabled "
                    "with Worker::set_resolve_token / Worker::set_mint_grant.");
            }
            if (!valid_routing_key(name)) {
                throw std::invalid_argument(where + ": '" + name +
                                            "' is not a valid protocol name "
                                            "([A-Za-z_][A-Za-z0-9_.]*, at most 255 bytes)");
            }
            if (name == primary_name) {
                throw std::invalid_argument(where + ": '" + name +
                                            "' is the worker's own protocol; give the hosted "
                                            "protocol a distinct name");
            }
            if (!seen.insert(name).second) {
                throw std::invalid_argument(owner + " lists protocol '" + name +
                                            "' twice; the name is the routing key, so each "
                                            "hosted protocol needs a distinct one");
            }
        }
        for (auto& protocol : hosted) builder.add_protocol(std::move(protocol));
    }

    // vgi_rpc.Identity.v1, on the transport that authenticates callers.  Its
    // allowlist is a list of principals, which stdio, unix and TCP do not
    // have, so it is not hosted there.  Absent unless a hook is set: hosted
    // and refusing would still be an oracle a dependency upgrade grew.
    if (transport == Transport::HTTP && (resolve_token_ || mint_grant_)) {
        vgi_rpc::IdentityOptions options;
        options.resolve_token = resolve_token_;
        options.mint_grant = mint_grant_;
        if (resolve_token_) {
            std::vector<std::string> principals;
            if (introspect_principals_) {
                principals = *introspect_principals_;
            } else {
                for (size_t i = 0; i + 1 < args.size(); ++i) {
                    if (args[i] == "--introspect-principals")
                        principals = split_principals(args[i + 1]);
                }
                if (principals.empty()) {
                    const char* env = std::getenv("VGI_INTROSPECT_PRINCIPALS");
                    if (env != nullptr) principals = split_principals(env);
                }
            }
            principals.erase(std::remove_if(principals.begin(), principals.end(),
                                            [](const std::string& p) { return p.empty(); }),
                             principals.end());
            if (principals.empty()) {
                // Fail closed and loud.  There is no permissive default: a
                // worker that resolves credentials and forgot the allowlist
                // must not start.
                throw std::invalid_argument(
                    "this worker sets a resolve_token hook, which hosts the vgi_rpc.Identity.v1 "
                    "protocol, but no introspector allowlist was configured. Set "
                    "VGI_INTROSPECT_PRINCIPALS (comma-separated), pass --introspect-principals, "
                    "or call Worker::set_introspect_principals. There is no permissive default "
                    "on purpose: introspection is a separate capability from authentication, "
                    "and allowing every authenticated caller lets any user resolve any other "
                    "user's credential to its owner.");
            }
            options.introspect_principals = std::move(principals);
        }
        builder.identity(std::make_shared<vgi_rpc::IdentityImpl>(std::move(options)));
    }

    return builder.build();
}

void Worker::run(int argc, char** argv) {
    // Arrow's compute kernels register themselves from a translation unit
    // nothing here references, so linking statically drops it and `add`,
    // `multiply` and friends are simply absent from the registry at runtime —
    // while `cast`, which lives elsewhere, keeps working. That asymmetry makes
    // it read like a missing feature flag rather than a linker artifact.
    if (auto status = arrow::compute::Initialize(); !status.ok()) {
        throw std::runtime_error("cannot initialize Arrow compute: " + status.ToString());
    }

    // Transport from argv, matching the Rust and Python workers so one
    // wrapper script can drive any of them.
    std::vector<std::string> args(argv + 1, argv + argc);
    // A malformed transport argument is a startup error, not a reason to fall
    // through to stdio: `--unix` with no path served the pipe transport on a
    // socket the caller was already waiting on, which reads as a hang.
    const auto refuse = [](const std::string& message) {
        std::fprintf(stderr, "vgi worker: %s\n", message.c_str());
        std::exit(2);
    };
    // A server that cannot be built -- a malformed hosted protocol, Identity
    // without an allowlist -- is a startup error, reported and refused.
    const auto build_or_refuse = [&](Transport transport) {
        try {
            return build_server(transport, args);
        } catch (const std::exception& error) {
            refuse(error.what());
        }
        return std::unique_ptr<vgi_rpc::Server>();
    };
    std::string iroh_upstream;
    std::string iroh_issuer;
    std::string http_host = "127.0.0.1";
    int configured_http_port = -1;
    std::vector<std::string> iroh_trusted_proxies;
    bool iroh_observe = false;
    const auto parse_http_port = [&refuse](const std::string& value) {
        char* end = nullptr;
        const long parsed = std::strtol(value.c_str(), &end, 10);
        if (end == value.c_str() || *end != '\0' || parsed < 0 || parsed > 65535) {
            refuse("HTTP port must be in 0..65535, got '" + value + "'");
        }
        return static_cast<int>(parsed);
    };
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--iroh-raw-upstream") {
            if (i + 1 >= args.size()) refuse("--iroh-raw-upstream needs [HOST:]PORT");
            iroh_upstream = args[++i];
        } else if (args[i] == "--iroh-issuer") {
            if (i + 1 >= args.size()) refuse("--iroh-issuer needs a value");
            iroh_issuer = args[++i];
        } else if (args[i] == "--iroh-trusted-proxy") {
            if (i + 1 >= args.size()) refuse("--iroh-trusted-proxy needs an exact IP address");
            iroh_trusted_proxies.push_back(args[++i]);
        } else if (args[i] == "--iroh-observe") {
            iroh_observe = true;
        } else if (args[i] == "--host") {
            if (i + 1 >= args.size()) refuse("--host needs a value");
            http_host = args[++i];
        } else if (args[i] == "--port") {
            if (i + 1 >= args.size()) refuse("--port needs a value");
            configured_http_port = parse_http_port(args[++i]);
        }
    }
    if (!iroh_upstream.empty()) {
        if (iroh_issuer.empty()) refuse("--iroh-raw-upstream requires --iroh-issuer");
        if (iroh_trusted_proxies.empty()) iroh_trusted_proxies.push_back("127.0.0.1");
        try {
            const auto [host, port] = parse_tcp_bind(iroh_upstream, "--iroh-raw-upstream");
            auto server = build_server(Transport::TCP, args);
            if (!is_loopback_bind(host)) {
                refuse("--iroh-raw-upstream must bind loopback; expose only the Iroh bridge");
            }
            vgi_rpc::TcpServerOptions options;
            options.proxy_protocol_v2_required = true;
            options.trusted_proxy_addresses = std::move(iroh_trusted_proxies);
            options.iroh_proxy_issuer = std::move(iroh_issuer);
            options.peer_authentication_policy = iroh_observe
                                                     ? vgi_rpc::observe_peer_identity
                                                     : vgi_rpc::peer_identity_primary("iroh");
            server->serve_tcp(host, port, options);
        } catch (const std::exception& error) {
            refuse(error.what());
        }
        std::exit(0);
    }
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--unix") {
            if (i + 1 >= args.size()) refuse("--unix needs a socket path");
            build_or_refuse(Transport::UNIX)->serve_unix(args[i + 1]);
            std::exit(0);
        }
        if (args[i] == "--http") {
            int port = configured_http_port < 0 ? 0 : configured_http_port;
            if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) {
                if (configured_http_port >= 0) refuse("use either --http PORT or --port, not both");
                port = parse_http_port(args[i + 1]);
            }
            auto server = build_or_refuse(Transport::HTTP);
            if (iroh_issuer.empty()) {
                vgi_rpc::HttpConfig config;
                config.host = http_host;
                config.port = port;
                configure_landing(config, disp_->catalog().name, server->server_id(),
                                  disp_->catalog().comment.value_or(""));
                configure_signing_key(config);
                configure_bearer_auth(config, bearer_tokens_from_env());
                if (configure_http_) configure_http_(config);
                disp_->set_split_token_signing_key(config.token_key);
                server->serve_http(config);
            } else {
                if (!is_loopback_bind(http_host)) {
                    refuse(
                        "Iroh HTTP bridge upstream must bind loopback; expose only the Iroh "
                        "bridge");
                }
                if (iroh_trusted_proxies.empty()) iroh_trusted_proxies.push_back("127.0.0.1");
                vgi_rpc::HttpConfig config;
                config.host = http_host;
                config.port = port;
                configure_landing(config, disp_->catalog().name, server->server_id(),
                                  disp_->catalog().comment.value_or(""));
                configure_signing_key(config);
                config.peer_identity_providers.push_back(vgi_rpc::iroh_forwarded_header_provider(
                    {std::move(iroh_issuer), std::move(iroh_trusted_proxies)}));
                config.peer_authentication_policy = iroh_observe
                                                        ? vgi_rpc::observe_peer_identity
                                                        : vgi_rpc::peer_identity_primary("iroh");
                if (configure_http_) configure_http_(config);
                disp_->set_split_token_signing_key(config.token_key);
                server->serve_http(config);
            }
            std::exit(0);
        }
    }

    // stdout is the Arrow-IPC channel; anything a worker wants to say goes to
    // stderr or it corrupts the stream.
    build_or_refuse(Transport::PIPE)->run();
    std::exit(0);
}

}  // namespace vgi
