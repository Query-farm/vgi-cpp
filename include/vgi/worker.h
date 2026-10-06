// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <vgi_rpc/http_config.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/token_identity.h>

#include "vgi/catalog.h"
#include "vgi/function.h"
#include "vgi/table_function.h"
#include "vgi/aggregate.h"
#include "vgi/buffering.h"
#include "vgi/copy_from.h"
#include "vgi/copy_to.h"
#include "vgi/table_in_out.h"

namespace vgi {

class Dispatcher;

// A VGI worker: register functions, then run().
//
// `run()` parses argv and serves until the engine disconnects — stdio by
// default, `--unix <path>` for the pooled launcher, `--http` for a standalone
// server, or `--iroh-raw-upstream <port> --iroh-issuer <namespace>` behind
// `vgi-iroh-bridge`. `--http --iroh-issuer ...` enables the same trusted
// bridge identity boundary for HTTP-over-Iroh. It does not return.
class Worker {
public:
    Worker();
    ~Worker();
    Worker(Worker&&) noexcept;
    Worker& operator=(Worker&&) noexcept;

    void set_catalog(CatalogModel catalog);
    // Mutable access, so a worker can declare tables and views after
    // registering the functions that back them.
    CatalogModel& catalog();
    // A second catalog this worker serves, created on first use.
    //
    // One process may serve several, and which one a call belongs to is
    // decided by the attachment, not by the process: two catalogs may declare
    // the same schema and the same function name.
    CatalogModel& catalog(const std::string& name);
    void set_server_id(std::string id);

    // Keep `name` out of the function surface the catalog advertises.
    //
    // A function that exists only to back a catalog table is still registered
    // — the engine calls it by name to scan the table — but it is not
    // something a user should find in `duckdb_functions()`, and a worker whose
    // surface is a cross-language contract has to be able to say so.
    void hide_function(std::string name);

    // Register in the catalog's default schema (`main`).
    void register_scalar(std::shared_ptr<ScalarFunction> fn);

    // Register in a named schema of a named catalog.
    //
    // A function's identity is (catalog, schema, name), not name alone: the
    // same name may be declared in two schemas with different implementations,
    // and a schema-qualified call has to reach the right one rather than
    // resolving as an ambiguous overload. Declaring a schema here also creates
    // it if the catalog does not list it.
    void register_scalar_in(std::string catalog, std::string schema,
                            std::shared_ptr<ScalarFunction> fn);
    void register_scalar_in(std::string catalog, SchemaPath schema_path,
                            std::shared_ptr<ScalarFunction> fn);

    void register_table(std::shared_ptr<TableFunction> fn);
    void register_table_in(std::string catalog, std::string schema,
                           std::shared_ptr<TableFunction> fn);
    void register_table_in(std::string catalog, SchemaPath schema_path,
                           std::shared_ptr<TableFunction> fn);

    void register_table_in_out(std::shared_ptr<TableInOutFunction> fn);
    void register_table_in_out_in(std::string catalog, std::string schema,
                                  std::shared_ptr<TableInOutFunction> fn);
    void register_table_in_out_in(std::string catalog, SchemaPath schema_path,
                                  std::shared_ptr<TableInOutFunction> fn);

    void register_aggregate(std::shared_ptr<AggregateFunction> fn);
    void register_aggregate_in(std::string catalog, std::string schema,
                               std::shared_ptr<AggregateFunction> fn);
    void register_aggregate_in(std::string catalog, SchemaPath schema_path,
                               std::shared_ptr<AggregateFunction> fn);

    // Serve a DDL-capable in-memory catalog (see MemoryCatalogOptions in
    // vgi/catalog.h) beside this worker's declared catalogs. Each ATTACH of it
    // is private and starts from an empty `main` schema.
    void register_memory_catalog(MemoryCatalogOptions options);

    void register_buffering(std::shared_ptr<TableBufferingFunction> fn);

    // A `COPY … TO (FORMAT …)` writer. Registers its handler as a buffering
    // function too, since that is the RPC path the engine drives it over.
    void register_copy_to(std::shared_ptr<CopyToFunction> writer);

    // A `COPY … FROM (FORMAT …)` reader. Registers its handler as a table
    // function too, since that is the RPC path the engine drives it over.
    void register_copy_from(std::shared_ptr<CopyFromFunction> reader);

    void register_buffering_in(std::string catalog, std::string schema,
                               std::shared_ptr<TableBufferingFunction> fn);
    void register_buffering_in(std::string catalog, SchemaPath schema_path,
                               std::shared_ptr<TableBufferingFunction> fn);

    // Additional vgi-rpc protocols this worker hosts beside `vgi.v2`.
    //
    // The hook returns `(protocol, implementation)` pairs -- here one
    // `vgi_rpc::ProtocolBuilder` each: the protocol's wire name and version,
    // and the handlers that implement it.  It is called **once**, when
    // `run()` builds the server, and its answer is hosted on **every**
    // transport the worker serves (stdio, `--unix`, `--http`, the Iroh
    // upstream), after `vgi.v2` and in the order returned -- so
    // `vgi_rpc.Reflection.v1/list_protocols` reports `vgi.v2` first, then
    // these.  It may consult configuration or the environment, but the set is
    // fixed for the life of the process, so reflection output and protocol
    // hashes stay stable.
    //
    // The protocol is the unit of optionality: there is no way to host part
    // of a protocol.  A capability that may be absent is its own protocol,
    // returned here or not.
    //
    // Requests are routed on their `vgi_rpc.protocol` key, so a hosted
    // protocol never changes how a `vgi.v2` request is dispatched.
    //
    // Each name must be a valid routing key, distinct from `vgi.v2` and from
    // every other entry, and outside the reserved `vgi_rpc.` prefix --
    // reflection is hosted automatically, and `vgi_rpc.Identity.v1` is
    // enabled with `set_resolve_token` / `set_mint_grant`.  A violation is a
    // startup error naming this hook.
    using HostedProtocolsHook = std::function<std::vector<vgi_rpc::ProtocolBuilder>()>;
    void set_hosted_protocols(HostedProtocolsHook hook);

    // Host `vgi_rpc.Identity.v1`'s `introspect_token`: resolve an opaque
    // bearer credential to the identity it authenticates as, for a reverse
    // proxy that must know the caller before it can authorize anything.
    //
    // Absent unless set -- not hosted-and-refusing -- so upgrading the SDK
    // never grows a credential-to-identity oracle on an existing worker.
    // Hosted over `--http` only: its allowlist is a list of *principals*,
    // which stdio, unix sockets and TCP do not have.
    //
    // Requires an allowlist of principals permitted to ask
    // (`set_introspect_principals`, `--introspect-principals`, or
    // `VGI_INTROSPECT_PRINCIPALS`).  There is no permissive default, and an
    // HTTP worker that sets this hook without one **refuses to start**:
    // authenticating and introspecting are different capabilities, and "any
    // authenticated caller" lets any user resolve any other user's credential
    // to its owner.
    //
    // Return an empty optional for "the store answered and this credential is
    // unknown".  For "the answer is not knowable" -- a backing store is down,
    // a timeout, a 5xx from a remote authority -- throw
    // `vgi_rpc::AuthUnavailableError(detail, retry_after)`, the same error
    // an authenticator throws for a 503.  The framework translates it into
    // `identity_unavailable` carrying your `retry_after` as `RetryInfo`, so a
    // caller can tell an outage from a refusal and knows when to ask again.
    // Never throw `std::invalid_argument` for an outage: it reaches the wire
    // as a `ValueError`, which a caller reads as "your input was wrong".
    void set_resolve_token(vgi_rpc::ResolveTokenHook hook);

    // Host `vgi_rpc.Identity.v1`'s `issue_grant`: mint a standing delegation
    // credential for the *calling* user (there is no subject parameter).
    // Hosted over `--http` only, absent unless set.  The same rule as above
    // for transient failures: throw `vgi_rpc::AuthUnavailableError`.
    void set_mint_grant(vgi_rpc::MintGrantHook hook);

    // Principals permitted to call `introspect_token`.  When unset, `run()`
    // reads `--introspect-principals a,b` and then `VGI_INTROSPECT_PRINCIPALS`.
    void set_introspect_principals(std::vector<std::string> principals);

    // Adjust the HTTP configuration `run()` builds for `--http`, after its own
    // settings (landing page, signing key, bearer auth) are applied.
    void configure_http(std::function<void(vgi_rpc::HttpConfig&)> hook);

    // Serve, selecting the transport from argv.  Never returns.
    [[noreturn]] void run(int argc, char** argv);

private:
    // The transports `run()` serves.  Only HTTP changes what is hosted.
    enum class Transport { PIPE, UNIX, TCP, HTTP };

    // The one place this worker's server is built, for every transport: the
    // vgi.v2 protocol, the hosted protocols, and -- on HTTP, when a hook is
    // set -- vgi_rpc.Identity.v1.  Reflection is hosted by vgi-rpc itself.
    std::unique_ptr<vgi_rpc::Server> build_server(Transport transport,
                                                  const std::vector<std::string>& args);

    std::unique_ptr<Dispatcher> disp_;
    std::string server_id_;
    HostedProtocolsHook hosted_protocols_;
    vgi_rpc::ResolveTokenHook resolve_token_;
    vgi_rpc::MintGrantHook mint_grant_;
    std::optional<std::vector<std::string>> introspect_principals_;
    std::function<void(vgi_rpc::HttpConfig&)> configure_http_;
};

}  // namespace vgi
