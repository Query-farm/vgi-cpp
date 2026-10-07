# Hosted protocols and Identity

A worker always serves `vgi.v2`, the protocol the DuckDB extension speaks, and
`vgi_rpc.Reflection.v1`, which vgi-rpc hosts itself. It can serve two more
things from the same process and the same listener.

## Additional protocols: `Worker::set_hosted_protocols`

```cpp
vgi::Worker worker;
worker.set_hosted_protocols([] {
    vgi_rpc::ProtocolBuilder reports("acme.Reports.v1", "1.0.0");
    reports.add_unary("render", params, result, render_handler);
    return std::vector<vgi_rpc::ProtocolBuilder>{std::move(reports)};
});
worker.run(argc, argv);
```

The hook returns `(protocol, implementation)` pairs. Each one is a
`vgi_rpc::ProtocolBuilder`, which holds the wire name, an optional version, and
the handlers.

- **Called once**, when `run()` builds the server. It may read configuration
  or the environment, but the set it returns is fixed for the life of the
  process. That keeps reflection output and protocol hashes stable.
- **Hosted on every transport**: stdio, `--unix`, `--http`, and the Iroh raw
  upstream. They come after `vgi.v2`, in the order returned, so
  `list_protocols` reports `vgi.v2` first.
- **Routing is unchanged for `vgi.v2`.** Every request names its protocol, so
  hosting more protocols cannot change how the extension's calls are
  dispatched.
- **The protocol is the unit of optionality.** You cannot host part of a
  protocol. If a capability may be absent, make it its own protocol and
  return it from the hook or not.
- **Names are validated at startup**, and the error names the hook. A name must
  be a valid routing key, distinct from `vgi.v2` and from every other entry,
  and outside the reserved `vgi_rpc.` prefix. Reflection is automatic, and
  Identity is enabled as described below.

Errors raised by a hosted protocol carry the full vgi-rpc error model (code,
kind and details). Throw `vgi_rpc::StatusError` to choose them; see vgi-rpc's
`docs/api/errors.md`.

## `vgi_rpc.Identity.v1`: `set_resolve_token` / `set_mint_grant`

A reverse proxy that terminates the only public listener has to know who the
caller is before it can authorize anything. Identity lets the worker answer
that question:

```cpp
worker.set_resolve_token([](const std::string& token) -> std::optional<vgi_rpc::TokenIdentity> {
    try {
        auto row = api_keys.lookup(token);  // your own store
        if (!row) return std::nullopt;      // the store answered: unknown credential
        return vgi_rpc::TokenIdentity{row->principal, row->label, 300};
    } catch (const StoreUnreachable& e) {
        // "I could not find out" -- not "unknown".
        throw vgi_rpc::AuthUnavailableError(e.what(), /*retry_after=*/5);
    }
});
worker.set_introspect_principals({"edge-proxy"});
```

- **Absent unless a hook is set.** The protocol is not hosted at all, rather
  than hosted and refusing every call. That way, upgrading the SDK can never
  give an existing worker a credential-to-identity oracle. Only the methods
  whose hooks you set are hosted: `resolve_token` hosts `introspect_token`,
  and `mint_grant` hosts `issue_grant`.
- **HTTP only.** The allowlist is a list of *principals*, and only HTTP
  authenticates callers. stdio, unix sockets and TCP have no caller identity
  to check against it.
- **An allowlist is required.** Set it with `set_introspect_principals`, with
  `--introspect-principals a,b`, or with `VGI_INTROSPECT_PRINCIPALS`. There is
  no permissive default, and a worker that sets `resolve_token` without an
  allowlist **refuses to start** on `--http`. Authenticating and introspecting
  are different capabilities: if any authenticated caller may introspect, any
  user can resolve any other user's credential to its owner.

### Which error to throw for a transient failure

Return an empty optional when the store answered and the credential is
unknown.

When the answer cannot be known (a backing store is down, a request timed out,
or a remote authority returned a 5xx), throw
**`vgi_rpc::AuthUnavailableError(detail, retry_after)`**. This is the same
error an authenticator throws to get a `503`. The framework translates it into
`identity_unavailable` with `RetryInfo{retry_after}`. A caller can then tell an
outage from a refusal, will not negative-cache it, and knows when to ask
again. `vgi_rpc::IdentityUnavailableError` is also accepted.

Never throw `std::invalid_argument` for an outage. It reaches the wire as a
`ValueError`, which callers read as "your input was wrong; do not retry".

## Grants as bearer credentials

`issue_grant` mints a credential for unattended automation to present later as
an ordinary bearer. With **grant keys** configured the worker both mints and
accepts them, with no storage and no hook of your own:

```bash
# 32-byte keys, standard base64. The first mints; every key verifies.
export VGI_RPC_GRANT_KEYS="$(openssl rand -base64 32)"
export VGI_RPC_GRANT_AUDIENCE=prod-reports          # optional, default ""
export VGI_RPC_GRANT_MAX_TTL_SECONDS=86400          # optional, default 7 days
my-worker --http 8080
# or: my-worker --http 8080 --grant-key KEY [--grant-key OLD_KEY]
```

- **Opt-in.** No key, no change. With keys, `vgi_rpc.Identity.v1` is hosted on
  `--http` with `issue_grant`, and vgi-rpc mints sealed `vgig1.` grants unless
  you call `set_mint_grant`. A malformed or duplicate key, or a non-positive
  lifetime, **refuses to start** the worker. `Worker::set_grant_keys` sets them
  in code (`std::nullopt` turns grants off whatever argv and the environment
  say).
- **Minting needs a fresh login.** The caller must carry an `auth_time` claim
  newer than 15 minutes, which a static `VGI_BEARER_TOKENS` token does not.
  A grant itself carries none, so a grant can never mint another grant.
- **Accepting.** A request with `Authorization: Bearer <grant>` authenticates
  as the grant's owner: `domain = "grant"`, the owner's principal, and claims
  `{grant_id, scopes, purpose}`. A tampered, wrong-key, wrong-audience or
  expired grant is a `401` (`VGI-Auth-Reason: invalid_credential`, or
  `expired_credential`).
- **Rotation and revocation.** Put the new key first and keep the old one after
  it until every grant it minted has expired. Grants are not individually
  revocable: expiry is the revocation, and removing a key revokes everything it
  minted.

The `--http` worker authenticates in this order: `VGI_BEARER_TOKENS`, then
sealed grants, then your `resolve_token` hook (when set). Only a token with
the exact `vgig1.` prefix reaches the grant verifier, and one that does not
verify stops there: it is never handed to `resolve_token`. A token
`resolve_token` resolves authenticates as `domain = "token"`; an unknown one is
a `401`, and an outage (`AuthUnavailableError`) a `503` with your
`Retry-After`. Without `VGI_BEARER_TOKENS`, a request with no `Authorization`
header stays anonymous.

`VGI_OPTIONAL_BEARER_TOKENS` (same `token=principal,…` format) takes its place
for a server that serves anonymous and bearer-identified callers side by side,
as the vgi-python and vgi-rust fixtures do. A known token is its principal, and
no token or an unknown one is anonymous, never a `401`. A `vgig1.` token still
goes to the grant verifier. `VGI_BEARER_TOKENS` wins when both are set.

The Iroh bridge (`--iroh-issuer`) authenticates from forwarded peer evidence.
Accepting a grant beside it would bypass that requirement, so a worker with
grant keys or `resolve_token` refuses to start there.

`--access-log PATH` writes vgi-rpc's JSON-lines access log, whose `principal`
and `auth_domain` fields show how each call was authenticated.

## Attach tickets

With `VGI_SIGNING_KEY` set as well as grant keys (or a `set_mint_grant`
minter), `--http` also hosts `vgi.attach_tickets.v1`, whose `seal_attach` seals
a user's ATTACH into a ticket a runner redeems with that user's grant. See
[Attach tickets](attach-tickets.md).

## Checking a worker

vgi-rpc's hosted-protocols group checks all of the above on every transport:

```bash
pip install "vgi-rpc[http,conformance]" pytest pytest-timeout
vgi-rpc-test-hosted --cmd build-release/example-worker/vgi-example-worker \
    --expect vgi.v2,conformance.Secondary.v1
vgi-rpc-test-hosted --unix /tmp/vgi.sock --expect vgi.v2,conformance.Secondary.v1
vgi-rpc-test-hosted --url http://127.0.0.1:PORT \
    --expect vgi.v2,conformance.Secondary.v1 --identity
```

The example worker hosts `conformance.Secondary.v1` through the hook. With
`--conformance-identity` (or `VGI_FIXTURE_IDENTITY=1`) on `--http`, plus
`--introspect-principals conformance-introspector`, it also
hosts Identity under vgi-rpc's conformance fixture policy, including its
spoofable `X-Conformance-Principal` header authentication. That flag is for
tests only. `--conformance-principal-header` turns on only that header
authentication, with no identity hooks, so `--grant-key` exercises the
framework's own minter (`tests/grant_auth_test.cpp`).
