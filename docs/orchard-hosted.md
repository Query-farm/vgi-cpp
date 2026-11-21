# Orchard hosted mode

`VGI_ORCHARD_HOSTED=1` runs the worker as an Orchard *Field Worker*. The
hosted base image sets it; you do not set it per deployment, and a worker built
on this SDK needs no code for it: `Worker::run` reads the environment. Callers
present a *Pass*: a short-lived [RFC 9068](https://www.rfc-editor.org/rfc/rfc9068)
JWT access token issued by the Orchard Front Desk for one canonical Desk
Address. The worker verifies every Pass locally and accepts no other
credential. vgi-python's `vgi/hosted.py` is the reference; the design is in the
extension repository's `docs/location-delegation.md`.

```bash
VGI_ORCHARD_HOSTED=1 \
VGI_JWT_ISSUER=https://auth.queryfarm.cloud \
VGI_JWT_AUDIENCE=https://desk.queryfarm.cloud/d/dep_123 \
VGI_JWT_JWKS_URI=https://auth.queryfarm.cloud/.well-known/jwks.json \
VGI_SIGNING_KEY=... \
my-worker --http --host 0.0.0.0 --port 8080
```

## Startup fails closed

The worker prints `Error: VGI_ORCHARD_HOSTED=1 (Orchard hosted mode): ...` and
exits with status 1, before anything listens, unless:

- `VGI_JWT_ISSUER`, `VGI_JWT_AUDIENCE` and `VGI_JWT_JWKS_URI` are all set.
  Unset and empty are both errors. Each takes exactly one value: a value
  containing a comma is refused.
- `VGI_SIGNING_KEY` is set and at least 32 bytes. It seals attach and
  transaction values and HTTP state tokens, so every machine of a deployment
  needs the same one, kept across restarts.
- the transport is `--http`. Stdio, `--unix` and the Iroh flags
  (`--iroh-raw-upstream`, `--iroh-issuer`, ...) carry no Pass.

It also refuses to start when any other way in is configured:

- `VGI_BEARER_TOKENS` or `VGI_OPTIONAL_BEARER_TOKENS`;
- sealed-grant keys (`VGI_RPC_GRANT_KEYS`, `--grant-key` or
  `Worker::set_grant_keys`);
- a `Worker::set_resolve_token` or `Worker::set_mint_grant` hook;
- `VGI_OAUTH_RESOURCE` (a Field Worker never advertises `resource_metadata`);
- `VGI_PROXY_PROOF_MODE` other than `off`;
- a `Worker::configure_http` hook that replaces the Pass verifier
  (`HttpConfig::bearer_authenticate`), relaxes the checks, or turns on the
  conformance header authentication, a proxy proof or peer identity;
- `VGI_JWT_ALGORITHMS` naming `none`, an `HS*` algorithm, or one this SDK
  cannot verify;
- a `VGI_ORCHARD_HOSTED` value other than `1`/`0`/`true`/`false`/`yes`/`no`/
  `on`/`off` (any case).

## Verification

- algorithm `ES256` only. `VGI_JWT_ALGORITHMS` (comma-separated) may name
  others from `ES256`, `ES384`, `ES512`, `RS256`, `RS384`, `RS512`, `PS256`,
  `PS384`, `PS512`; `none` and `HS*` are always refused;
- the JOSE header `typ` must be `at+jwt` (case-insensitive, `application/`
  prefix optional), and a `crit` header is refused;
- `iss` must equal `VGI_JWT_ISSUER`, and `aud` must be exactly one value (a
  string or a one-element list) equal to `VGI_JWT_AUDIENCE`;
- `exp` is required; `exp` and `nbf` are checked with 60 seconds of clock skew.
  A lapsed Pass is `expired_credential`, every other refusal
  `invalid_credential`;
- `sub` is required: it is the caller's principal;
- the JWKS is cached for at most 10 minutes, or less if its
  `Cache-Control: max-age` is shorter. When that time is up the keys are
  discarded. If the re-fetch fails, requests get a 503 with `Retry-After` until
  it succeeds; stale keys are never used;
- a Pass with an unknown `kid` triggers a JWKS re-fetch at most once every 60
  seconds. Between re-fetches it is simply rejected.

The JWKS is fetched with libcurl, without following redirects. `SSL_CERT_FILE`
and `SSL_CERT_DIR` point it at a CA bundle when the image keeps one somewhere
libcurl does not look.

## Every request needs a Pass

Except `HEAD /health` (the Front Desk's liveness probe) and CORS preflights,
which browsers send without credentials. That includes `GET /health`,
`OPTIONS /health`, the landing page, the browser client and unknown paths. A
request without a valid Pass gets:

```text
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer error="invalid_token"
VGI-Auth-Reason: missing_credential | invalid_credential | expired_credential
VGI-Supported-Encodings: ...        (and the other VGI capability headers)
```

There is no `resource_metadata` in the challenge. A hosted worker always has a
request cap: `VGI_RPC_MAX_REQUEST_BYTES`, or 32 MiB when unset, advertised as
`VGI-Max-Request-Bytes`; a larger body is 413. The check runs before the
request body is read, size-checked or decompressed (a client waiting on
`Expect: 100-continue` is refused instead of told to go ahead). CORS is open
(`Access-Control-Allow-Origin: *`), because the only credential is the bearer
token, which a browser never attaches on its own. The `Authorization` header is
never logged; a refusal is logged by its category only.

## Claims in function code

A function reads the caller from `ProcessParams::auth`: `domain` is
`"orchard"`, `principal` is the Pass's `sub`, and `claims` is the verified claim
set exactly as decoded (`sub`, `kind`, `email`, `email_verified`, `name`,
`org_id`, `org_name`, `federated_iss`, `federated_sub`, `client_id`, `iat`,
`nbf`, `exp`, `jti`, ...), with JSON types intact: `email_verified` is a
boolean and `exp` an integer.

```cpp
std::shared_ptr<arrow::RecordBatch> process(const vgi::ProcessParams& params,
                                            const std::shared_ptr<arrow::RecordBatch>& batch) const override {
    const auto& claims = params.auth.claims;
    const std::string org = claims.value("org_id", "");
    ...
}
```

## Testing

The example worker registers `pass_claims(x BIGINT) -> VARCHAR` when hosted
mode is on: each row is `{"principal", "domain", "claims"}` as JSON. It is the
fixture function of the SDK-neutral hosted-mode harness in the extension
repository (`test/hosted/`), which runs a fake Orchard issuer and checks every
rule above over HTTP. `tests/hosted_test.cpp` pins the verifier against a fake
JWKS with hand-moved clocks; `tests/hosted_e2e_test.cpp` drives the built
example worker and every startup refusal.
