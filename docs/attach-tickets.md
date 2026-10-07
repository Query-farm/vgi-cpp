# Attach tickets

An attach ticket lets a runner reattach a user's catalog later, as that user,
without ever seeing the user's attach options. The normative spec is
vgi-python's `docs/protocol/vgi-attach-tickets.md`; this page is how this SDK
implements it.

A ticket says *what* to attach. A sealed grant (`vgi_rpc.Identity.v1`
`issue_grant`, see [Hosted protocols and Identity](hosted-protocols.md)) says
*who* attaches:

1. While the user is attached and logged in, a client calls
   `vgi.attach_tickets.v1` `seal_attach` with the catalog name and the options
   the user attached with, secret ones included. The worker seals them into a
   `vgia1.` ticket that only it can open, and only for the same principal.
2. The client also calls `issue_grant` for a grant.
3. Later a runner presents `Authorization: Bearer <grant>` and attaches with
   one option:

   ```sql
   ATTACH 'anything' AS sales (TYPE vgi, LOCATION 'https://…',
       bearer_token '<grant>', vgi_attach_ticket '<ticket>');
   ```

   Before any catalog code runs, the framework opens the ticket under the
   caller's principal and replaces the request with the one the user made:
   the sealed catalog name, options and version specs. The request's own name
   is ignored, and the ticket is redeemed before the request is routed to a
   memory catalog, so the sealed catalog is the one that attaches.

A ticket carries no authority: another user's grant cannot open it, and a
leaked ticket alone attaches nothing.

## Enabling it

Nothing in the worker's code. `vgi.attach_tickets.v1` is hosted on `--http`
when **both**:

- `VGI_SIGNING_KEY` is set. It is the key tickets are sealed under (32 bytes
  are used as-is, anything else is hashed with SHA-256), and the key HTTP
  state tokens already use. A key generated for the process never enables
  tickets, since every ticket would die on restart.
- The worker can issue grants: grant keys are configured (`--grant-key` or
  `VGI_RPC_GRANT_KEYS`), or it sets `Worker::set_mint_grant`.

```bash
VGI_SIGNING_KEY=<stable secret> VGI_RPC_GRANT_KEYS=<base64 of 32 bytes> \
    my-worker --http 8080
```

Otherwise the protocol is absent, not hosted-and-refusing, so a client learns
the answer from `vgi_rpc.Reflection.v1` `list_protocols`. Off HTTP there is no
caller identity, and a ticket presented there is `attach_ticket_invalid`.

A ticket lives at most as long as a grant: the grant keys' maximum lifetime
(`VGI_RPC_GRANT_MAX_TTL_SECONDS`, 7 days by default). With a minter of the
worker's own and no grant keys, the ceiling is `VGI_RPC_GRANT_MAX_TTL_SECONDS`
when set, otherwise none. `ttl_seconds = 0` asks for the ceiling. Rotating
`VGI_SIGNING_KEY` invalidates every ticket.

## `seal_attach`

`seal_attach(request: SealAttachRequest) -> AttachTicket`, wire-identical to
vgi-python's (the protocol hash is the same):

| `SealAttachRequest` | Type |
|---|---|
| `catalog_name` | `utf8` |
| `options` | `binary`, Arrow IPC of the one-row options record, as `CatalogAttachRequest.options`; null for none |
| `data_version_spec` | `utf8`, `""` for none |
| `implementation_version` | `utf8`, `""` for none |
| `ttl_seconds` | `int64`; `0` = as long as allowed |

It returns `ticket` (`utf8`) and `expires_at` (`float64` Unix seconds, `+inf`
for no expiry). The caller must be authenticated (`action_denied` otherwise)
but needs no fresh login. The options are validated against the catalog's
declared attach options, case-insensitively: an undeclared option, a missing
required one, a nested `vgi_attach_ticket`, more than one row, an unknown
catalog, a negative TTL or options over 16 KiB are each a `BadRequest`
violation in one `invalid_request` error.

## Redemption errors

| `error_kind` | When |
|---|---|
| `invalid_request` | Another option rides beside `vgi_attach_ticket` (checked before the ticket is opened) |
| `attach_ticket_invalid` | Malformed, tampered, another worker's key, another principal, anonymous caller, no signing key |
| `attach_ticket_expired` | An authentic ticket outside its lifetime (60 s clock skew allowed) |

Neither the ticket nor a restored option appears in any error or log line.

## Declaring attach options

`vgi_attach_ticket` is reserved: a catalog declaring an attach option with that
name, in any letter case, refuses to start. Declare a credential option with
`AttachOptionSpec::secret = true`; it is advertised as such at discovery, and
the engine masks and redacts it.

## The `ticket_probe` fixture

The example worker serves the cross-SDK `ticket_probe` catalog: options
`region` (default `'us-east-1'`) and `api_key` (required, secret), and the
table `main.probe` reporting `region` and the first 12 hex characters of
SHA-256(`api_key`). With `VGI_FIXTURE_TEST_BEARERS=1` on `--http`, the bearers
`vgi-test-alice` and `vgi-test-bob` authenticate as fresh logins of `alice` and
`bob`, a `vgig1.` bearer falls through to grants, and any other bearer stays
anonymous. Unlike vgi-python's fixture, a request with no `Authorization`
header is then a 401: vgi-rpc-cpp's bearer authenticator is the server's only
door once configured, which is why the switch is opt-in.

`tests/attach_ticket_test.cpp` runs the shared vectors
(`tests/data/attach_ticket_vectors.json`, vendored verbatim from vgi-python);
`tests/attach_ticket_e2e_test.cpp` drives the whole flow over HTTP.
