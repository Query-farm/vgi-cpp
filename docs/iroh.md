# Iroh

`vgi::Worker::run` accepts the portable bridge flags:

```console
worker --iroh-raw-upstream 127.0.0.1:9400 --iroh-issuer production
worker --http --host 127.0.0.1 --port 9401 --iroh-issuer production
```

Both listeners bind loopback. Raw mode requires the Iroh identity PROXY-v2
preamble; HTTP mode validates `VGI-Forwarded-Iroh-Endpoint`. Repeat
`--iroh-trusted-proxy <exact-IP>` to replace the loopback trust list and use
`--iroh-observe` only when authorization intentionally remains anonymous.
The older `--http 9401` shorthand remains supported.

Native C++ clients build with `VGI_RPC_WITH_IROH_CABI=ON`. `RpcClient` supports
stateful `iroh://`; `HttpClient` supports `httpi://` while retaining HTTP
budgets, continuation tokens, and externalized batches. No connector binary is
downloaded at runtime.
