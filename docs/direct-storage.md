# Direct object storage (S3 / GCS)

A self-hosted worker can keep its large values in your own bucket instead of
the blob vendor:

- externalized results;
- upload grants for large requests and argument references;
- StateRef and argument objects.

The worker signs URLs itself. Everything is configured by environment, with
the same variables as every other VGI SDK (direct-storage contract v1).

## Build

The backends are opt-in vcpkg features, because aws-sdk-cpp and
google-cloud-cpp are long builds:

```bash
cmake --preset release -DVCPKG_MANIFEST_FEATURES=s3 -DVGI_RPC_WITH_S3=ON
# GCS: -DVCPKG_MANIFEST_FEATURES="s3;gcs" -DVGI_RPC_WITH_GCS=ON as well
```

A worker built without a backend refuses to start when that backend is
selected, and the message names the feature it needs.

## Choosing a backend

`VGI_STORAGE_BACKEND` is `vendor`, `s3`, `gcs` or `none`. If it is unset:

1. `vendor` if `VGI_BLOB_VENDOR_URL` is set;
2. else `s3` if `VGI_STORAGE_S3_BUCKET` is set;
3. else `gcs` if `VGI_STORAGE_GCS_BUCKET` is set;
4. else `none`: everything stays inline.

Setting more than one of those three variables without `VGI_STORAGE_BACKEND`
refuses to start. Orchard hosted mode (`VGI_ORCHARD_HOSTED=1`) normally uses
the vendor. `s3` or `gcs` with hosted mode is allowed, and is logged at
startup.

## S3 and S3-compatible services

| Variable | Meaning | Default |
|---|---|---|
| `VGI_STORAGE_S3_BUCKET` | bucket (required) | |
| `VGI_STORAGE_S3_PREFIX` | key prefix, e.g. `vgi/`; objects go under `<prefix>rpc/` | empty |
| `VGI_STORAGE_S3_ENDPOINT` | custom endpoint (MinIO, Ceph, RustFS, Cloudflare R2, Wasabi, Backblaze B2, LocalStack) | AWS |
| `VGI_STORAGE_S3_REGION` | signing region (`auto` for R2) | `AWS_REGION`, then `AWS_DEFAULT_REGION`, then `us-east-1` |
| `VGI_STORAGE_S3_FORCE_PATH_STYLE` | `true` puts the bucket in the path | `true` with an endpoint, else `false` |
| `VGI_STORAGE_S3_PUBLIC_ENDPOINT` | endpoint written into the URLs handed to the engine, when the worker reaches storage at another address (see below) | the endpoint |
| `VGI_STORAGE_S3_CHECKSUM` | `sha256`, or `none` for services that do not enforce `x-amz-checksum-sha256` | `sha256` |
| `VGI_STORAGE_S3_ACCESS_KEY_ID`, `_SECRET_ACCESS_KEY`, `_SESSION_TOKEN` | explicit credentials | the AWS chain |
| `VGI_STORAGE_S3_SESSION_EXPIRES` | when an explicit session token expires (RFC 3339 or epoch seconds) | unknown |
| `VGI_STORAGE_S3_VERIFY` | `1`: check the bucket before serving (otherwise on first use) | off |
| `VGI_STORAGE_URL_TTL_SECONDS` | pre-signed URL lifetime, at most 7 days | 3600 |
| `VGI_STORAGE_ALLOW_PLAIN_HTTP` | `1`: allow an `http://` endpoint. The engine also needs `SET vgi_storage_allow_plain_http = true` | off |

### Credentials

Without explicit keys, the AWS SDK's default chain is used, in this order:

1. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`;
2. the shared profile (`AWS_PROFILE`, `~/.aws/credentials` and `~/.aws/config`), including `role_arn` / `source_profile` and `credential_process`;
3. SSO;
4. web identity (`AWS_WEB_IDENTITY_TOKEN_FILE` with `AWS_ROLE_ARN`);
5. container credentials (ECS, EKS Pod Identity);
6. EC2 instance metadata.

Temporary credentials are refreshed before they expire.

A pre-signed URL dies with the credentials that signed it. So its lifetime is
`min(VGI_STORAGE_URL_TTL_SECONDS, credential expiry - now - 60 s)`.

- For an explicit session token, set `VGI_STORAGE_S3_SESSION_EXPIRES`.
  Without it the worker can't cap URL lifetimes, and warns at startup.
  Prefer `credential_process` or the standard chain, which refresh
  themselves.
- Upload grants need 15 minutes. With a shorter session the worker refreshes
  once, then signs for what remains, with a warning.

Secrets never appear in logs or errors.

### Public endpoint

`VGI_STORAGE_S3_PUBLIC_ENDPOINT` changes only the host in the URLs given to
the engine. The worker still stores and fetches through
`VGI_STORAGE_S3_ENDPOINT`. Example: Docker-internal `http://minio:9000` for
the worker, `https://s3.example.com` for clients.

The worker must still be able to reach the public endpoint, because
request-upload pointers are fetched by URL. Argument references and StateRefs
are fetched by object id and are unaffected.

### `VGI_STORAGE_S3_CHECKSUM=none`

Upload v2 relies on storage refusing a body whose SHA-256 differs from the
grant. On a service that does not enforce `x-amz-checksum-sha256`, set
`none`. Then:

- the worker issues no upload grants: `VGI-Upload-URL-Support`,
  `VGI-Max-Upload-Bytes` and `VGI-Arguments-Ref-Bytes` are not advertised, so
  large requests and arguments stay inline;
- results are still externalized (the engine verifies their SHA-256);
- StateRef and argument objects are still stored (the worker verifies them on
  fetch).

## GCS

| Variable | Meaning |
|---|---|
| `VGI_STORAGE_GCS_BUCKET` | bucket (required) |
| `VGI_STORAGE_GCS_PREFIX` | key prefix; objects go under `<prefix>rpc/` |
| `VGI_STORAGE_GCS_ENDPOINT` | custom endpoint (an emulator such as fake-gcs-server) |
| `VGI_STORAGE_GCS_CREDENTIALS_FILE` | service-account key file; otherwise Application Default Credentials |
| `VGI_STORAGE_GCS_VERIFY` | `1`: check the bucket before serving |

URLs are V4-signed. Signing needs a key: either a service-account key, or
`ExternalStorageConfig::signing_account` with IAM signing.

## Lifecycle

The worker never deletes. Give the bucket a lifecycle rule that expires
`<prefix>rpc/` after one day, the same as the blob vendor.

## Timeouts

Connect 30 s. Stall 30 s (no bytes moving), or
`VGI_RPC_STORAGE_STALL_TIMEOUT_SECONDS`. There is no whole-request deadline.
