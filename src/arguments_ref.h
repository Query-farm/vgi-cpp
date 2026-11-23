// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// Argument references (protocol 3.4, vgi docs/large-values-3.4.md §2, §5, §5a).
//
// A client binding a function with a large argument over HTTP uploads the
// arguments once, through this worker's `__upload_url__` grant, and sends a
// `BindRequest` whose `arguments` is null and whose `arguments_location`
// carries the grant's sealed `ref` plus the upload's size and SHA-256. Every
// request that embeds that bind call (cardinality, statistics, plan,
// projection, dynamic-to-string, init) carries the small location, never the
// value.
//
// The framework resolves the location once per turn, at the turn's entry
// (`Dispatcher::read_bind_request`, which every bind-call reader goes
// through), and hands user code resolved `Arguments`. The bind-call bytes are
// never rewritten: what is persisted or compared is always what the client
// sent (the reference form). vgi-rpc does the opening, the object cache (one
// `get` per object id per process, single-flight) and the size / SHA-256
// checks.
//
// A worker advertises support with the `VGI-Arguments-Ref-Bytes` capability
// header (`install_header`); without it the extension never externalizes
// arguments.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <arrow/record_batch.h>
#include <vgi_rpc/call_context.h>
#include <vgi_rpc/http_config.h>

namespace vgi::arguments_ref {

// The capability header, and the operator override of its value.
inline constexpr const char* kHeader = "VGI-Arguments-Ref-Bytes";
inline constexpr const char* kBytesEnv = "VGI_ARGUMENTS_REF_BYTES";
inline constexpr int64_t kDefaultBytes = 4 * 1024 * 1024;
// Test switch: the HTTP server stops sending the header (the extension's
// `arguments_ref_fallback.test`), behaving like a worker without argument
// references while still offering upload URLs.
inline constexpr const char* kDisableEnv = "VGI_TEST_DISABLE_ARGUMENTS_REF";

inline constexpr const char* kInvalidBindRequestKind = "invalid_bind_request";
inline constexpr const char* kInvalidBindRequestMessage =
    "BindRequest must set exactly one of 'arguments' and 'arguments_location'";
inline constexpr const char* kInvalidArgumentsRefKind = "invalid_arguments_ref";
inline constexpr const char* kMissingKind = "arguments_ref_missing";
// The exact message for an argument object storage reports gone (spec §5a,
// "Prepared statements"); copied from vgi-python `vgi/arguments_ref.py`.
inline constexpr const char* kMissingMessage =
    "this prepared statement's large argument has expired; prepare it again";

// `VGI_ARGUMENTS_REF_BYTES` when set, else 4 MiB. Throws std::invalid_argument
// for anything but a positive decimal integer of at most 19 digits without a
// sign or leading zero (the header's grammar), so a typo stops startup.
int64_t bytes_from_env();

// Advertise argument-reference support when the configured server can resolve
// what its grant mints: upload URLs plus an object store. Called after every
// other configuration step. Returns the advertised value, or nullopt.
std::optional<int64_t> install_header(vgi_rpc::HttpConfig& config);

// A bind call's `arguments_location`, validated (size >= 1, sha256 64
// lowercase hex; otherwise `invalid_arguments_ref`).
struct Location {
    std::string ref;
    int64_t size = 0;
    std::string sha256;
};

// Exactly one of `arguments` / `arguments_location` is set, else
// `invalid_bind_request`. Returns the location when that is the one set.
std::optional<Location> check(const std::shared_ptr<arrow::RecordBatch>& bind_call);

// The bind call's arguments IPC bytes: inline, or resolved through `context`
// (spec §5a steps 2-5). Errors: `invalid_bind_request`, `invalid_arguments_ref`,
// `arguments_ref_integrity`, `arguments_ref_missing` (the prepared-statement
// message), and vgi-rpc's `storage_unavailable`.
std::string resolve(const std::shared_ptr<arrow::RecordBatch>& bind_call,
                    vgi_rpc::CallContext* context);

// The arguments' part of a split-token fingerprint (spec §5a): `0x01 ||
// u64le(size) || sha256[32]` for a location (never the sealed ref, which holds
// a random nonce), `0x00 || u64le(len) || bytes` inline.
std::string fingerprint_input(const std::shared_ptr<arrow::RecordBatch>& bind_call);

}  // namespace vgi::arguments_ref
