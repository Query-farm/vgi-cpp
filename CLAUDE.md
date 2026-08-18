# CLAUDE.md — vgi-c++

## What this is

The **C++ worker SDK for VGI** (Vector Gateway Interface): a library for
writing DuckDB workers in C++, built on the C++ `vgi-rpc` port. It is the C++
sibling of `~/Development/vgi-rust` and must be wire-compatible with the
canonical Python SDK.

A *worker* is an ordinary binary that DuckDB launches and talks to over Arrow
IPC. It exposes scalar / table / table-in-out / aggregate / buffering
functions and whole catalogs that behave like native DuckDB objects. The
engine side is the C++ DuckDB extension at `~/Development/vgi` — this repo
does **not** build a DuckDB extension.

## The four repos this sits between

| path | what it is | how it matters here |
|---|---|---|
| `~/Development/vgi-python` | the **canonical** implementation, and the protocol's source of truth | owns the code generators; when the protocol changes it changes here first |
| `~/Development/vgi-rust` | the Rust worker SDK | the functional spec to mirror — read it before designing anything |
| `~/Development/vgi` | the DuckDB C++ extension (engine side) | owns the integration tests this SDK has to pass |
| `~/Development/vgi-rpc-c++` | the RPC framework | the transport/dispatch layer underneath; built as a sibling checkout |

## Definition of done

**Passing `~/Development/vgi`'s integration test suite** — 309 `.test` files —
with an example worker built from this SDK, exactly as `vgi-rust` does it via
`scripts/run_tests.sh`. The extension selects the worker through the
`VGI_TEST_WORKER` environment variable:

```bash
VGI_TEST_WORKER=/path/to/vgi-example-worker \
  ~/Development/vgi/build/release/test/unittest "test/sql/integration/*"
```

`~/Development/vgi-rust/scripts/run_tests.sh` and `ci/run-integration.sh` are
the templates for the harness — read them rather than inventing one. Note that
one binary is routed into several catalogs by wrapper scripts and the
`VGI_WORKER_CATALOG_NAME` environment variable.

## Protocol code is generated, not written

The Arrow schemas, constants and protocol version in `src/generated/` come
from generators in `vgi-python`. **Never hand-edit them.** Regenerate with:

```bash
scripts/regenerate_protocol.sh
```

That wraps the three `python -m vgi.codegen.cpp_*` generators, passes
`--namespace vgi::generated`, and prints the resulting `VGI_PROTOCOL_VERSION`.

Two things to know:

- **The generators default to `duckdb::vgi::generated`**, because their first
  consumer was the DuckDB extension. VGI is a wire protocol, not a DuckDB
  feature, and a worker built on this SDK links no DuckDB — so this repo
  passes `--namespace vgi::generated`. Leaving the default alone is what keeps
  the extension's own copies byte-identical.
- **Regenerating adopts whatever protocol version vgi-python is at.** If that
  is ahead of the engine you test against, the version gate refuses every
  request and the whole suite fails at once. The script prints the version for
  exactly this reason; check it against the engine before trusting a red run.

`vgi_request_builders.hpp` used to be vendored here. It builds *client*
requests — a worker parses requests and builds responses — and it
`#include`s a DuckDB header this repo does not have, so it could never have
compiled. It was dropped. Read it in `~/Development/vgi` if you need it as
reference for field layout.

## The protocol surface

69 RPC methods, enumerated by
`grep -oE 'params_schema_for\("[a-z_]+"\)' ~/Development/vgi-rust/vgi-protocol/src/generated/request_params.rs | sort -u`.
They group as:

- **catalog discovery / DDL** (~45): `catalog_attach`, `catalog_schemas`,
  `catalog_schema_contents_{functions,tables,views,macros,indexes}`,
  `catalog_table_*`, `catalog_view_*`, `catalog_transaction_*`, …
- **function lifecycle**: `bind`, `init`
- **aggregates** (12): `aggregate_{bind,update,combine,finalize,window,…}`
- **table buffering**: `table_buffering_{process,combine,destructor}`
- **table function planning**: `table_function_{cardinality,statistics,dynamic_to_string}`

Per-function call methods (`scalar_function`, `table_function`, …) are
registered dynamically rather than appearing in that list.

## Build

```bash
export VCPKG_ROOT=~/Development/vgi-rpc-c++/vcpkg   # or any vcpkg checkout
cmake --preset default    # Debug + tests
cmake --preset release
cmake --build build
```

`VCPKG_ROOT` rather than a `vcpkg/` symlink in the tree: the symlink that used
to be committed pointed at an absolute path in one developer's home directory,
which is broken for every other clone.

`vgi-rpc-c++` is built from the sibling checkout by default (they move
together, and an installed copy goes stale silently). Override with
`-DVGI_RPC_SOURCE_DIR=<path>`, or `-DVGI_USE_INSTALLED_RPC=ON` to
`find_package` an installed one.

`-Wall -Wextra` are on for our own targets. They are not decoration:
`-Wreturn-stack-address` caught a reference bound into the temporary
`shared_ptr` that `ArrayBuilder::type()` returns, in code the whole
integration suite ran past. Do not silence one without understanding it.

## Formatting

```bash
scripts/format.sh          # rewrite in place
scripts/format.sh --check  # name what differs, exit non-zero
```

clang-format, Google style with 4-space indent and a 100-column limit — the
two deviations this codebase already had by hand. The script pins the major
version, because clang-format's output changes between releases and two
contributors on different versions reformat each other's files on every
commit. `src/generated/` is excluded; formatting it only guarantees the next
regeneration produces a diff.

## Testing philosophy

Inherited from `vgi-rpc-c++`, and it applies with more force here: **the
integration suite in `~/Development/vgi` is the definitive test source.**
Keep C++ unit tests minimal and pointed at internal utilities — type mapping,
argument coercion, wire encoding. Do not write C++ tests that re-assert
protocol behaviour; they will rot as the protocol moves, and the canonical
Python implementation is always authoritative.

Run the release build against the suite first (debug is slow), then isolate
failures in debug. Always `tee` the output — otherwise you end up running the
suite twice to read it.

## Status

**The whole in-scope suite passes**: 273 test cases, 10,763 assertions, with
36 skipped for want of environment the harness does not set (HTTP transport,
Iceberg, the companion and writable fixture workers). Run
`./scripts/run_tests.sh` for the current figure — it is the only number that means anything here.
`docs/roadmap.md` lists what is and is not implemented.

## Things worth knowing before changing anything

Each of these cost real debugging time, and none is guessable from the code:

- **Every non-void method answers `{result: binary}`.** The generated "result
  schema" describes what is *inside* those bytes, not the response batch.
- **`arguments` and `output_schema` are IPC-serialized *schemas*, not
  batches.** A parameter list is fields plus metadata; `vgi_const` above all,
  because a const parameter must not appear in the per-row batch.
- **Two enums spell "scalar".** `FunctionInfo.function_type` is `scalar`; the
  filter the engine sends is `SCALAR_FUNCTION`. And case is load-bearing —
  a lowercase `null_handling` is silently ignored, a lowercase
  `order_dependence` rejects the catalog.
- **The engine runs several worker processes per query.** A buffering sink
  fans out across them and finalizes in another, so anything remembered in a
  member variable is gone. Use `ProcessParams::storage`.
- **Execution ids must be unique across processes**, since that store is
  shared by every worker of the uid.
- **Buffering functions are advertised under the `table` filter.** The engine
  never asks for `table_buffering`.
- **A cardinality of -1 is an answer, not an absence.** The engine reads the
  field as optional, takes the -1, skips `table_function_cardinality`, and
  clamps the estimate to one row. Send null.
- **`attach_opaque_data` is the only thing most catalog calls carry that says
  which attachment they belong to.** This SDK seals the catalog, the resolved
  data version, a per-ATTACH id and the merged ATTACH options into it. One
  binary serves several catalogs, and two of them may declare the same
  function in the same schema.
- **An exchange tick must answer with exactly one data batch**, even a
  zero-row one. A tick that emits nothing leaves the caller waiting on a reply
  that never comes, and the query hangs rather than failing.
- **Conditional-request validators ride the first *tick*, not the init.** A
  producer that overrides `produce` never sees them.
- **The engine scans the two macro kinds in separate calls.** Answering both
  with every macro registers each of them twice.
- **A worker pool hands out a different process per RPC**, so aggregate group
  state, buffering state and per-attachment state all belong in
  `FunctionStorage`, not in a member.
- **`RecordBatch::Make` validates nothing, and neither does the IPC writer.**
  A batch whose arrays disagree with its schema is written cleanly and read
  back as *different numbers* — a double column declared int64 decodes to the
  bit patterns. `wire::encode_ipc` calls `Validate()` on the way out so this
  fails loudly; do not build a batch some other way.
- **`ArrayBuilder::type()` and `RecordBatch::column()` return by value.**
  Binding a reference to `*builder.type()` or to `batch->column(i)` leaves it
  pointing into something freed at the end of that statement. This has
  segfaulted twice; `-Wreturn-stack-address` is on for exactly that reason.
- **A single-branch table honours the AT clause through
  `catalog_table_scan_branches_get`.** The engine does not fall back to
  `catalog_table_scan_function_get` once the branches method answers, so
  resolving the version in only one of the two silently serves the newest
  version for every AT.
- **The engine keys its COPY registry on the alias-scoped format name and
  skips a name it has already registered.** A format declared once as a writer
  and once as a reader must go out as a single entry with
  `direction: "both"`, or whichever came second is dropped with no error.
- **`supports_time_travel` is answered once, at ATTACH, and DuckDB's binder
  refuses an AT clause outright when it is false** — before any per-table
  logic runs. Every way a table can say it travels has to be counted there.
- **A statistic with no bound must send null, not a default.** `[0, 0]` and
  "no nulls" are promises the optimizer acts on: it will fold `WHERE s > 10`
  to zero rows on the strength of them.
