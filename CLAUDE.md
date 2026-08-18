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

The Arrow schemas, constants, protocol version, and request builders in
`src/generated/` come from generators in `vgi-python`. **Never hand-edit
them.** Regenerate:

```bash
cd ~/Development/vgi-python
uv run python -m vgi.codegen.cpp_schemas          > ~/Development/vgi-c++/src/generated/vgi_protocol_schemas.hpp
uv run python -m vgi.codegen.cpp_constants        > ~/Development/vgi-c++/src/generated/vgi_protocol_constants.hpp
uv run python -m vgi.codegen.cpp_protocol_version > ~/Development/vgi-c++/src/generated/vgi_protocol_version.hpp
uv run python -m vgi.codegen.cpp_request_builders > ~/Development/vgi-c++/src/generated/vgi_request_builders.hpp
```

There is no `vgi-gen-cpp-*` console script — those entry points are not
registered in `vgi-python`'s `pyproject.toml`, so run the modules with
`python -m` as above.

Two things to know about the generated headers:

- **They are namespaced `duckdb::vgi::generated`**, because their original
  consumer was the DuckDB extension. That namespace is meaningless here. The
  SDK aliases it rather than post-processing generated output. Adding a
  `--namespace` flag to `vgi/codegen/cpp_schemas.py` upstream would be the
  clean fix; it is deliberately not done yet, to avoid a fifth repo in flight.
- **`vgi_request_builders.hpp` builds *client* requests.** A worker parses
  requests and builds *responses*, so it is reference material for field
  layout, not something to call. The schemas, by contrast, are
  direction-agnostic and used directly.

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
cmake --preset default    # Debug + tests
cmake --preset release
cmake --build build
```

`vgi-rpc-c++` is built from the sibling checkout by default (they move
together, and an installed copy goes stale silently). Override with
`-DVGI_RPC_SOURCE_DIR=<path>`, or `-DVGI_USE_INSTALLED_RPC=ON` to
`find_package` an installed one.

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

**125 of 309 integration test cases, 4,069 of 4,253 assertions.** Run
`./scripts/run_tests.sh` for the current figure — it is the only number that
means anything here.

Implemented in the SDK: scalar / table / table-in-out / aggregate / buffering
functions, catalogs with tables and views, per-schema registration and
overload resolution, cross-process storage, result-cache advertisements,
settings, secrets, and parallel scans.

Not yet: filter and projection pushdown as a *declared* capability, COPY
from/to, time travel, multi-branch scans, partitioning, batch indexes, late
materialization, and the HTTP transport's stream-continuation path. Most of
the remaining failures are unwritten fixtures rather than missing SDK
capability — `grep 'does not exist' /tmp/vgi-cpp-test-cache/run.log` lists them
by name.

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
