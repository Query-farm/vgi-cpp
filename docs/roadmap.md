# Roadmap

Ordered so that each milestone is testable against `~/Development/vgi`'s suite
rather than against assertions written here. The suite is the specification;
anything this file claims that the suite disagrees with is wrong.

## 1. Attach and one scalar function

The narrowest end-to-end slice, and the one that proves the wire format:

- `catalog_attach`, `catalog_schemas`, `catalog_schema_contents_functions`
- `bind`, `init`
- the per-function scalar call path

Target: `ATTACH` succeeds and `SELECT upper_case('x')` returns from a C++
worker. Then the narrowest scalar tests under
`~/Development/vgi/test/sql/integration/`.

## 2. The rest of the scalar surface

Overloads, named and constant arguments, `any`-typed arguments, volatility,
NULL handling, error propagation. `~/Development/vgi-rust/vgi-example-worker/src/scalar/`
enumerates what the fixtures expect.

## 3. Table functions

`table_function` producer streams, `table_function_cardinality`,
`table_function_statistics`, projection and filter pushdown.

## 4. Catalogs

Schemas, tables, views, macros, indexes — the ~45 `catalog_*` methods. Read
only first; DDL and transactions after.

## 5. Aggregates and buffering

The 12 `aggregate_*` methods, then `table_buffering_*`. Windowed aggregates
last — they are the largest single piece.

## 6. Transports beyond stdio

`--unix` for the pooled launcher (the default regression path in `~/Development/vgi`'s
Makefile is `test_launcher`, so this is needed earlier than it looks), then
`--http`.

## Open questions

- **Namespace of generated code.** `duckdb::vgi::generated` is an artifact of
  the generators' original consumer. Adding a `--namespace` flag upstream in
  `vgi-python` is the clean fix. Aliased for now.
- **Response builders.** `vgi_request_builders.hpp` is client-side. Whether
  worker-side response builders should be generated too — mirroring what
  `vgi-rust` does by hand in `vgi-protocol/src/protocol/dtos.rs` — is
  undecided, and worth settling before hand-writing 69 methods' worth of
  encoding.
