# Roadmap

Ordered so that each milestone is testable against `~/Development/vgi`'s suite
rather than against assertions written here. The suite is the specification;
anything this file claims that the suite disagrees with is wrong.

## Done

Every function shape the protocol has, and the catalog surface around them.

1. **Scalar functions** — overloads resolved by argument type and by arity,
   per-schema and per-catalog registration, type bounds, declared ranges
   enforced at bind, stability, null handling, per-value result caching.
2. **Table functions** — producer streams, cardinality, per-column statistics,
   projection / filter / sampling pushdown, ORDER BY and TABLESAMPLE hints,
   dynamic filters delivered per tick, partitioning and batch indexes, row
   ids, late materialization, required filters, time travel.
3. **Table-in-out functions** — the exchange path with fan-out and the
   per-output-row provenance a batched correlated LATERAL needs, per-substream
   finalize, per-emission cache advertisements and conditional requests.
4. **Aggregates** — bind / update / combine / finalize / destructor, windowed
   and batched-window, streaming-partitioned, with per-group state in
   cross-process storage.
5. **Table buffering** — sink, combine, and a finalize producer, with input
   batch indexes and the order guarantees around them.
6. **Catalogs** — schemas, tables bound to scan functions, multi-branch tables
   stitched from several scans, views, macros, constraints, comments, tags,
   object counts, time travel, transactions, and several catalogs from one
   process.
7. **ATTACH** — npm-style data-version resolution, per-version schemas, typed
   ATTACH options with defaults, per-attachment identity.
8. **Result cache** — TTLs, nonces, scopes, revalidation, partition scopes,
   spill, and the poison and never-partial guarantees.
9. **COPY** — TO and FROM, sharing one format name between a writer and a
   reader.
10. **Settings and secrets** — declared at ATTACH, delivered per call, with
    two-phase resolution and secrets looked up by type and by scope.
11. **In-band logging** from every call, not only from a stream.
12. **Global functions** published into the engine's own namespace.

## Not done

- **HTTP transport.** The producer stream's state-token continuation path is
  exercised by `vgi-rpc-c++`'s own conformance suite but not by a VGI worker,
  and the `*_http` variants of the integration tests are skipped here.
- **Writable catalogs.** `catalog_table_insert_function_get` and its siblings
  refuse; `simple_writable`'s tests are skipped for want of the fixture
  worker.
- **The auth principal.** Nothing carries the resolved caller identity onto
  `BindParams`, so identity-scoped caching cannot be observed.
- **Iceberg and companion catalogs**, both gated behind environment the suite
  does not set here.

## Open questions

- **Namespace of generated code.** `duckdb::vgi::generated` is an artifact of
  the generators' original consumer. A `--namespace` flag upstream in
  `vgi-python` is the clean fix. Aliased for now.
- **Response builders.** `vgi_request_builders.hpp` is client-side. Whether
  worker-side response builders should be generated too is still undecided;
  the hand-written `ResultBuilder` has held up well enough that it may not be
  worth it.
