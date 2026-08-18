# Roadmap

Ordered so that each milestone is testable against `~/Development/vgi`'s suite
rather than against assertions written here. The suite is the specification;
anything this file claims that the suite disagrees with is wrong.

Current: **125 of 309 test cases, 4,069 of 4,253 assertions.**

## Done

1. **Attach and scalar functions** — catalog discovery, `bind`, `init`, the
   scalar exchange path, overloads resolved by argument type, per-schema
   registration, type bounds, stability and null handling.
2. **Table functions** — producer streams, cardinality, parallel scans via
   `max_workers` + `on_init` + the shared work queue.
3. **Table-in-out functions** — the exchange path with fan-out, and
   name-based projection.
4. **Aggregates** — bind / update / combine / finalize / destructor, with the
   per-group state rule that keeps an all-NULL group NULL.
5. **Table buffering** — sink, combine, and a finalize producer, over
   cross-process storage.
6. **Catalogs** — schemas, tables bound to scan functions, views.
7. **Result cache** — `vgi.cache.*` advertisements on the first batch.
8. **Settings and secrets** — declared at ATTACH, delivered per call, with
   two-phase secret resolution.

## Next, roughly by how many tests each unblocks

- **The remaining fixtures.** Most failures are a missing example function
  rather than missing capability. `grep 'does not exist'` on the run log lists
  them.
- **Filter and projection pushdown** as declared capabilities, with the
  `pushdown_filters` request field parsed.
- **COPY from / to** — the `catalog_copy_from_formats` surface and the
  reader/writer pairing that shares one name.
- **Multi-branch scans** — `catalog_table_scan_branches_get`.
- **Time travel** — versioned table schemas selected by the AT clause.
- **Partitioning, batch indexes, late materialization.**
- **HTTP transport** — producer streams need the state-token continuation
  path, which the pipe transport does not exercise.

## Open questions

- **Namespace of generated code.** `duckdb::vgi::generated` is an artifact of
  the generators' original consumer. A `--namespace` flag upstream in
  `vgi-python` is the clean fix. Aliased for now.
- **Response builders.** `vgi_request_builders.hpp` is client-side. Whether
  worker-side response builders should be generated too is still undecided;
  the hand-written `ResultBuilder` has held up well enough that it may not be
  worth it.
