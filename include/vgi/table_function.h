// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <functional>
#include <map>

#include "vgi/cache_control.h"
#include "vgi/function.h"
#include "vgi/statistics.h"
#include "vgi/types.h"

namespace vgi {

// A table function's estimated output size, used by the planner.
//
// Both are optional because "I do not know" is a legitimate and common answer,
// and is different from an estimate of zero.
struct TableCardinality {
    std::optional<int64_t> estimate;
    std::optional<int64_t> max;
};

// One execution's scan state.
//
// `next_batch` is called repeatedly until it returns null. A producer is
// created per execution and owns whatever position the scan needs, which is
// why it is a separate object from the function: the function is shared and
// must stay immutable, the scan is not.
class TableProducer {
public:
    virtual ~TableProducer() = default;

    // The next batch, or null when the scan is exhausted.
    virtual std::shared_ptr<arrow::RecordBatch> next_batch() = 0;

    // Called before every tick, with a sink for messages the client should
    // see. The sink lives only until that tick's reply is written, so keep the
    // callback rather than the collector it closes over — the framework hands
    // over a fresh one each time for exactly that reason.
    //
    // A producer that has something to say — a warning about an argument, a
    // note about what it skipped — has nowhere else to put it: stdout is the
    // Arrow channel and stderr is swallowed by the engine.
    virtual void set_log(std::function<void(LogLevel, const std::string&)> log) { (void)log; }

    // Filters the engine discovered after the scan began, handed over before
    // each `next_batch` that follows.
    //
    // A join's build side is not known when the scan is planned, so its
    // predicate arrives per tick rather than at init. Ignoring them is always
    // safe — the engine re-checks every predicate — but a scan that can skip
    // work on them is the whole reason they are sent.
    virtual void on_dynamic_filters(const PushdownFilters& /*filters*/) {}

    // The validators a conditional request carried, handed over before the
    // first `next_batch`.
    //
    // A producer that recognises its own etag answers by emitting a zero-row
    // batch whose `last_metadata` says `not_modified`, and the engine serves
    // the payload it already holds. Ignoring them is always safe — the rows
    // are then recomputed, which is slower and not wrong.
    virtual void on_conditional_request(const std::optional<std::string>& /*if_none_match*/,
                                        const std::optional<std::string>& /*if_modified_since*/) {}

    // Wire metadata for the batch just returned. Called once after each
    // `next_batch` that produced one.
    //
    // Separate from the batch because Arrow's RecordBatch carries schema
    // metadata, not batch metadata, and the protocol needs the latter — a
    // cache advertisement describes this batch, not this shape.
    virtual std::map<std::string, std::string> last_metadata() const { return {}; }
};

// A function that generates rows without consuming any.
//
// Unlike a scalar function, its output schema is almost always decided at
// One named, independently redeemable unit of scan work.
//
// A split *names* work rather than describing it. "These three files at version
// 47" survives a retry; "rows 0-999 of whatever this returns now" does not — and
// a distributed engine will retry, so the difference is correctness, not
// tidiness. The same split may also be redeemed more than once, so redemption
// has to be replayable.
//
// Set `payload` and nothing else. The framework stamps the token envelope
// around it — the consistency anchor and the bind fingerprint — so a function
// author cannot forget either, and never writes the envelope by hand.
struct ScanSplit {
    std::string payload;
    std::optional<int64_t> estimated_rows;
    // True when `estimated_rows` is exact rather than an estimate, which is what
    // lets the engine answer COUNT(*) from the plan.
    bool rows_exact = false;
    std::optional<int64_t> estimated_bytes;
};

// What the engine asks for when it plans a scan.
struct PlanParams {
    // The primary sizing lever. A function should emit splits of comparable
    // cost: the engine cannot see per-split cost and claims them greedily as
    // interchangeable units, so one oversized split bounds the whole scan.
    std::optional<int64_t> target_split_bytes;
    // A parallelism floor. A small but expensive table still needs enough
    // splits to occupy the engine's readers.
    std::optional<int64_t> min_splits;
    // A pagination cap, not a sizing hint.
    std::optional<int64_t> max_splits_per_response;
    // Resume point in the *enumeration of splits*, not a position in the data.
    std::optional<std::string> cursor;
    std::optional<int64_t> row_limit;
    // False means more filter refinement may still arrive, so a function may
    // hold splits back; true says stop waiting.
    bool filters_complete = true;
    // The predicates known at plan time.
    PushdownFilters pushdown_filters;
};

// A plan: the splits, plus what the engine needs to schedule them.
struct PlanResult {
    std::vector<ScanSplit> splits;
    // How many workers the engine may run. Absent leaves it to the engine.
    std::optional<int64_t> max_workers;
    std::optional<int64_t> estimated_total_splits;
    std::optional<int64_t> estimated_total_rows;
    std::optional<int64_t> estimated_total_bytes;
    // The snapshot these splits were planned against. It is sealed into every
    // token, and a redemption whose anchor no longer matches is refused as
    // stale rather than served from a different version.
    std::optional<int64_t> catalog_version;
    // Where enumeration should resume, when this response is a page rather than
    // the whole plan.
    std::optional<std::string> next_cursor;
};

// bind — from its arguments, or from a resource it inspects — so `bind` is
// required rather than defaulted.
class TableFunction {
public:
    virtual ~TableFunction() = default;

    virtual std::string name() const = 0;
    virtual FunctionMetadata metadata() const = 0;
    virtual std::vector<ArgSpec> argument_specs() const = 0;

    virtual std::shared_ptr<arrow::Schema> bind(const BindParams& params) const = 0;

    virtual std::unique_ptr<TableProducer> init(const ProcessParams& params) const = 0;

    // The planner's estimate. The default declines to guess, which is what
    // most generators should do.
    virtual TableCardinality cardinality(const ProcessParams&) const { return {}; }

    // Whether this function divides its scan into splits.
    //
    // Off by default, and that default is what keeps every existing function
    // working: the framework answers a plan request for a non-split function
    // with one empty-payload split standing for the whole scan, so the engine's
    // planning path costs a round trip and changes nothing else.
    virtual bool supports_splits() const { return false; }

    // Divide this scan into named splits. Required when `supports_splits`.
    //
    // Return only the `payload` bytes on each split; the framework stamps the
    // envelope. Whatever state has to travel from planning to reading belongs
    // in the payload or in `FunctionStorage` — the process that plans is not
    // the process that reads, and on a cluster it is not even the same host.
    virtual PlanResult plan(const BindParams&, const PlanParams&) const {
        throw std::runtime_error("table function '" + name() +
                                 "' declares supports_splits but does not override plan()");
    }

    // Per-column bounds for the optimizer, or nothing when the function cannot
    // promise any.
    //
    // Answering is what lets DuckDB fold `WHERE n > 100000` over a scan bounded
    // at 9,999 into an empty result without ever calling the function — and,
    // by the same token, what makes a wrong bound a wrong answer rather than a
    // slow one.
    virtual std::optional<std::vector<ColumnStatistics>> statistics(const ProcessParams&) const {
        return std::nullopt;
    }

    // The secrets this call needs resolved, when which ones depends on the
    // arguments — a scope taken from a path argument, say.
    //
    // Answered from `metadata()` by default, which is enough for a fixed list.
    // A function whose scope is an argument cannot use that list: `metadata()`
    // is asked before any call site exists.
    virtual std::vector<SecretLookup> secret_lookups(const BindParams&) const {
        return metadata().required_secrets;
    }

    // Post-execution diagnostics, shown as Extra Info under EXPLAIN ANALYZE.
    //
    // Fired once per scan thread after the stream ends, so it cannot read the
    // producer — that object is long gone, and in the parallel case it lived in
    // another process. Whatever it reports the producer must have persisted to
    // `storage` under `global_execution_id`, which is the id the primary init
    // minted for the whole query.
    virtual std::vector<std::pair<std::string, std::string>> dynamic_to_string(
        const std::string& /*global_execution_id*/, FunctionStorage&) const {
        return {};
    }

    // How many workers the engine may run this scan across.
    //
    // 1 serializes it, which is the safe default: a producer that has not been
    // written to divide its work would otherwise emit the whole result once
    // per worker. Raise it only alongside `on_init`, which is where the work
    // gets divided.
    virtual int64_t max_workers(const ProcessParams&) const { return 1; }

    // Runs once per execution, before any producer is built.
    //
    // Only the *primary* init calls it — the one the engine sends without an
    // execution id — so it is the one place a parallel scan can divide its
    // work exactly once, typically by pushing chunks onto the shared queue
    // that every worker then pops from.
    virtual void on_init(const ProcessParams&) const {}
};

}  // namespace vgi
