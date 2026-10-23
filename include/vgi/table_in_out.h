// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/cache_control.h"
#include "vgi/function.h"
#include "vgi/types.h"

namespace vgi {

// One batch a table-in-out emits, and what it says about itself.
//
// Convertible from a bare batch on purpose: a function that has nothing to add
// still writes `return {batch};`, and the annotations exist only for the shapes
// that need them.
struct EmittedBatch {
    EmittedBatch() = default;
    EmittedBatch(std::shared_ptr<arrow::RecordBatch> batch)  // NOLINT(google-explicit-constructor)
        : batch(std::move(batch)) {}

    std::shared_ptr<arrow::RecordBatch> batch;

    // Which input row produced each output row, one entry per emitted row.
    //
    // Absent means the identity map, which is what the engine assumes — and
    // under that assumption it requires exactly as many output rows as input
    // rows. A function that fans one row out to several, or drops one, has to
    // say which is which or the engine cannot stamp the correlated columns.
    std::optional<std::vector<int32_t>> parent_rows;

    // A cache advertisement for this batch alone, where `cache_control()`
    // speaks for every batch the function emits.
    std::optional<CacheControl> cache_control;

    // Anything else this batch carries, merged last so a fixture can send a
    // deliberately malformed value the typed fields above could not express.
    std::map<std::string, std::string> metadata;

    // Whether the input batch has more output after this reply
    // (`vgi.status: HAVE_MORE_OUTPUT`). The engine then asks for the next
    // reply with a continuation tick, which the framework answers through
    // `ResumableTableInOutFunction::resume`. A reply says more when any batch
    // in it does.
    //
    // A field rather than raw `metadata`, because the framework has to see it:
    // the next input must be a tick exactly when the last reply said more.
    // Refused from a function that cannot resume, and from `finish`.
    bool more = false;
};

// One stream's resumable side: the position a function that continues its
// output keeps between replies. `ResumableTableInOutFunction` implements it;
// a function author does not.
class TableInOutContinuation {
public:
    virtual ~TableInOutContinuation() = default;

    // A real input batch: its first reply.
    virtual std::vector<EmittedBatch> process(const ProcessParams& params,
                                              const std::shared_ptr<arrow::RecordBatch>& batch) = 0;

    // A continuation tick: the next reply for the last real input batch.
    virtual std::vector<EmittedBatch> resume(const ProcessParams& params) = 0;
};

// A function that takes a table and produces a table.
//
// Driven as an exchange stream, but unlike a scalar function it is not
// row-for-row: one input batch may produce zero, one, or many output batches.
// That is the whole difference in the contract, and it is why `process`
// returns a vector rather than a batch.
class TableInOutFunction {
public:
    virtual ~TableInOutFunction() = default;

    virtual std::string name() const = 0;
    virtual FunctionMetadata metadata() const = 0;
    virtual std::vector<ArgSpec> argument_specs() const = 0;

    // The output schema. Defaults to passing the input schema through, which
    // is what a filter or a row-wise transform wants.
    virtual std::shared_ptr<arrow::Schema> bind(const BindParams& params) const;

    virtual std::vector<EmittedBatch> process(
        const ProcessParams& params, const std::shared_ptr<arrow::RecordBatch>& batch) const = 0;

    // Whether the function accumulates across the whole stream and flushes at
    // the end. Declaring it drives the engine's FINALIZE phase; without it,
    // `finish` is never called.
    //
    // The finalize runs per *substream*, and may land on a worker that never
    // saw a tick — so whatever `finish` reports has to have been written to
    // `ProcessParams::storage` under the substream id, not held in memory.
    virtual bool has_finish() const { return false; }

    virtual std::vector<EmittedBatch> finish(const ProcessParams&) const { return {}; }

    // The secrets this call needs resolved, when which ones depends on the
    // arguments — a scope taken from a path argument, say.
    //
    // Answered from `metadata()` by default, which is enough for a fixed list.
    // A function whose scope is an argument cannot use that list: `metadata()`
    // is asked before any call site exists.
    virtual std::vector<SecretLookup> secret_lookups(const BindParams&) const {
        return metadata().required_secrets;
    }

    // A result-cache advertisement for this function's output. See
    // `ScalarFunction::cache_control` for when advertising is sound.
    virtual std::optional<CacheControl> cache_control() const { return std::nullopt; }

    // The per-stream state of a function that continues its output across
    // replies, opened once per exchange. Null, the default, for a function
    // that answers each input batch with one reply. Derive from
    // `ResumableTableInOutFunction` rather than overriding this.
    virtual std::unique_ptr<TableInOutContinuation> open_continuation(const ProcessParams&) const {
        return nullptr;
    }
};

// A table-in-out function that may answer one input batch with several
// replies — one upstream API page per reply, say, rather than every page
// built into one batch.
//
// `process` answers a real input batch and `resume` each continuation tick
// after it; each returns one reply, with `more` set on every reply but the
// input's last. Every reply's `parent_rows` index the batch `process` saw.
//
// `resume` gets no input, so `State` must hold everything it needs to go on:
// which rows remain, the next page token. Keep it a position, never buffered
// output. It lives for the exchange, one per stream, so on HTTP it rides the
// stream's server-side state between turns like any other exchange state.
//
// The validators of a conditional request (`if_none_match`,
// `if_modified_since`) belong to the input that carried them: `process` sees
// them, `resume` never does, and only the input's first reply may carry cache
// control.
template <typename State>
class ResumableTableInOutFunction : public TableInOutFunction {
    // A stateless resume could only start over, so refuse it where it is
    // declared rather than on the first query that continues.
    static_assert(!std::is_empty_v<State>,
                  "resume() continues from its State alone, so State must hold the resume "
                  "position");
    static_assert(std::is_default_constructible_v<State> && std::is_move_constructible_v<State>,
                  "State must be default- and move-constructible");

public:
    // The state a new stream starts from.
    virtual State initial_state(const ProcessParams&) const { return State{}; }

    // The first reply for a real input batch.
    virtual std::vector<EmittedBatch> process(const ProcessParams& params,
                                              const std::shared_ptr<arrow::RecordBatch>& batch,
                                              State& state) const = 0;

    // The next reply for the last input batch, asked by a continuation tick.
    virtual std::vector<EmittedBatch> resume(const ProcessParams& params, State& state) const = 0;

    // Never called: the framework drives the stateful overload above.
    std::vector<EmittedBatch> process(const ProcessParams&,
                                      const std::shared_ptr<arrow::RecordBatch>&) const final {
        throw std::logic_error("table-in-out '" + name() +
                               "' is resumable; call it through its continuation");
    }

    std::unique_ptr<TableInOutContinuation> open_continuation(
        const ProcessParams& params) const final {
        return std::make_unique<Continuation>(*this, initial_state(params));
    }

private:
    class Continuation : public TableInOutContinuation {
    public:
        // The function outlives the exchange that holds this, which also holds
        // the function.
        Continuation(const ResumableTableInOutFunction& fn, State state)
            : fn_(fn), state_(std::move(state)) {}

        std::vector<EmittedBatch> process(
            const ProcessParams& params,
            const std::shared_ptr<arrow::RecordBatch>& batch) override {
            return fn_.process(params, batch, state_);
        }

        std::vector<EmittedBatch> resume(const ProcessParams& params) override {
            return fn_.resume(params, state_);
        }

    private:
        const ResumableTableInOutFunction& fn_;
        State state_;
    };
};

// Project `batch` onto `schema` by column *name*, filling absent columns with
// nulls.
//
// By name and not by position: the engine narrows a table-in-out's output
// schema when a projection is pushed down, and the surviving columns keep
// their names but not their indices.
std::shared_ptr<arrow::RecordBatch> project_batch(const std::shared_ptr<arrow::RecordBatch>& batch,
                                                  const std::shared_ptr<arrow::Schema>& schema);

}  // namespace vgi
