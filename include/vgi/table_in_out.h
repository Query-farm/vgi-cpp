// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
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
