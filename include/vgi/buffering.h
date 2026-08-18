// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/function.h"
#include "vgi/table_function.h"
#include "vgi/types.h"

namespace vgi {

// A function that consumes a whole relation before producing anything.
//
// Two phases, and the engine drives them through different calls:
//
//   1. **Sink** — `process` is called with each input batch and returns an
//      opaque state id naming what it stored. `combine` then reduces those
//      ids to the set that will be drained.
//   2. **Source** — `init` opens with the TABLE_BUFFERING_FINALIZE phase and
//      `finalize_producer` builds a producer per surviving state id.
//
// The split exists because the engine parallelizes the sink: several workers
// may call `process` for the same query, and `combine` is where their partial
// states are reconciled. A function that only ever sees one worker still has
// to answer both.
class TableBufferingFunction {
public:
    virtual ~TableBufferingFunction() = default;

    virtual std::string name() const = 0;
    virtual FunctionMetadata metadata() const = 0;
    virtual std::vector<ArgSpec> argument_specs() const = 0;

    virtual std::shared_ptr<arrow::Schema> bind(const BindParams& params) const = 0;

    // Sink one batch; the returned id is opaque to the engine, which only
    // hands it back to `combine`.
    virtual std::string process(const ProcessParams& params,
                                const std::shared_ptr<arrow::RecordBatch>& batch) = 0;

    // Reduce the sunk state ids to the ones worth draining.
    virtual std::vector<std::string> combine(const ProcessParams& params,
                                             const std::vector<std::string>& state_ids) = 0;

    // The producer that drains one finalize state id.
    virtual std::unique_ptr<TableProducer> finalize_producer(
        const ProcessParams& params, const std::string& finalize_state_id) = 0;
};

}  // namespace vgi
