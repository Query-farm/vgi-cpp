// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/record_batch.h>

#include "vgi/function.h"
#include "vgi/types.h"

namespace vgi {

// A `COPY … TO '<path>' (FORMAT <name>)` writer.
//
// Driven over the buffering RPC path, because the shape is the same: the
// engine sinks batches and then closes once. The difference is that there is
// no source phase — nothing is read back — so `combine` returns nothing to
// finalize and no producer is ever built.
//
// Two names, and they are not the same thing: `format` is what a user types
// after FORMAT, and `handler_name` is the worker function the RPCs address.
class CopyToFunction {
public:
    virtual ~CopyToFunction() = default;

    virtual std::string format() const = 0;
    virtual std::string handler_name() const = 0;

    virtual std::optional<std::string> comment() const { return std::nullopt; }
    virtual FunctionMetadata metadata() const { return {}; }

    // The COPY options, declared as named arguments. Their types and docs
    // become the option metadata the engine surfaces.
    virtual std::vector<ArgSpec> argument_specs() const = 0;

    // Secrets this writer needs. Scoped to the COPY destination, which is why
    // the bind params are the argument: a cloud write wants the credential
    // that matches the bucket it is writing to, not any credential of that
    // type.
    virtual std::vector<SecretLookup> secret_lookups(const BindParams&) const { return {}; }

    // Whether rows must arrive in source order. Declaring it makes the engine
    // install a single-threaded sink, so one worker sees every batch in order;
    // leaving it false lets the sink shard across workers.
    virtual bool ordered() const { return false; }

    // Persist one batch. Called per sink batch and possibly in parallel across
    // worker processes, so state belongs in `params.storage` scoped by
    // `execution_id` — never on `this`.
    virtual void write(const ProcessParams& params,
                       const std::shared_ptr<arrow::RecordBatch>& batch) = 0;

    // Write and close the destination, once. Called even when nothing was
    // written, so an empty COPY still produces an empty or header-only file.
    // The returned count is informational; DuckDB reports its own.
    virtual int64_t close(const ProcessParams& params) = 0;
};

}  // namespace vgi
