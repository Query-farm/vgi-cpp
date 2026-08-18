// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/record_batch.h>

#include "vgi/function.h"
#include "vgi/types.h"

namespace vgi {

// A `COPY … FROM '<path>' (FORMAT <name>)` reader.
//
// Driven over the table-function path, because a reader is a producer: it
// generates the rows the engine inserts and consumes none. What makes it a
// COPY format instead of a table function is that the catalog advertises it,
// so the engine registers the name the user types after FORMAT.
//
// Two names, and they are not the same thing: `format` is what a user types
// after FORMAT, and `handler_name` is the worker function the RPCs address.
class CopyFromFunction {
public:
    virtual ~CopyFromFunction() = default;

    virtual std::string format() const = 0;
    virtual std::string handler_name() const = 0;

    virtual std::optional<std::string> comment() const { return std::nullopt; }
    virtual FunctionMetadata metadata() const { return {}; }

    // The COPY options, declared as named arguments. Their types and docs
    // become the option metadata the engine surfaces.
    virtual std::vector<ArgSpec> argument_specs() const = 0;

    // Secrets this reader needs. Scoped to the COPY source, which is why the
    // bind params are the argument: a cloud read wants the credential that
    // matches the bucket it is reading from, not any credential of that type.
    virtual std::vector<SecretLookup> secret_lookups(const BindParams&) const { return {}; }

    // Parse the source at `params.copy_from_path` into batches whose schema is
    // exactly `params.output_schema` — the COPY target's columns. Exactly:
    // DuckDB inserts no cast between this scan and the INSERT, so a column of
    // the wrong type is an error rather than a conversion.
    virtual std::vector<std::shared_ptr<arrow::RecordBatch>> read(
        const ProcessParams& params) const = 0;
};

}  // namespace vgi
