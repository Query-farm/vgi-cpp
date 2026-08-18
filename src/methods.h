// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/type.h>

namespace vgi {

// How a protocol method answers.  The distinction is not cosmetic — vgi-rpc
// registers a void method differently from one with a result, and a stream
// differently again, so the table has to carry it.
enum class MethodKind {
    Void,    // -> None
    Result,  // -> a dataclass with a generated result schema
    Binary,  // -> bytes | None, carried as one "result" column
    Stream,  // init(): an exchange stream with a header
};

struct MethodSpec {
    std::string name;
    MethodKind kind;
    std::shared_ptr<arrow::Schema> params;
    // The schema of the *payload*, not of the response batch.
    //
    // Every non-void method answers with the same one-column envelope,
    // `{result: binary}`; what varies is what those bytes decode to. For a
    // Result method they are an IPC stream of a one-row batch in this schema.
    // For a Binary method they are the returned bytes verbatim, and this is
    // null. Registration uses `envelope_schema()`; handlers build against this.
    std::shared_ptr<arrow::Schema> payload;
};

// The response schema every non-void method registers: one binary column named
// "result", wrapping whatever the method actually returns.
const std::shared_ptr<arrow::Schema>& envelope_schema();

// The payload schema declared for `method`, or null if it has none.
const std::shared_ptr<arrow::Schema>& payload_schema_of(const std::string& method);

inline const std::shared_ptr<arrow::Schema> kNoSchema = nullptr;

// Every method of VgiProtocol, in declaration order, with the schemas the
// generators emitted for it.  Derived mechanically from
// `vgi-python`'s `VgiProtocol` and `vgi_protocol_schemas.hpp` rather than
// transcribed — see scripts/regenerate_methods.py.
const std::vector<MethodSpec>& protocol_methods();

}  // namespace vgi
