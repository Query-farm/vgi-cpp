// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
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
    // null. Registration uses `declared_envelope_schema()`; handlers build
    // against this.
    std::shared_ptr<arrow::Schema> payload;
    // Whether the reference's return annotation admits None (`bytes | None`).
    // Only such a method may answer a null `result`, and only its envelope
    // column is declared nullable.
    bool optional_result;
};

// The envelope every non-void handler builds its answer in: one binary column
// named "result", wrapping whatever the method actually returns. Nullable,
// because it has to hold the `bytes | None` answers too.
const std::shared_ptr<arrow::Schema>& envelope_schema();

// The same envelope as `spec` declares it on the wire, which is what a method
// is registered with. Nullability is part of an Arrow type, and so of the
// protocol description `vgi_rpc.Reflection.v1` reports and hashes: `result`
// is nullable only where the return is optional, as vgi-python derives it.
// dispatcher.cpp re-declares each handler's answer under this schema.
const std::shared_ptr<arrow::Schema>& declared_envelope_schema(const MethodSpec& spec);

// The payload schema declared for `method`, or null if it has none.
const std::shared_ptr<arrow::Schema>& payload_schema_of(const std::string& method);

inline const std::shared_ptr<arrow::Schema> kNoSchema = nullptr;

// Every method of VgiProtocol, in declaration order, with the schemas the
// generators emitted for it.  Derived mechanically from
// `vgi-python`'s `VgiProtocol` and `vgi_protocol_schemas.hpp` rather than
// transcribed — see scripts/regenerate_methods.py.
const std::vector<MethodSpec>& protocol_methods();

}  // namespace vgi
