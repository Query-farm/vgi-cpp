// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/type.h>

namespace vgi {

// How a protocol method answers.
enum class MethodKind {
    Void,    // -> None
    Result,  // -> a dataclass with a generated result schema
    Binary,  // -> bytes | None, carried as one "result" column
    Stream,  // init(): an exchange stream with a header
};

// One vgi.v2 method as handler code sees it: a view of a row of the generated
// registration table (`generated::VGI_METHODS`, src/generated/vgi_service.hpp),
// which is what the server registers.
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
    // null.
    std::shared_ptr<arrow::Schema> payload;
    // Whether the reference's return annotation admits None (`bytes | None`).
    // Only such a method may answer a null `result`, and only its envelope
    // column is declared nullable.
    bool optional_result;
};

// The envelope every non-void handler builds its answer in: one binary column
// named "result", wrapping whatever the method actually returns. Nullable,
// because it has to hold the `bytes | None` answers too. Dispatcher::install
// re-declares each answer under the `result` its method is registered with.
const std::shared_ptr<arrow::Schema>& envelope_schema();

// The payload schema declared for `method`, or null if it has none.
const std::shared_ptr<arrow::Schema>& payload_schema_of(const std::string& method);

// Every method of VgiProtocol, by wire name, from the generated table.
const std::vector<MethodSpec>& protocol_methods();

}  // namespace vgi
