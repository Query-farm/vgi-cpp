// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Reading VGI request parameters and building VGI responses.
//
// The shape of the wire, learned once so the 70 method handlers do not each
// re-learn it:
//
//   * A params batch is exactly one row, with one column per declared
//     parameter, in the generated params schema's order.
//   * A parameter whose type is a dataclass travels as `binary` holding a
//     complete, self-describing Arrow IPC stream of a one-row batch. That is
//     why no request-DTO schemas are generated: the bytes carry their own.
//   * A result batch is one row of the generated result schema. A response
//     that is a *list* of items (schemas, tables, functions…) carries them as
//     `list<binary>`, each element itself an IPC stream.
//
// Everything here throws on a mismatch rather than returning an optional. A
// param that is absent or the wrong type is a protocol violation, not a case
// a handler should have to branch on, and the exception carries the method and
// field so the failure names itself.

#pragma once

#include <cstdint>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "vgi/types.h"

namespace vgi::wire {

// ── Reading params ────────────────────────────────────────────────────────

// Column lookup that says which field was missing and what was there instead,
// because "field not found" alone has cost more debugging time than it saves.
std::shared_ptr<arrow::Array> column(const std::shared_ptr<arrow::RecordBatch>& batch,
                                     const std::string& field);

std::string get_string(const std::shared_ptr<arrow::RecordBatch>& batch,
                       const std::string& field);
std::optional<std::string> get_optional_string(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field);

std::string get_binary(const std::shared_ptr<arrow::RecordBatch>& batch,
                       const std::string& field);
std::optional<std::string> get_optional_binary(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field);

bool get_bool(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field);
std::optional<bool> get_optional_bool(const std::shared_ptr<arrow::RecordBatch>& batch,
                                      const std::string& field);
int64_t get_int64(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field);
std::optional<int64_t> get_optional_int64(const std::shared_ptr<arrow::RecordBatch>& batch,
                                          const std::string& field);
std::optional<double> get_optional_double(const std::shared_ptr<arrow::RecordBatch>& batch,
                                          const std::string& field);

// A dictionary<int16, utf8> parameter, read back as its string value.
std::string get_enum(const std::shared_ptr<arrow::RecordBatch>& batch,
                     const std::string& field);

// The same, for an optional enum: nullopt when the column is absent or null.
std::optional<std::string> get_optional_enum(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field);

// ── IPC-embedded values ───────────────────────────────────────────────────

// Decode a self-describing IPC stream into its single batch.  Returns null for
// a stream carrying a schema but no batches, which is how an empty optional
// dataclass is represented.
std::shared_ptr<arrow::RecordBatch> decode_ipc(const std::string& bytes);

// A `list<binary>` or `list<large_binary>` parameter, as its elements.
// A `list<int64>` parameter, as its elements.
std::vector<int64_t> get_int64_list(const std::shared_ptr<arrow::RecordBatch>& batch,
                                    const std::string& field);

std::vector<std::string> get_binary_list(const std::shared_ptr<arrow::RecordBatch>& batch,
                                        const std::string& field);

// A struct parameter's fields, rendered as strings. Nullopt when the column is
// absent or null.
std::optional<std::map<std::string, std::string>> get_struct_fields(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field);

// Read a dataclass parameter: the named binary column, decoded.
std::shared_ptr<arrow::RecordBatch> get_ipc(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field);

std::string encode_ipc(const std::shared_ptr<arrow::RecordBatch>& batch);

// An IPC stream carrying a schema and no batches.
//
// Distinct from encode_ipc on purpose: the `arguments` and `output_schema`
// fields of FunctionInfo are schemas, not batches. Their field *metadata* is
// the payload — `vgi_const`, `vgi_type`, `vgi_doc` and friends — so a batch
// with rows would be both wrong and lossy.
std::string encode_schema(const std::shared_ptr<arrow::Schema>& schema);

// The schema of an IPC stream, whether or not it carries batches.  Returns
// null for empty input, which is how an absent optional schema travels.
std::shared_ptr<arrow::Schema> decode_schema(const std::string& bytes);

// Read a schema-valued parameter: the named binary column, decoded.
std::shared_ptr<arrow::Schema> get_schema(const std::shared_ptr<arrow::RecordBatch>& batch,
                                          const std::string& field);

// ── Building results ──────────────────────────────────────────────────────

// A one-row result batch, built field by field against `schema` so a missing
// or misordered field is caught here rather than by the consumer.
class ResultBuilder {
public:
    explicit ResultBuilder(std::shared_ptr<arrow::Schema> schema);

    ResultBuilder& set_string(const std::string& field, const std::string& value);
    ResultBuilder& set_binary(const std::string& field, const std::string& value);
    ResultBuilder& set_bool(const std::string& field, bool value);
    ResultBuilder& set_int64(const std::string& field, int64_t value);
    ResultBuilder& set_null(const std::string& field);
    // Sets the value, or null when there is none. The pair appears often
    // enough that spelling it out at each site invites getting one wrong.
    ResultBuilder& set_optional_string(const std::string& field,
                                       const std::optional<std::string>& value);
    // A dictionary<int16, utf8> column.  The protocol spells its enums that
    // way — DuckDB reads them as dictionary-encoded strings — so a plain utf8
    // here is rejected by the consumer rather than coerced.
    ResultBuilder& set_enum(const std::string& field, const std::string& value);

    // Give every unset field the protocol's default for its type: null where
    // nullable, an empty list or map, false, 0, "".
    //
    // For a method *result* every field is deliberate and leaving one unset is
    // a bug, which is why finish() refuses. The Info dataclasses are the
    // opposite: thirty-odd fields of which a given function cares about five,
    // and the rest genuinely are "the default". Calling this makes that
    // explicit at the call site instead of hiding it in finish().
    ResultBuilder& fill_defaults();
    // A list<binary> column: each element is an encoded IPC stream.
    ResultBuilder& set_binary_list(const std::string& field,
                                   const std::vector<std::string>& values);
    // A map<utf8, utf8> column.  Arrow spells map entries key/value (not
    // keys/values), which is what the canonical Python protocol emits.
    ResultBuilder& set_string_list(const std::string& field,
                                   const std::vector<std::string>& values);
    // A `list<list<utf8>>` column.
    ResultBuilder& set_string_list_list(
        const std::string& field, const std::vector<std::vector<std::string>>& groups);
    // The `required_secrets` column: a list of
    // {secret_type, scope, secret_name}, where an absent scope or name is a
    // null rather than an empty string — the engine treats "" as a real scope.
    ResultBuilder& set_secret_lookups(
        const std::string& field,
        const std::vector<std::tuple<std::string, std::string, std::string>>& lookups);
    // The `examples` column: a list of {sql, description, expected_output}.
    ResultBuilder& set_examples(const std::string& field,
                                const std::vector<FunctionExample>& examples);
    ResultBuilder& set_string_map(
        const std::string& field,
        const std::vector<std::pair<std::string, std::string>>& entries);

    // Fails if any field was left unset — a silently-null required column
    // surfaces far from here, usually as a confusing engine-side error.
    std::shared_ptr<arrow::RecordBatch> finish();

private:
    int field_index(const std::string& field) const;

    std::shared_ptr<arrow::Schema> schema_;
    std::vector<std::shared_ptr<arrow::Array>> arrays_;
};

}  // namespace vgi::wire
