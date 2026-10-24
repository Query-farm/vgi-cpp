// Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once
#include <arrow/api.h>
#include <arrow/extension_type.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/util/base64.h>
#include <arrow/util/utf8.h>

#include <algorithm>
#include <stdexcept>

#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "wire.h"

namespace vgi {
inline constexpr const char* GENERATED_COLUMN_KEY = "vgi.column.generated.v1";
inline constexpr const char* GENERATION_COLUMN_KEY = "vgi.column.generation.v1";
inline void generated_text(const std::string& value, size_t maximum) {
    arrow::util::InitializeUTF8();
    if (value.empty() || value.size() > maximum || value.find('\0') != std::string::npos ||
        !arrow::util::ValidateUTF8(value))
        throw std::invalid_argument("invalid generated expression text");
}
inline void validate_generated_expression(const std::string& encoded, const SchemaPath& owner) {
    constexpr size_t maximum = 8 * 1024 * 1024;
    if (encoded.empty() || encoded.size() > 4 * ((maximum + 2) / 3))
        throw std::invalid_argument("invalid expression size");
    const auto bytes = wire::base64_decode(encoded);
    if (bytes.empty() || bytes.size() > maximum || arrow::util::base64_encode(bytes) != encoded)
        throw std::invalid_argument("expression requires canonical base64 IPC");
    auto source = std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(bytes));
    auto opened = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!opened.ok()) throw std::invalid_argument(opened.status().ToString());
    auto reader = opened.ValueOrDie();
    std::shared_ptr<arrow::RecordBatch> row, extra;
    auto status = reader->ReadNext(&row);
    if (!status.ok() || !row || row->num_rows() != 1 ||
        !row->schema()->Equals(*generated::SqlExpressionSchema(), false))
        throw std::invalid_argument("invalid SqlExpression record schema");
    if (!row->ValidateFull().ok()) throw std::invalid_argument("invalid expression buffers");
    for (const auto& column : row->columns())
        if (column->IsNull(0)) throw std::invalid_argument("required expression field is NULL");
    const auto sql = wire::get_string(row, "sql");
    generated_text(sql, 1024 * 1024);
    if (sql.find_first_not_of(" \t\r\n") == std::string::npos ||
        wire::get_string(row, "profile") != "vgi.expression.duckdb.scalar.v1")
        throw std::invalid_argument("unsupported generated profile or empty SQL");
    const auto path = wire::get_string_list(row, "schema_path");
    if (path.size() > 32 || path != owner)
        throw std::invalid_argument("generated expression owner differs from table schema");
    for (const auto& part : path) generated_text(part, 4096);
    const auto context =
        std::static_pointer_cast<arrow::StructArray>(wire::column(row, "evaluation_context"));
    for (const auto& value : context->fields())
        if (value->IsNull(0)) throw std::invalid_argument("required expression context is NULL");
    const auto string_at = [&](int i) {
        return std::static_pointer_cast<arrow::StringArray>(context->field(i))->GetString(0);
    };
    generated_text(string_at(0), 256);
    generated_text(string_at(1), 256);
    const auto order = string_at(2), null_order = string_at(3);
    if ((order != "asc" && order != "desc") ||
        (null_order != "nulls_first" && null_order != "nulls_last" &&
         null_order != "nulls_first_on_asc_last_on_desc" &&
         null_order != "nulls_last_on_asc_first_on_desc"))
        throw std::invalid_argument("invalid expression evaluation context");
    if (!reader->ReadNext(&extra).ok() || extra ||
        source->Tell().ValueOrDie() != static_cast<int64_t>(bytes.size()))
        throw std::invalid_argument("expression requires exactly one complete IPC batch");
}
inline void validate_column_expressions(const std::shared_ptr<arrow::Schema>& schema,
                                        const SchemaPath& owner, bool allow_generated = true) {
    std::function<void(const std::shared_ptr<arrow::DataType>&)> type;
    const std::function<void(const std::shared_ptr<arrow::Field>&)> field = [&](const auto& f) {
        if (const auto& metadata = f->metadata()) {
            for (const auto& key : metadata->keys()) {
                if ((key.rfind("vgi.column.generated.", 0) == 0 && key != GENERATED_COLUMN_KEY) ||
                    (key.rfind("vgi.column.generation.", 0) == 0 && key != GENERATION_COLUMN_KEY))
                    throw std::invalid_argument("unknown generated metadata version");
            }
            if (metadata->Contains("generated_expression"))
                throw std::invalid_argument(
                    "legacy generated expression requires explicit canonical translation");
            const auto generated = metadata->Contains(GENERATED_COLUMN_KEY),
                       generation = metadata->Contains(GENERATION_COLUMN_KEY);
            if (generated != generation)
                throw std::invalid_argument(
                    "generated expression and category must occur together");
            if (generated) {
                if (metadata->Get(GENERATION_COLUMN_KEY).ValueOrDie() != "virtual")
                    throw std::invalid_argument("only VIRTUAL generated columns are supported");
                if (metadata->Contains("default") || metadata->Contains("vgi.column.default.v1"))
                    throw std::invalid_argument("generated column cannot have a default");
                validate_generated_expression(metadata->Get(GENERATED_COLUMN_KEY).ValueOrDie(),
                                              owner);
                if (!allow_generated)
                    throw std::invalid_argument("generated columns unsupported by this catalog");
            }
        }
        type(f->type());
    };
    type = [&](const auto& t) {
        if (t->id() == arrow::Type::DICTIONARY)
            type(std::static_pointer_cast<arrow::DictionaryType>(t)->value_type());
        else if (t->id() == arrow::Type::EXTENSION)
            type(std::static_pointer_cast<arrow::ExtensionType>(t)->storage_type());
        else
            for (const auto& child : t->fields()) field(child);
    };
    for (const auto& f : schema->fields()) field(f);
}
}  // namespace vgi
