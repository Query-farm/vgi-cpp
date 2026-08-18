// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "wire.h"

#include <sstream>
#include <stdexcept>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_base.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/builder_dict.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>
#include <arrow/status.h>

namespace vgi::wire {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

template <typename T>
T unwrap(arrow::Result<T> result, const std::string& context) {
    if (!result.ok()) fail(context + ": " + result.status().ToString());
    return std::move(result).ValueUnsafe();
}

void check_ok(const arrow::Status& status, const std::string& context) {
    if (!status.ok()) fail(context + ": " + status.ToString());
}

// Every accessor needs the same three checks, and getting one of them wrong is
// a segfault rather than an exception, so they live in one place.
template <typename ArrayType>
std::shared_ptr<ArrayType> typed_column(const std::shared_ptr<arrow::RecordBatch>& batch,
                                        const std::string& field,
                                        const char* expected) {
    auto arr = column(batch, field);
    auto typed = std::dynamic_pointer_cast<ArrayType>(arr);
    if (!typed) {
        fail("param '" + field + "' is " + arr->type()->ToString() + ", expected " + expected);
    }
    if (typed->length() < 1) fail("param '" + field + "' batch has no rows");
    return typed;
}

// An *optional* parameter may be absent as well as null.
//
// The protocol adds nullable columns additively — `schema_name` arrived in
// 1.1.0 and spread to fifteen more requests in 1.2.0 — so a peer built against
// an older surface simply does not send them. Treating absence as an error
// makes every bind from such a peer fail, which is precisely the compatibility
// the additive rule exists to provide. Both references tolerate it.
bool has_column(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    return batch && batch->GetColumnByName(field) != nullptr;
}

}  // namespace

std::shared_ptr<arrow::Array> column(const std::shared_ptr<arrow::RecordBatch>& batch,
                                     const std::string& field) {
    if (!batch) fail("param '" + field + "' requested from a null params batch");
    auto arr = batch->GetColumnByName(field);
    if (!arr) {
        std::ostringstream have;
        for (int i = 0; i < batch->num_columns(); ++i) {
            if (i) have << ", ";
            have << batch->schema()->field(i)->name();
        }
        fail("no param '" + field + "'; params carry [" + have.str() + "]");
    }
    return arr;
}

std::string get_string(const std::shared_ptr<arrow::RecordBatch>& batch,
                       const std::string& field) {
    auto arr = typed_column<arrow::StringArray>(batch, field, "string");
    if (arr->IsNull(0)) fail("param '" + field + "' is null but not optional");
    return arr->GetString(0);
}

std::optional<std::string> get_optional_string(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    if (!has_column(batch, field)) return std::nullopt;
    auto arr = typed_column<arrow::StringArray>(batch, field, "string");
    if (arr->IsNull(0)) return std::nullopt;
    return arr->GetString(0);
}

std::string get_binary(const std::shared_ptr<arrow::RecordBatch>& batch,
                       const std::string& field) {
    auto arr = typed_column<arrow::BinaryArray>(batch, field, "binary");
    if (arr->IsNull(0)) fail("param '" + field + "' is null but not optional");
    return arr->GetString(0);
}

std::optional<std::string> get_optional_binary(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    if (!has_column(batch, field)) return std::nullopt;
    // `binary` and `large_binary` alike: the protocol picks the wide form for
    // fields that can exceed 2 GiB — `pushdown_filters` is declared
    // large_binary — and they are the same value to every caller here.
    auto arr = column(batch, field);
    if (auto wide = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(arr)) {
        return wide->IsNull(0) ? std::nullopt : std::optional<std::string>(wide->GetString(0));
    }
    auto narrow = typed_column<arrow::BinaryArray>(batch, field, "binary or large_binary");
    if (narrow->IsNull(0)) return std::nullopt;
    return narrow->GetString(0);
}

bool get_bool(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    auto arr = typed_column<arrow::BooleanArray>(batch, field, "boolean");
    if (arr->IsNull(0)) fail("param '" + field + "' is null but not optional");
    return arr->Value(0);
}

std::optional<bool> get_optional_bool(const std::shared_ptr<arrow::RecordBatch>& batch,
                                      const std::string& field) {
    if (!has_column(batch, field)) return std::nullopt;
    auto arr = typed_column<arrow::BooleanArray>(batch, field, "boolean");
    if (arr->IsNull(0)) return std::nullopt;
    return arr->Value(0);
}

int64_t get_int64(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    auto arr = typed_column<arrow::Int64Array>(batch, field, "int64");
    if (arr->IsNull(0)) fail("param '" + field + "' is null but not optional");
    return arr->Value(0);
}

std::optional<int64_t> get_optional_int64(const std::shared_ptr<arrow::RecordBatch>& batch,
                                          const std::string& field) {
    if (!has_column(batch, field)) return std::nullopt;
    auto arr = typed_column<arrow::Int64Array>(batch, field, "int64");
    if (arr->IsNull(0)) return std::nullopt;
    return arr->Value(0);
}

std::string get_enum(const std::shared_ptr<arrow::RecordBatch>& batch,
                     const std::string& field) {
    auto arr = typed_column<arrow::DictionaryArray>(batch, field, "dictionary");
    if (arr->IsNull(0)) fail("param '" + field + "' is null but not optional");
    auto values = std::dynamic_pointer_cast<arrow::StringArray>(arr->dictionary());
    if (!values) fail("param '" + field + "' is a dictionary of non-strings");
    const auto* indices = dynamic_cast<const arrow::Int16Array*>(arr->indices().get());
    if (!indices) fail("param '" + field + "' has non-int16 dictionary indices");
    const auto index = indices->Value(0);
    // Range-checked: an out-of-range index from a malformed stream would
    // otherwise read past the dictionary.
    if (index < 0 || index >= values->length()) {
        fail("param '" + field + "' dictionary index " + std::to_string(index) +
             " is out of range");
    }
    return values->GetString(index);
}

std::optional<std::string> get_optional_enum(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    if (!has_column(batch, field)) return std::nullopt;
    auto arr = typed_column<arrow::DictionaryArray>(batch, field, "dictionary");
    if (arr->IsNull(0)) return std::nullopt;
    auto values = std::dynamic_pointer_cast<arrow::StringArray>(arr->dictionary());
    if (!values) return std::nullopt;
    const auto* indices = dynamic_cast<const arrow::Int16Array*>(arr->indices().get());
    if (!indices) return std::nullopt;
    return values->GetString(indices->Value(0));
}

std::shared_ptr<arrow::RecordBatch> decode_ipc(const std::string& bytes) {
    if (bytes.empty()) return nullptr;
    auto buffer = arrow::Buffer::FromString(bytes);
    auto source = std::make_shared<arrow::io::BufferReader>(buffer);
    auto reader = unwrap(arrow::ipc::RecordBatchStreamReader::Open(source),
                         "reading an embedded IPC value");
    std::shared_ptr<arrow::RecordBatch> batch;
    check_ok(reader->ReadNext(&batch), "reading an embedded IPC batch");
    // A schema with no batches is how an absent optional dataclass travels;
    // that is a legitimate value, not a truncated stream.
    return batch;
}

std::optional<std::map<std::string, std::string>> get_struct_fields(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    if (!has_column(batch, field)) return std::nullopt;
    auto structs = std::dynamic_pointer_cast<arrow::StructArray>(column(batch, field));
    if (!structs || structs->length() == 0 || structs->IsNull(0)) return std::nullopt;

    std::map<std::string, std::string> fields;
    const auto& type = std::static_pointer_cast<arrow::StructType>(structs->type());
    for (int i = 0; i < type->num_fields(); ++i) {
        auto child = structs->field(i);
        if (!child || child->IsNull(0)) continue;
        if (child->type()->id() == arrow::Type::STRING) {
            fields[type->field(i)->name()] =
                static_cast<const arrow::StringArray&>(*child).GetString(0);
            continue;
        }
        auto scalar = child->GetScalar(0);
        if (scalar.ok()) fields[type->field(i)->name()] = scalar.ValueUnsafe()->ToString();
    }
    return fields;
}

std::vector<std::string> get_binary_list(const std::shared_ptr<arrow::RecordBatch>& batch,
                                         const std::string& field) {
    std::vector<std::string> items;
    if (!has_column(batch, field)) return items;
    auto list = std::dynamic_pointer_cast<arrow::ListArray>(column(batch, field));
    if (!list || list->length() == 0 || list->IsNull(0)) return items;

    const int64_t begin = list->value_offset(0);
    const int64_t end = list->value_offset(1);
    // Both widths, for the same reason get_optional_binary accepts both.
    if (auto wide = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(list->values())) {
        for (int64_t i = begin; i < end; ++i) {
            if (!wide->IsNull(i)) items.push_back(wide->GetString(i));
        }
        return items;
    }
    if (auto narrow = std::dynamic_pointer_cast<arrow::BinaryArray>(list->values())) {
        for (int64_t i = begin; i < end; ++i) {
            if (!narrow->IsNull(i)) items.push_back(narrow->GetString(i));
        }
    }
    return items;
}

std::shared_ptr<arrow::RecordBatch> get_ipc(
    const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    auto bytes = get_optional_binary(batch, field);
    if (!bytes) return nullptr;
    return decode_ipc(*bytes);
}

std::string encode_ipc(const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto sink = unwrap(arrow::io::BufferOutputStream::Create(), "allocating an IPC sink");
    auto writer = unwrap(arrow::ipc::MakeStreamWriter(sink, batch->schema()),
                         "opening an IPC writer");
    check_ok(writer->WriteRecordBatch(*batch), "writing an embedded IPC batch");
    check_ok(writer->Close(), "closing an embedded IPC stream");
    auto buffer = unwrap(sink->Finish(), "finishing an IPC sink");
    return buffer->ToString();
}

std::shared_ptr<arrow::Schema> decode_schema(const std::string& bytes) {
    if (bytes.empty()) return nullptr;
    auto buffer = arrow::Buffer::FromString(bytes);
    auto source = std::make_shared<arrow::io::BufferReader>(buffer);
    auto reader = unwrap(arrow::ipc::RecordBatchStreamReader::Open(source),
                         "reading an embedded IPC schema");
    return reader->schema();
}

std::shared_ptr<arrow::Schema> get_schema(const std::shared_ptr<arrow::RecordBatch>& batch,
                                          const std::string& field) {
    auto bytes = get_optional_binary(batch, field);
    if (!bytes) return nullptr;
    return decode_schema(*bytes);
}

std::string encode_schema(const std::shared_ptr<arrow::Schema>& schema) {
    auto sink = unwrap(arrow::io::BufferOutputStream::Create(), "allocating a schema sink");
    auto writer = unwrap(arrow::ipc::MakeStreamWriter(sink, schema), "opening a schema writer");
    check_ok(writer->Close(), "closing a schema-only IPC stream");
    auto buffer = unwrap(sink->Finish(), "finishing a schema sink");
    return buffer->ToString();
}

ResultBuilder::ResultBuilder(std::shared_ptr<arrow::Schema> schema)
    : schema_(std::move(schema)), arrays_(static_cast<size_t>(schema_->num_fields())) {}

int ResultBuilder::field_index(const std::string& field) const {
    int index = schema_->GetFieldIndex(field);
    if (index < 0) fail("result schema has no field '" + field + "'");
    return index;
}

ResultBuilder& ResultBuilder::set_string(const std::string& field, const std::string& value) {
    arrow::StringBuilder b;
    check_ok(b.Append(value), "building result field '" + field + "'");
    arrays_[static_cast<size_t>(field_index(field))] =
        unwrap(b.Finish(), "finishing result field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_binary(const std::string& field, const std::string& value) {
    arrow::BinaryBuilder b;
    check_ok(b.Append(value), "building result field '" + field + "'");
    arrays_[static_cast<size_t>(field_index(field))] =
        unwrap(b.Finish(), "finishing result field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_bool(const std::string& field, bool value) {
    arrow::BooleanBuilder b;
    check_ok(b.Append(value), "building result field '" + field + "'");
    arrays_[static_cast<size_t>(field_index(field))] =
        unwrap(b.Finish(), "finishing result field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_int64(const std::string& field, int64_t value) {
    arrow::Int64Builder b;
    check_ok(b.Append(value), "building result field '" + field + "'");
    arrays_[static_cast<size_t>(field_index(field))] =
        unwrap(b.Finish(), "finishing result field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_null(const std::string& field) {
    const int index = field_index(field);
    const auto& type = schema_->field(index)->type();
    std::unique_ptr<arrow::ArrayBuilder> builder;
    check_ok(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder),
             "building null result field '" + field + "'");
    check_ok(builder->AppendNull(), "appending null to result field '" + field + "'");
    arrays_[static_cast<size_t>(index)] =
        unwrap(builder->Finish(), "finishing result field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_binary_list(const std::string& field,
                                              const std::vector<std::string>& values) {
    auto values_builder = std::make_shared<arrow::BinaryBuilder>();
    arrow::ListBuilder b(arrow::default_memory_pool(), values_builder);
    check_ok(b.Append(), "opening list result field '" + field + "'");
    for (const auto& v : values) {
        check_ok(values_builder->Append(v), "appending to list result field '" + field + "'");
    }
    arrays_[static_cast<size_t>(field_index(field))] =
        unwrap(b.Finish(), "finishing list result field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_string_list(const std::string& field,
                                              const std::vector<std::string>& values) {
    auto values_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder b(arrow::default_memory_pool(), values_builder);
    check_ok(b.Append(), "opening list result field '" + field + "'");
    for (const auto& v : values) {
        check_ok(values_builder->Append(v), "appending to list result field '" + field + "'");
    }
    arrays_[static_cast<size_t>(field_index(field))] =
        unwrap(b.Finish(), "finishing list result field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_examples(const std::string& field,
                                           const std::vector<FunctionExample>& examples) {
    const int index = field_index(field);
    // Build against the schema's own struct type rather than a locally
    // declared one: field order and nullability are the generator's to decide.
    const auto& list_type = schema_->field(index)->type();
    std::unique_ptr<arrow::ArrayBuilder> raw;
    check_ok(arrow::MakeBuilder(arrow::default_memory_pool(), list_type, &raw),
             "building examples field '" + field + "'");
    auto* list = dynamic_cast<arrow::ListBuilder*>(raw.get());
    if (!list) fail("result field '" + field + "' is not a list");
    auto* entry = dynamic_cast<arrow::StructBuilder*>(list->value_builder());
    if (!entry) fail("result field '" + field + "' is not a list of structs");

    check_ok(list->Append(), "opening examples field '" + field + "'");
    for (const auto& example : examples) {
        check_ok(entry->Append(), "opening an example struct");
        auto* sql = dynamic_cast<arrow::StringBuilder*>(entry->field_builder(0));
        auto* description = dynamic_cast<arrow::StringBuilder*>(entry->field_builder(1));
        auto* expected = dynamic_cast<arrow::StringBuilder*>(entry->field_builder(2));
        if (!sql || !description || !expected) fail("unexpected example struct layout");
        check_ok(sql->Append(example.sql), "appending example sql");
        check_ok(description->Append(example.description), "appending example description");
        check_ok(example.expected_output ? expected->Append(*example.expected_output)
                                         : expected->AppendNull(),
                 "appending example expected_output");
    }
    arrays_[static_cast<size_t>(index)] =
        unwrap(list->Finish(), "finishing examples field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_string_map(
    const std::string& field,
    const std::vector<std::pair<std::string, std::string>>& entries) {
    auto key_builder = std::make_shared<arrow::StringBuilder>();
    auto item_builder = std::make_shared<arrow::StringBuilder>();
    arrow::MapBuilder b(arrow::default_memory_pool(), key_builder, item_builder);
    check_ok(b.Append(), "opening map result field '" + field + "'");
    for (const auto& [key, value] : entries) {
        check_ok(key_builder->Append(key), "appending map key to '" + field + "'");
        check_ok(item_builder->Append(value), "appending map value to '" + field + "'");
    }
    arrays_[static_cast<size_t>(field_index(field))] =
        unwrap(b.Finish(), "finishing map result field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::set_enum(const std::string& field, const std::string& value) {
    const int index = field_index(field);
    const auto& type = schema_->field(index)->type();
    const auto* dict_type = dynamic_cast<const arrow::DictionaryType*>(type.get());
    if (!dict_type) {
        fail("result field '" + field + "' is " + type->ToString() + ", not a dictionary");
    }
    arrow::StringBuilder values;
    check_ok(values.Append(value), "building enum field '" + field + "'");
    auto dictionary = unwrap(values.Finish(), "finishing enum field '" + field + "'");

    // One value, so index 0 always; the dictionary carries exactly the string
    // this row uses rather than a shared vocabulary.
    arrow::Int16Builder indices;
    check_ok(indices.Append(0), "building enum index for '" + field + "'");
    auto index_array = unwrap(indices.Finish(), "finishing enum index for '" + field + "'");

    arrays_[static_cast<size_t>(index)] = unwrap(
        arrow::DictionaryArray::FromArrays(type, index_array, dictionary),
        "assembling enum field '" + field + "'");
    return *this;
}

ResultBuilder& ResultBuilder::fill_defaults() {
    for (int i = 0; i < schema_->num_fields(); ++i) {
        if (arrays_[static_cast<size_t>(i)]) continue;
        const auto& field = schema_->field(i);
        const auto& type = field->type();
        if (field->nullable()) {
            set_null(field->name());
            continue;
        }
        // A non-nullable field still needs a value, and the empty one is what
        // the canonical dataclasses default to.
        switch (type->id()) {
            case arrow::Type::STRING:  set_string(field->name(), ""); break;
            case arrow::Type::BINARY:  set_binary(field->name(), ""); break;
            case arrow::Type::BOOL:    set_bool(field->name(), false); break;
            case arrow::Type::INT64:   set_int64(field->name(), 0); break;
            case arrow::Type::MAP:     set_string_map(field->name(), {}); break;
            default: {
                std::unique_ptr<arrow::ArrayBuilder> builder;
                check_ok(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder),
                         "building default for '" + field->name() + "'");
                // Lists and structs default to one empty element, not to null:
                // the field is non-nullable, so null is not available.
                if (auto* list = dynamic_cast<arrow::ListBuilder*>(builder.get())) {
                    check_ok(list->Append(), "opening default list '" + field->name() + "'");
                } else {
                    check_ok(builder->AppendEmptyValue(),
                             "appending default for '" + field->name() + "'");
                }
                arrays_[static_cast<size_t>(i)] =
                    unwrap(builder->Finish(), "finishing default for '" + field->name() + "'");
            }
        }
    }
    return *this;
}

std::shared_ptr<arrow::RecordBatch> ResultBuilder::finish() {
    for (size_t i = 0; i < arrays_.size(); ++i) {
        if (!arrays_[i]) {
            // Defaulting to null here would produce a batch the engine accepts
            // and then misreads; naming the field is far cheaper.
            fail("result field '" + schema_->field(static_cast<int>(i))->name() +
                 "' was never set");
        }
    }
    return arrow::RecordBatch::Make(schema_, 1, arrays_);
}

}  // namespace vgi::wire
