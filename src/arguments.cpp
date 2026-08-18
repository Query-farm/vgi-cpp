// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/arguments.h"

#include <charconv>

#include <arrow/compute/cast.h>

#include "wire.h"

namespace vgi {
namespace {

// "positional_3" -> 3; nullopt for anything else.
std::optional<size_t> positional_index(const std::string& name) {
    static constexpr std::string_view kPrefix = "positional_";
    if (name.size() <= kPrefix.size() || name.compare(0, kPrefix.size(), kPrefix) != 0) {
        return std::nullopt;
    }
    size_t value = 0;
    const char* first = name.data() + kPrefix.size();
    const char* last = name.data() + name.size();
    auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec != std::errc{} || ptr != last) return std::nullopt;
    return value;
}

// Cast a one-row array to `type` and hand back the converted array, or null if
// the value is absent or the cast is not possible.
std::shared_ptr<arrow::Array> cast_scalar(const std::shared_ptr<arrow::Array>& array,
                                          const std::shared_ptr<arrow::DataType>& type) {
    if (!array || array->length() < 1 || array->IsNull(0)) return nullptr;
    if (array->type()->Equals(*type)) return array;
    auto result = arrow::compute::Cast(*array, type);
    if (!result.ok()) return nullptr;
    return result.MoveValueUnsafe();
}

}  // namespace

Arguments Arguments::parse(const std::string& ipc_bytes) {
    Arguments args;
    if (ipc_bytes.empty()) return args;

    auto batch = wire::decode_ipc(ipc_bytes);
    if (!batch) return args;

    // DuckDB's form: one `args` struct column whose children are the
    // arguments. Anything else is treated as a direct column-per-argument
    // mapping, which is what a hand-built call site produces.
    if (batch->num_columns() == 1 && batch->schema()->field(0)->name() == "args") {
        auto structs = std::dynamic_pointer_cast<arrow::StructArray>(batch->column(0));
        if (!structs) return args;
        const auto& type = std::static_pointer_cast<arrow::StructType>(structs->type());

        std::vector<std::pair<size_t, std::shared_ptr<arrow::Array>>> positional;
        std::vector<std::pair<size_t, std::shared_ptr<arrow::Field>>> fields;
        for (int i = 0; i < type->num_fields(); ++i) {
            const auto& field = type->field(i);
            const auto& name = field->name();
            auto column = structs->field(i);
            if (auto index = positional_index(name)) {
                positional.emplace_back(*index, column);
                fields.emplace_back(*index, field);
                // Also reachable by its wire name, which is how a caller that
                // knows the layout addresses it.
                args.named_[name] = column;
            } else if (name.rfind("named_", 0) == 0) {
                args.named_[name.substr(std::string("named_").size())] = column;
                args.named_[name] = column;
            } else {
                args.named_[name] = column;
            }
        }

        if (!positional.empty()) {
            size_t highest = 0;
            for (const auto& [index, _] : positional) highest = std::max(highest, index);
            args.positional_.assign(highest + 1, nullptr);
            args.positional_fields_.assign(highest + 1, nullptr);
            for (auto& [index, array] : positional) args.positional_[index] = std::move(array);
            for (auto& [index, field] : fields) args.positional_fields_[index] = std::move(field);
        }
        return args;
    }

    for (int i = 0; i < batch->num_columns(); ++i) {
        args.named_[batch->schema()->field(i)->name()] = batch->column(i);
        args.positional_.push_back(batch->column(i));
        args.positional_fields_.push_back(batch->schema()->field(i));
    }
    return args;
}

std::shared_ptr<arrow::Array> Arguments::positional(size_t index) const {
    return index < positional_.size() ? positional_[index] : nullptr;
}

std::shared_ptr<arrow::Array> Arguments::named(const std::string& name) const {
    auto it = named_.find(name);
    return it == named_.end() ? nullptr : it->second;
}

bool Arguments::positional_is_null(size_t index) const {
    auto array = positional(index);
    return array && array->length() > 0 && array->IsNull(0);
}

bool Arguments::named_is_null(const std::string& name) const {
    auto array = named(name);
    return array && array->length() > 0 && array->IsNull(0);
}

std::shared_ptr<arrow::DataType> Arguments::positional_type(size_t index) const {
    if (index >= positional_fields_.size() || !positional_fields_[index]) return nullptr;
    return positional_fields_[index]->type();
}

std::shared_ptr<arrow::Field> Arguments::positional_field(size_t index) const {
    if (index >= positional_fields_.size()) return nullptr;
    return positional_fields_[index];
}

std::optional<int64_t> Arguments::const_int64(size_t index) const {
    auto array = cast_scalar(positional(index), arrow::int64());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::Int64Array>(array)->Value(0);
}

std::optional<double> Arguments::const_double(size_t index) const {
    auto array = cast_scalar(positional(index), arrow::float64());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::DoubleArray>(array)->Value(0);
}

std::optional<std::string> Arguments::const_string(size_t index) const {
    auto array = cast_scalar(positional(index), arrow::utf8());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::StringArray>(array)->GetString(0);
}

std::optional<bool> Arguments::const_bool(size_t index) const {
    auto array = cast_scalar(positional(index), arrow::boolean());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::BooleanArray>(array)->Value(0);
}

std::optional<int64_t> Arguments::named_int64(const std::string& name) const {
    auto array = cast_scalar(named(name), arrow::int64());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::Int64Array>(array)->Value(0);
}

std::optional<double> Arguments::named_double(const std::string& name) const {
    auto array = cast_scalar(named(name), arrow::float64());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::DoubleArray>(array)->Value(0);
}

std::optional<bool> Arguments::named_bool(const std::string& name) const {
    auto array = cast_scalar(named(name), arrow::boolean());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::BooleanArray>(array)->Value(0);
}

std::optional<std::string> Arguments::named_string(const std::string& name) const {
    auto array = cast_scalar(named(name), arrow::utf8());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::StringArray>(array)->GetString(0);
}

std::string serialize_scan_arguments(
    const std::vector<std::shared_ptr<arrow::Array>>& positional,
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& named) {
    if (positional.empty() && named.empty()) {
        return wire::encode_schema(arrow::schema({}));
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (size_t i = 0; i < positional.size(); ++i) {
        fields.push_back(arrow::field("arg_" + std::to_string(i), positional[i]->type(),
                                      /*nullable=*/false));
        columns.push_back(positional[i]);
    }
    for (const auto& [name, value] : named) {
        fields.push_back(arrow::field(name, value->type(), /*nullable=*/false));
        columns.push_back(value);
    }
    const int64_t rows = columns.empty() ? 0 : columns.front()->length();
    return wire::encode_ipc(arrow::RecordBatch::Make(arrow::schema(fields), rows, columns));
}

}  // namespace vgi
