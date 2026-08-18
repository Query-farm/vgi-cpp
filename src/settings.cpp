// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/settings.h"

#include <arrow/compute/cast.h>
#include <arrow/record_batch.h>

#include "wire.h"

namespace vgi {
namespace {

// Cast row 0 to `type`, or null if the value is absent or cannot convert.
std::shared_ptr<arrow::Array> as_type(const std::shared_ptr<arrow::Array>& array,
                                      const std::shared_ptr<arrow::DataType>& type) {
    if (!array || array->length() < 1 || array->IsNull(0)) return nullptr;
    if (array->type()->Equals(*type)) return array;
    auto result = arrow::compute::Cast(*array, type);
    if (!result.ok()) return nullptr;
    return result.MoveValueUnsafe();
}

// A secret field rendered the way a worker will use it. Strings pass through
// unquoted; everything else takes Arrow's own rendering, which is what the
// reference implementations do.
std::string render(const std::shared_ptr<arrow::Array>& array, int64_t row) {
    if (!array || row >= array->length() || array->IsNull(row)) return {};
    if (array->type()->id() == arrow::Type::STRING) {
        return static_cast<const arrow::StringArray&>(*array).GetString(row);
    }
    auto scalar = array->GetScalar(row);
    return scalar.ok() ? scalar.ValueUnsafe()->ToString() : std::string{};
}

}  // namespace

Settings Settings::parse(const std::string& ipc_bytes) {
    Settings settings;
    if (ipc_bytes.empty()) return settings;
    auto batch = wire::decode_ipc(ipc_bytes);
    if (!batch) return settings;
    for (int i = 0; i < batch->num_columns(); ++i) {
        settings.values_[batch->schema()->field(i)->name()] = batch->column(i);
    }
    return settings;
}

std::shared_ptr<arrow::Array> Settings::get(const std::string& name) const {
    auto it = values_.find(name);
    return it == values_.end() ? nullptr : it->second;
}

std::optional<int64_t> Settings::get_int64(const std::string& name) const {
    auto array = as_type(get(name), arrow::int64());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::Int64Array>(array)->Value(0);
}

std::optional<double> Settings::get_double(const std::string& name) const {
    auto array = as_type(get(name), arrow::float64());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::DoubleArray>(array)->Value(0);
}

std::optional<std::string> Settings::get_string(const std::string& name) const {
    auto array = as_type(get(name), arrow::utf8());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::StringArray>(array)->GetString(0);
}

std::optional<bool> Settings::get_bool(const std::string& name) const {
    auto array = as_type(get(name), arrow::boolean());
    if (!array) return std::nullopt;
    return std::static_pointer_cast<arrow::BooleanArray>(array)->Value(0);
}

Secrets Secrets::parse(const std::string& ipc_bytes) {
    Secrets secrets;
    if (ipc_bytes.empty()) return secrets;
    auto batch = wire::decode_ipc(ipc_bytes);
    if (!batch) return secrets;

    for (int i = 0; i < batch->num_columns(); ++i) {
        const auto& name = batch->schema()->field(i)->name();
        auto column = batch->column(i);
        std::map<std::string, std::string> fields;

        // One column per secret; a struct column carries the secret's fields,
        // and a flat column is a single-field secret named after itself.
        if (auto structs = std::dynamic_pointer_cast<arrow::StructArray>(column)) {
            const auto& type = std::static_pointer_cast<arrow::StructType>(structs->type());
            for (int j = 0; j < type->num_fields(); ++j) {
                fields[type->field(j)->name()] = render(structs->field(j), 0);
            }
        } else {
            fields[name] = render(column, 0);
        }
        secrets.by_name_[name] = std::move(fields);
    }
    return secrets;
}

std::optional<std::string> Secrets::field(const std::string& secret_name,
                                          const std::string& field_name) const {
    auto secret = by_name_.find(secret_name);
    if (secret == by_name_.end()) return std::nullopt;
    auto value = secret->second.find(field_name);
    if (value == secret->second.end()) return std::nullopt;
    return value->second;
}

const std::map<std::string, std::string>* Secrets::secret(
    const std::string& secret_name) const {
    auto it = by_name_.find(secret_name);
    return it == by_name_.end() ? nullptr : &it->second;
}

const std::map<std::string, std::string>* Secrets::of_type(const std::string& type) const {
    for (const auto& [name, fields] : by_name_) {
        (void)name;
        auto kind = fields.find("type");
        if (kind != fields.end() && kind->second == type) return &fields;
    }
    return nullptr;
}

std::optional<std::string> Secrets::typed_field(const std::string& type,
                                                const std::string& field_name) const {
    const auto* fields = of_type(type);
    if (!fields) return std::nullopt;
    auto value = fields->find(field_name);
    if (value == fields->end()) return std::nullopt;
    return value->second;
}

const std::map<std::string, std::string>* Secrets::for_scope_of_type(
    const std::string& path, const std::string& type) const {
    const std::map<std::string, std::string>* best = nullptr;
    size_t best_length = 0;
    for (const auto& [name, fields] : by_name_) {
        (void)name;
        auto kind = fields.find("type");
        if (kind == fields.end() || kind->second != type) continue;
        auto scope = fields.find("scope");
        // An unscoped secret of the right type is a candidate of length 0: it
        // matches anything, and loses to any scoped secret that also matches.
        const std::string prefix = scope == fields.end() ? "" : scope->second;
        // Plain prefix matching, which is DuckDB's own rule for secret scopes:
        // `s3://bucket` covers `s3://bucket/key` and, deliberately, also
        // `s3://bucket-two`. Longest prefix wins, and the first registration
        // wins a tie so the answer does not depend on map iteration order.
        if (!prefix.empty() && path.compare(0, prefix.size(), prefix) != 0) continue;
        if (!best || prefix.size() > best_length) {
            best = &fields;
            best_length = prefix.size();
        }
    }
    return best;
}

}  // namespace vgi
