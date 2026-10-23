// Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once
#include <arrow/api.h>
#include <arrow/extension_type.h>
#include <arrow/util/utf8.h>

#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace vgi {
inline void validate_column_type(const std::shared_ptr<arrow::DataType>& type);
inline void validate_column_field(const std::shared_ptr<arrow::Field>& field) {
    if (const auto& metadata = field->metadata()) {
        std::set<std::string> metadata_keys;
        for (int64_t i = 0; i < metadata->size(); ++i) {
            const auto& key = metadata->key(i);
            const auto& value = metadata->value(i);
            if (!metadata_keys.insert(key).second)
                throw std::invalid_argument("duplicate column metadata key");
            if (key != "comment" && key != "tags") continue;
            if (!arrow::util::ValidateUTF8(value))
                throw std::invalid_argument("column metadata must be UTF-8");
            if (key == "tags") {
                std::set<std::string> keys;
                const auto tags = nlohmann::json::parse(
                    value, [&](int, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
                        if (event == nlohmann::json::parse_event_t::key &&
                            !keys.insert(parsed.get<std::string>()).second)
                            throw std::invalid_argument("duplicate column tag key");
                        return true;
                    });
                if (!tags.is_object())
                    throw std::invalid_argument("column tags must be a JSON object of strings");
                for (const auto& tag : tags.items())
                    if (!tag.value().is_string())
                        throw std::invalid_argument("column tag values must be strings");
            }
        }
    }
    validate_column_type(field->type());
}
inline void validate_column_type(const std::shared_ptr<arrow::DataType>& type) {
    if (type->id() == arrow::Type::DICTIONARY)
        validate_column_type(std::static_pointer_cast<arrow::DictionaryType>(type)->value_type());
    else if (type->id() == arrow::Type::EXTENSION)
        validate_column_type(std::static_pointer_cast<arrow::ExtensionType>(type)->storage_type());
    else
        for (const auto& field : type->fields()) validate_column_field(field);
}
inline void validate_column_metadata(const std::shared_ptr<arrow::Schema>& schema) {
    arrow::util::InitializeUTF8();
    for (const auto& field : schema->fields()) validate_column_field(field);
}
}  // namespace vgi
