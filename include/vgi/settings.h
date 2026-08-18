// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include <arrow/array.h>

namespace vgi {

// DuckDB settings the engine forwarded to this call.
//
// One column per setting in a one-row batch, so a value is an Arrow array
// rather than a scalar — the setting's type is DuckDB's, not something the
// worker declared, and only Arrow can carry it faithfully.
class Settings {
public:
    static Settings parse(const std::string& ipc_bytes);

    std::shared_ptr<arrow::Array> get(const std::string& name) const;
    std::optional<int64_t> get_int64(const std::string& name) const;
    std::optional<double> get_double(const std::string& name) const;
    std::optional<std::string> get_string(const std::string& name) const;
    std::optional<bool> get_bool(const std::string& name) const;

    bool empty() const noexcept { return values_.empty(); }

private:
    std::map<std::string, std::shared_ptr<arrow::Array>> values_;
};

// Secrets the engine resolved for this call.
//
// Keyed by secret name, then by field. Values are rendered to strings because
// that is what a worker does with them — a token goes into a header, a path
// into a URL — and because the field types vary per secret type.
class Secrets {
public:
    static Secrets parse(const std::string& ipc_bytes);

    // One field of one secret, or nullopt if either is absent.
    std::optional<std::string> field(const std::string& secret_name,
                                     const std::string& field_name) const;
    // Every field of one secret.
    const std::map<std::string, std::string>* secret(const std::string& secret_name) const;

    bool empty() const noexcept { return by_name_.empty(); }

private:
    std::map<std::string, std::map<std::string, std::string>> by_name_;
};

}  // namespace vgi
