// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
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
    // Every field of one secret, by the name the user gave it in
    // `CREATE SECRET <name> (...)`.
    const std::map<std::string, std::string>* secret(const std::string& secret_name) const;

    // Every resolved secret, as (name, fields).
    //
    // Needed because a function almost never knows the *name*: it knows the
    // type it asked for, and the user chose the name. The lookups below are
    // built on this, and so is anything else that has to search.
    const std::map<std::string, std::map<std::string, std::string>>& all() const noexcept {
        return by_name_;
    }

    // The first secret of `type`. Each resolved secret carries its own `type`
    // field, which is what makes this answerable.
    const std::map<std::string, std::string>* of_type(const std::string& type) const;
    // One field of the first secret of `type`.
    std::optional<std::string> typed_field(const std::string& type,
                                           const std::string& field_name) const;

    // The secret of `type` whose scope is the longest prefix of `path`, matched
    // by `scope_matches` (DuckDB's own rule, authority boundary included). A
    // scope is a newline-joined list of prefixes; an unscoped secret of `type`
    // is the fallback when no scoped one matches.
    //
    // Longest-prefix, not first-match: two secrets may both cover a path — one
    // scoped to a bucket and one to a prefix within it — and the more specific
    // is the one the user meant.
    const std::map<std::string, std::string>* for_scope_of_type(const std::string& path,
                                                                const std::string& type) const;

    bool empty() const noexcept { return by_name_.empty(); }

private:
    std::map<std::string, std::map<std::string, std::string>> by_name_;
};

// Whether the secret scope `prefix` covers `path`, exactly as DuckDB decides
// (BaseSecret::MatchScore as of the Haybarn 2.0.1 engine, duckdb#28684).
//
// A scope matches when it is a string prefix of `path`, except that an
// authority-only URI scope (`scheme://authority` with nothing after the
// authority) has an implicit trailing `/`: `snmp://10.0.0.1` covers
// `snmp://10.0.0.1` and `snmp://10.0.0.1/x` but not `snmp://10.0.0.15` or
// `snmp://10.0.0.1:161`, and `s3://bucket` does not cover `s3://bucket-evil/`.
// Matching DuckDB keeps a worker from choosing a secret DuckDB itself would not
// resolve for that path. An empty `prefix` is not a scope and never matches;
// callers treat unscoped secrets as the fallback separately.
bool scope_matches(const std::string& prefix, const std::string& path);

}  // namespace vgi
