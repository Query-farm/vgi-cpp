// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The catalog half of the protocol: what the engine asks at ATTACH and while
// resolving names.

#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <random>
#include <sstream>
#include <string_view>
#include <unistd.h>
#include <tuple>

#include "arg_schema.h"
#include "dispatcher.h"
#include "enums.h"
#include "generated/vgi_protocol_schemas.hpp"
#include "methods.h"
#include "wire.h"

namespace vgi {
namespace {
// A fresh id per ATTACH. Random rather than a counter: a worker pool spreads
// one query's calls over several processes, and two of them would otherwise
// mint the same id for different attachments.
std::string next_attachment_id() {
    static std::mt19937_64 generator{std::random_device{}()};
    std::ostringstream out;
    out << std::hex << generator() << generator();
    return out.str();
}


namespace gen = ::duckdb::vgi::generated;

std::shared_ptr<arrow::Schema> advertised_aggregate_schema(const AggregateFunction& fn) {
    if (auto declared = fn.metadata().return_type) {
        return arrow::schema({arrow::field("result", declared, /*nullable=*/true)});
    }
    try {
        if (auto bound = fn.bind(BindParams{})) return bound;
    } catch (const std::exception&) {
        // bind needs an input schema it cannot have at registration time.
    }
    return build_scalar_output_schema(nullptr);
}

// `"a", "b", "c"` — for an error naming what a worker does serve.
std::string join_quoted(const std::vector<std::string>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ", ";
        out += "\"" + values[i] + "\"";
    }
    return out;
}

const char* stability_wire_value(Stability stability) {
    switch (stability) {
        case Stability::Volatile: return enums::stability::kVolatile;
        case Stability::ConsistentWithinQuery:
            return enums::stability::kConsistentWithinQuery;
        case Stability::Consistent: break;
    }
    return enums::stability::kConsistent;
}

// The partition kind is carried as a raw wire string rather than an enum so a
// fixture can name a kind this SDK does not model; empty means unpartitioned.
// `major.minor.patch`, or nothing when the string is not exactly that.
std::optional<std::array<uint32_t, 3>> parse_semver(const std::string& text) {
    std::array<uint32_t, 3> parts{};
    size_t offset = 0;
    for (int i = 0; i < 3; ++i) {
        if (offset >= text.size() || !std::isdigit(static_cast<unsigned char>(text[offset]))) {
            return std::nullopt;
        }
        size_t consumed = 0;
        parts[static_cast<size_t>(i)] =
            static_cast<uint32_t>(std::stoul(text.substr(offset), &consumed));
        offset += consumed;
        if (i < 2) {
            if (offset >= text.size() || text[offset] != '.') return std::nullopt;
            ++offset;
        }
    }
    return offset == text.size() ? std::optional{parts} : std::nullopt;
}

// The highest supported version an npm-style range covers.
//
// Five forms, which is what the wire actually carries: an exact `X.Y.Z`, a
// caret `^X.Y.Z` (same major), a tilde `~X.Y.Z` (same major.minor), a bare
// major `X`, and a bare `X.Y` pinned to `X.Y.0`. An empty request takes the
// catalog's default; anything else is an ATTACH-time error, because a caller
// asking for something this worker cannot serve should learn so now rather
// than mid-query.
std::string resolve_version_npm(const std::optional<std::string>& spec,
                                const std::vector<std::string>& supported,
                                const std::string& fallback, const char* label) {
    if (!spec || spec->empty()) return fallback;

    std::vector<std::pair<std::array<uint32_t, 3>, const std::string*>> sorted;
    for (const auto& version : supported) {
        if (auto parsed = parse_semver(version)) sorted.emplace_back(*parsed, &version);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    const auto unsupported = [&] {
        return std::invalid_argument("Unsupported " + std::string(label) + " \"" + *spec +
                                     "\"; this worker serves one of " + join_quoted(supported));
    };
    // Highest first, so the first match is the answer.
    const auto best = [&](auto&& covers) -> std::string {
        for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
            if (covers(it->first)) return *it->second;
        }
        throw unsupported();
    };

    if (parse_semver(*spec)) {
        if (std::find(supported.begin(), supported.end(), *spec) == supported.end()) {
            throw unsupported();
        }
        return *spec;
    }

    const char prefix = spec->front();
    const bool ranged = prefix == '^' || prefix == '~';
    const std::string bare = ranged ? spec->substr(1) : *spec;

    if (ranged) {
        const auto base = parse_semver(bare);
        if (!base) throw unsupported();
        return best([&](const std::array<uint32_t, 3>& v) {
            return v[0] == (*base)[0] && v >= *base && (prefix == '^' || v[1] == (*base)[1]);
        });
    }

    const auto dot = bare.find('.');
    if (dot == std::string::npos) {
        uint32_t major = 0;
        try {
            major = static_cast<uint32_t>(std::stoul(bare));
        } catch (const std::exception&) {
            throw unsupported();
        }
        return best([&](const std::array<uint32_t, 3>& v) { return v[0] == major; });
    }
    // `X.Y` pins to `X.Y.0` rather than taking the highest patch, matching the
    // reference implementations.
    if (bare.find('.', dot + 1) == std::string::npos) {
        const auto pinned = bare + ".0";
        if (std::find(supported.begin(), supported.end(), pinned) == supported.end()) {
            throw unsupported();
        }
        return pinned;
    }
    throw unsupported();
}

const char* partition_kind_wire_value(const FunctionMetadata& metadata) {
    return metadata.partition_kind.empty() ? partition_kinds::kNotPartitioned
                                           : metadata.partition_kind.c_str();
}

// Wrap a payload batch in the `{result: binary}` envelope every non-void
// method answers with.  The engine unwraps it and validates the inner schema
// against its own generated copy, so a drifted payload is caught there rather
// than silently misread.
vgi_rpc::Result envelope(const std::shared_ptr<arrow::RecordBatch>& payload) {
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(payload))
                                      .finish());
}

// An empty `items` list, which is the honest answer for every catalog category
// a worker does not populate.  The payload differs per method only in name, so
// the shape is shared.
vgi_rpc::Result empty_items(const std::string& method) {
    return envelope(wire::ResultBuilder(payload_schema_of(method))
                        .set_binary_list("items", {})
                        .finish());
}

}  // namespace

// One IPC entry per declared setting.
//
// The batch schema is `{name, description, type, default_value}`, where `type`
// is itself the IPC schema of a single `value` field of the setting's type —
// a schema inside a schema, which is how a setting's type travels without a
// type-name vocabulary of its own.
// Whether any table time-travels, or the catalog declared it outright.
bool Dispatcher::supports_time_travel(const CatalogModel& model) const {
    if (model.supports_time_travel) return true;
    for (const auto& schema : model.schemas) {
        for (const auto& table : schema->tables) {
            if (!table.time_travel.empty()) return true;
        }
    }
    return false;
}

std::vector<std::string> Dispatcher::encode_settings(const CatalogModel& model) const {
    static const auto schema = arrow::schema({
        arrow::field("name", arrow::utf8(), /*nullable=*/false),
        arrow::field("description", arrow::utf8(), /*nullable=*/false),
        arrow::field("type", arrow::binary(), /*nullable=*/false),
        arrow::field("default_value", arrow::binary(), /*nullable=*/true),
    });

    std::vector<std::string> entries;
    entries.reserve(model.settings.size());
    for (const auto& setting : model.settings) {
        auto value_schema =
            arrow::schema({arrow::field("value", setting.type, /*nullable=*/true)});
        entries.push_back(wire::encode_ipc(wire::ResultBuilder(schema)
                                               .set_string("name", setting.name)
                                               .set_string("description", setting.description)
                                               .set_binary("type", wire::encode_schema(value_schema))
                                               .set_null("default_value")
                                               .finish()));
    }
    return entries;
}

// The `required_secrets` list, as the struct entries the wire carries.
//
// Written for every function kind. Only the two-phase bind path consulted the
// metadata before, and aggregates have no such path — so an aggregate that
// declared a secret could never receive one.
std::vector<std::tuple<std::string, std::string, std::string>> secret_entries(
    const FunctionMetadata& metadata) {
    std::vector<std::tuple<std::string, std::string, std::string>> entries;
    entries.reserve(metadata.required_secrets.size());
    for (const auto& lookup : metadata.required_secrets) {
        entries.emplace_back(lookup.secret_type, lookup.scope.value_or(""),
                             lookup.secret_name.value_or(""));
    }
    return entries;
}

// One IPC entry per declared secret type.
//
// `parameters_schema` is the IPC schema of the secret's fields, metadata and
// all — which is how the `redact=true` marker on a field reaches the engine
// and keeps the value out of logs and error messages.
std::vector<std::string> Dispatcher::encode_secret_types(const CatalogModel& model) const {
    static const auto schema = arrow::schema({
        arrow::field("name", arrow::utf8(), /*nullable=*/false),
        arrow::field("description", arrow::utf8(), /*nullable=*/false),
        arrow::field("parameters_schema", arrow::binary(), /*nullable=*/false),
    });

    std::vector<std::string> entries;
    entries.reserve(model.secret_types.size());
    for (const auto& secret : model.secret_types) {
        entries.push_back(
            wire::encode_ipc(wire::ResultBuilder(schema)
                                 .set_string("name", secret.name)
                                 .set_string("description", secret.description)
                                 .set_binary("parameters_schema",
                                             wire::encode_schema(secret.parameters))
                                 .finish()));
    }
    return entries;
}

const CatalogSchema* Dispatcher::schema_for(const vgi_rpc::Request& request,
                                            const std::string& name) const {
    const auto attachment = attachment_of(request);
    const auto* model = find_catalog(attachment.catalog);
    if (!model) model = &catalog();

    // A version-shaped catalog answers from the version this attachment
    // resolved to; every other one from its declared schemas.
    if (!attachment.data_version.empty()) {
        const auto version = model->version_schemas.find(attachment.data_version);
        if (version != model->version_schemas.end()) {
            for (const auto& schema : version->second) {
                if (schema.name == name) return &schema;
            }
            return nullptr;
        }
    }
    return model->find_schema(name);
}

std::vector<std::string> Dispatcher::encode_global_functions(const CatalogModel& model) const {
    // Published from `main`, because a global name is an alias the engine
    // resolves back to a schema-resident registration — bind dispatch is keyed
    // on (schema, name), and there is nowhere else for it to look.
    std::vector<std::string> items;
    items.reserve(model.global_functions.size());
    for (const auto& name : model.global_functions) {
        if (auto fn = find_buffering(name, {model.name, "main"})) {
            items.push_back(encode_buffering_info(*fn, "main"));
            continue;
        }
        if (auto transform = find_table_in_out(name, {model.name, "main"})) {
            items.push_back(encode_table_in_out_info(*transform, "main"));
            continue;
        }
        if (auto table = find_table(name, {model.name, "main"})) {
            items.push_back(encode_table_function_info(*table, "main"));
            continue;
        }
        const auto aggregates = aggregates_in_schema({model.name, "main"});
        const auto aggregate = std::find_if(aggregates.begin(), aggregates.end(),
                                            [&](const auto& fn) { return fn->name() == name; });
        if (aggregate != aggregates.end()) {
            items.push_back(encode_aggregate_info(**aggregate, "main"));
            continue;
        }
        // An overload set publishes under one name, so the first registration
        // is the one the engine is told about.
        const auto scalars = scalars_named(name, {model.name, "main"});
        if (scalars.empty()) {
            throw std::invalid_argument("catalog publishes '" + name +
                                        "' globally but no function of that name is registered");
        }
        items.push_back(encode_function_info(*scalars.front(), "main"));
    }
    return items;
}

vgi_rpc::Result Dispatcher::catalog_attach(const vgi_rpc::Request& request) {
    // The request dataclass rides in one binary column as a self-describing
    // IPC stream; the params schema is only ever {request: binary}.
    auto attach = wire::get_ipc(request.batch(), "request");
    if (!attach) throw std::runtime_error("catalog_attach: empty request");

    // `name` is the catalog the user asked for. It is also the ATTACH alias
    // when they wrote one, which is why it is echoed back in the seal — but
    // what selects the model is the name matching a catalog this worker
    // serves; anything else attaches the primary, which is what a worker with
    // one catalog has always done under any alias.
    const auto name = wire::get_string(attach, "name");
    const auto* requested = find_catalog(name);
    const CatalogModel& model = requested ? *requested : catalog();

    // Version negotiation. An exact match against what this worker serves,
    // rather than a range check: a caller asking for something the worker
    // cannot serve should be told at ATTACH, not discover it mid-query.
    const auto requested_data = wire::get_optional_string(attach, "data_version_spec");
    const auto requested_impl = wire::get_optional_string(attach, "implementation_version");

    std::optional<std::string> resolved_data;
    if (!model.supported_data_versions.empty()) {
        if (model.npm_version_resolution) {
            resolved_data =
                resolve_version_npm(requested_data, model.supported_data_versions,
                                    model.default_data_version.value_or(""),
                                    "data_version_spec");
        } else if (requested_data) {
            const auto& supported = model.supported_data_versions;
            if (std::find(supported.begin(), supported.end(), *requested_data) ==
                supported.end()) {
                throw std::invalid_argument("Unsupported data_version_spec \"" +
                                            *requested_data + "\"; this worker serves one of " +
                                            join_quoted(supported));
            }
            resolved_data = requested_data;
        } else {
            resolved_data = model.default_data_version;
        }
    }

    std::optional<std::string> resolved_impl;
    if (!model.implementation_version.empty()) {
        const auto& supported = model.supported_implementation_versions.empty()
                                    ? std::vector<std::string>{model.implementation_version}
                                    : model.supported_implementation_versions;
        if (model.npm_version_resolution &&
            !model.supported_implementation_versions.empty()) {
            resolved_impl = resolve_version_npm(requested_impl, supported,
                                                model.implementation_version,
                                                "implementation_version");
        } else if (requested_impl) {
            if (std::find(supported.begin(), supported.end(), *requested_impl) ==
                supported.end()) {
                throw std::invalid_argument("Unsupported implementation_version \"" +
                                            *requested_impl + "\"; this worker serves " +
                                            join_quoted(supported));
            }
            resolved_impl = requested_impl;
        } else {
            resolved_impl = model.implementation_version;
        }
    }

    auto batch = wire::ResultBuilder(payload_schema_of("catalog_attach"))
                     // Opaque to the engine, which only stores and echoes it.
                     // Both the catalog and the resolved version have to
                     // travel: most later calls carry nothing else that says
                     // which attachment they belong to, and one worker may
                     // serve several catalogs whose schemas and function names
                     // collide.
                     .set_binary("attach_opaque_data",
                                 seal_attachment({model.name, resolved_data.value_or(""), name,
                                                  next_attachment_id()}))
                     .set_bool("supports_transactions", model.supports_transactions)
                     .set_bool("supports_time_travel", supports_time_travel(model))
                     .set_bool("catalog_version_frozen", true)
                     .set_int64("catalog_version", 1)
                     .set_bool("attach_opaque_data_required", true)
                     .set_string("default_schema", "main")
                     // On, and per-table from here: the flag gates every
                     // declared statistic, so leaving it off silently
                     // discarded them all.
                     .set_bool("supports_column_statistics", true)
                     .set_binary_list("global_functions", encode_global_functions(model))
                     .set_string("global_function_prefix", model.global_function_prefix)
                     .set_string_map("tags", model.tags)
                     .set_optional_string("resolved_data_version", resolved_data)
                     .set_optional_string("resolved_implementation_version", resolved_impl)
                     .set_binary_list("settings", encode_settings(model))
                     .set_binary_list("secret_types", encode_secret_types(model));
    if (model.comment) {
        batch.set_string("comment", *model.comment);
    } else {
        batch.set_null("comment");
    }
    return envelope(batch.fill_defaults().finish());
}

vgi_rpc::Result Dispatcher::catalog_version(const vgi_rpc::Request&) {
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_version"))
                                      .set_int64("version", 1)
                                      .finish());
}

void Dispatcher::catalog_detach(const vgi_rpc::Request&) {
    // Nothing is held per attachment yet. When something is, it is freed here.
}

namespace {

// The storage scope one transaction's state lives under.
std::string transaction_scope(const std::string& id) { return "txn:" + id; }

}  // namespace

vgi_rpc::Result Dispatcher::catalog_transaction_begin(const vgi_rpc::Request&) {
    // Minted here rather than by the engine, and unique across processes: the
    // pool hands a different worker to each RPC of one transaction, so two
    // transactions that collided on an id would read each other's state
    // through the shared store.
    static std::atomic<uint64_t> counter{0};
    const auto id = std::to_string(static_cast<uint64_t>(::getpid())) + '-' +
                    std::to_string(counter.fetch_add(1)) + '-' +
                    std::to_string(static_cast<uint64_t>(
                        std::chrono::steady_clock::now().time_since_epoch().count()));
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_transaction_begin"))
                        .set_binary("transaction_opaque_data", id)
                        .finish());
}

void Dispatcher::catalog_transaction_commit(const vgi_rpc::Request& request) {
    // Committing ends the transaction, so whatever it remembered goes with it.
    // Rollback is the same discard — nothing here is written through to
    // anything durable, so there is nothing to undo.
    const auto id = wire::get_optional_binary(request.batch(), "transaction_opaque_data");
    if (id && !id->empty()) default_storage()->clear(transaction_scope(*id));
}

void Dispatcher::catalog_transaction_rollback(const vgi_rpc::Request& request) {
    catalog_transaction_commit(request);
}

vgi_rpc::Result Dispatcher::catalog_catalogs(const vgi_rpc::Request&) {
    // Every catalog this worker serves. Discovery runs before any ATTACH, so
    // what each entry says is what the catalog *is* — not what some attachment
    // resolved it to.
    std::vector<std::string> items;
    items.reserve(catalogs_.size());
    for (const auto& model : catalogs_) {
        auto builder = wire::ResultBuilder(gen::CatalogInfoSchema());
        builder.set_string("name", model->name);
        builder.set_optional_string(
            "implementation_version",
            model->implementation_version.empty()
                ? std::nullopt
                : std::optional<std::string>(model->implementation_version));
        builder.set_optional_string("data_version_spec", model->data_version_spec);
        builder.set_optional_string(
            "source_url",
            model->source_url.empty() ? std::nullopt
                                      : std::optional<std::string>(model->source_url));
        items.push_back(wire::encode_ipc(builder.fill_defaults().finish()));
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_catalogs"))
                        .set_binary_list("items", std::move(items))
                        .finish());
}

// A table's wire record.
//
// `scan_function` and `scan_arguments` are what make a VGI table work: the
// table is a name bound to a function, and this is the binding. Inlining the
// scan here saves the engine a `catalog_table_scan_function_get` round trip
// per query.
// The version an AT clause selects, or null for a table that does not
// time-travel.
//
// Throws rather than falling back when the clause names a version that does
// not exist: silently serving the current one would answer a question the user
// did not ask.
const TimeTravelVersion* resolve_version(const CatalogTable& table,
                                         const std::optional<std::string>& at_unit,
                                         const std::optional<std::string>& at_value) {
    const bool has_clause = at_unit && !at_unit->empty();
    if (table.time_travel.empty()) {
        if (has_clause && !table.branches && !table.supports_time_travel) {
            throw std::invalid_argument("this table does not support time travel");
        }
        return nullptr;
    }

    const TimeTravelVersion* newest = nullptr;
    for (const auto& version : table.time_travel) {
        if (!newest || version.version > newest->version) newest = &version;
    }
    if (!has_clause) return newest;

    std::string unit = *at_unit;
    for (auto& c : unit) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    if (unit == "VERSION") {
        const int64_t wanted = at_value ? std::strtoll(at_value->c_str(), nullptr, 10) : -1;
        for (const auto& version : table.time_travel) {
            if (version.version == wanted) return &version;
        }
        throw std::invalid_argument("Unknown version: " + at_value.value_or(""));
    }
    if (unit == "TIMESTAMP") {
        // Only the year is compared: the fixtures step versions by year, and a
        // finer comparison would need a real timestamp parser for no gain.
        const int year =
            at_value && at_value->size() >= 4 ? std::atoi(at_value->substr(0, 4).c_str()) : 0;
        const TimeTravelVersion* best = nullptr;
        for (const auto& version : table.time_travel) {
            if (!version.timestamp_year || *version.timestamp_year > year) continue;
            if (!best || version.version > best->version) best = &version;
        }
        if (!best) {
            // Naming the earliest year rather than the requested timestamp:
            // the user's question is "what did this look like then", and the
            // useful answer is when "then" starts being answerable.
            int earliest = 0;
            for (const auto& version : table.time_travel) {
                if (version.timestamp_year && (earliest == 0 || *version.timestamp_year < earliest)) {
                    earliest = *version.timestamp_year;
                }
            }
            throw std::invalid_argument("table did not exist before " + std::to_string(earliest));
        }
        return best;
    }
    throw std::invalid_argument("unsupported AT unit: " + unit);
}

namespace {

// Constraint columns that exist in a schema of `column_count` fields.
std::vector<int32_t> within(const std::vector<int32_t>& columns, int column_count) {
    std::vector<int32_t> kept;
    for (const auto column : columns) {
        if (column >= 0 && column < column_count) kept.push_back(column);
    }
    return kept;
}

// The same, for a list of column groups. A group that loses any member is
// dropped whole: a composite key missing a column is a different constraint,
// not a narrower one.
std::vector<std::vector<int32_t>> within(const std::vector<std::vector<int32_t>>& groups,
                                         int column_count) {
    std::vector<std::vector<int32_t>> kept;
    for (const auto& group : groups) {
        if (within(group, column_count).size() == group.size()) kept.push_back(group);
    }
    return kept;
}

// One IPC entry per foreign key. The referenced schema travels alongside the
// table because the wire allows a cross-schema reference, even though every
// fixture here stays inside one.
std::vector<std::string> encode_foreign_keys(const CatalogTable& table,
                                             const std::string& schema_name) {
    static const auto schema = arrow::schema({
        arrow::field("fk_columns", arrow::list(arrow::utf8()), /*nullable=*/true),
        arrow::field("pk_columns", arrow::list(arrow::utf8()), /*nullable=*/true),
        arrow::field("referenced_table", arrow::utf8(), /*nullable=*/true),
        arrow::field("referenced_schema", arrow::utf8(), /*nullable=*/true),
    });

    std::vector<std::string> entries;
    entries.reserve(table.foreign_keys.size());
    for (const auto& key : table.foreign_keys) {
        entries.push_back(wire::encode_ipc(
            wire::ResultBuilder(schema)
                .set_string_list("fk_columns", key.columns)
                .set_string_list("pk_columns", key.referenced_columns)
                .set_string("referenced_table", key.referenced_table)
                .set_string("referenced_schema", schema_name)
                .finish()));
    }
    return entries;
}

}  // namespace

std::string Dispatcher::encode_table_info(const CatalogTable& table,
                                          const std::string& schema_name,
                                          const TimeTravelVersion* version) {
    auto scan =
        wire::ResultBuilder(gen::ScanFunctionResultSchema())
            .set_string("function_name", version ? version->scan_function : table.scan_function)
            .set_binary("arguments", version ? version->scan_arguments : table.scan_arguments)
            .set_string_list("required_extensions", {})
            .finish();

    // An older version has fewer columns, and the constraints are declared
    // against the newest. One naming a column this version does not have is
    // dropped rather than sent: the engine rejects an out-of-range index and
    // the whole AT read fails.
    const auto& shape = version ? version->columns : table.columns;
    const int column_count = shape ? shape->num_fields() : 0;

    auto builder = wire::ResultBuilder(gen::TableInfoSchema());
    builder.set_string("name", table.name)
        .set_string("schema_name", schema_name)
        .set_binary("columns",
                    wire::encode_schema(version ? version->columns : table.columns))
        .set_binary("scan_function", table.inline_scan ? wire::encode_ipc(scan) : std::string{})
        .set_bool("supports_insert", false)
        .set_bool("supports_update", false)
        .set_bool("supports_delete", false)
        .set_bool("supports_returning", false)
        .set_int32_list("not_null_constraints", within(table.not_null, column_count))
        .set_int32_list_list("unique_constraints", within(table.unique, column_count))
        .set_int32_list_list("primary_key_constraints", within(table.primary_key, column_count))
        .set_string_list("check_constraints", table.check)
        .set_binary_list("foreign_key_constraints", encode_foreign_keys(table, schema_name))
        .set_bool("supports_column_statistics", !table.column_statistics.empty())
        .set_string_map("tags", table.tags)
        .set_string_list_list("required_filters", table.required_filters);
    // Null, not -1, when the table does not know its size. The engine reads
    // these as optional, so a present -1 *is* an answer: it takes it, skips
    // `table_function_cardinality` entirely, and DuckDB clamps it to one row.
    if (table.cardinality) {
        builder.set_int64("cardinality_estimate", *table.cardinality)
            .set_int64("cardinality_max", *table.cardinality);
    } else {
        builder.set_null("cardinality_estimate").set_null("cardinality_max");
    }
    if (table.comment) {
        builder.set_string("comment", *table.comment);
    } else {
        builder.set_null("comment");
    }
    return wire::encode_ipc(builder.fill_defaults().finish());
}

vgi_rpc::Result Dispatcher::catalog_table_column_statistics_get(
    const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "schema_name");
    const auto name = wire::get_string(request.batch(), "name");

    // A Binary method: the bytes are the statistics batch, and a null is "this
    // table declares none" — which sends the engine on to the scan function.
    wire::ResultBuilder result(envelope_schema());
    const CatalogTable* found = nullptr;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& table : schema->tables) {
            if (table.name == name) found = &table;
        }
    }
    if (found && !found->column_statistics.empty()) {
        result.set_binary("result", serialize_column_statistics(found->column_statistics));
    } else {
        result.set_null("result");
    }
    return vgi_rpc::Result::value(result.finish());
}

vgi_rpc::Result Dispatcher::catalog_table_get(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "schema_name");
    const auto name = wire::get_string(request.batch(), "name");

    const auto at_unit = wire::get_optional_string(request.batch(), "at_unit");
    const auto at_value = wire::get_optional_string(request.batch(), "at_value");

    std::vector<std::string> items;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& table : schema->tables) {
            if (table.name != name) continue;
            items.push_back(encode_table_info(table, schema_name,
                                              resolve_version(table, at_unit, at_value)));
        }
    }
    // Zero or one item; absence is how "no such table" is spelled, and the
    // engine goes on to look for a function of that name.
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_table_get"))
                        .set_binary_list("items", items)
                        .finish());
}

std::string Dispatcher::encode_view_info(const CatalogView& view,
                                         const std::string& schema_name) {
    auto builder = wire::ResultBuilder(gen::ViewInfoSchema());
    builder.set_string("name", view.name)
        .set_string("schema_name", schema_name)
        // `definition`, not `sql` — the latter is not a ViewInfo field, and
        // ResultBuilder refuses an unknown one, so every view encode threw and
        // took `SHOW TABLES` on the schema with it.
        .set_string("definition", view.definition);
    if (view.comment) {
        builder.set_string("comment", *view.comment);
    } else {
        builder.set_null("comment");
    }
    return wire::encode_ipc(builder.fill_defaults().finish());
}

vgi_rpc::Result Dispatcher::catalog_view_get(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "schema_name");
    const auto name = wire::get_string(request.batch(), "name");
    std::vector<std::string> items;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& view : schema->views) {
            if (view.name == name) items.push_back(encode_view_info(view, schema_name));
        }
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_view_get"))
                        .set_binary_list("items", items)
                        .finish());
}

// The two blobs a macro carries beyond its name and text.
//
// `parameter_default_values` is a one-row batch, one column per defaulted
// parameter. `arguments_schema` is one nullable field per parameter *in
// declaration order* — order is the positional call order — typed from the
// default where there is one, carrying the description as `vgi_doc`.
// The wire spells a kind either bare (`scalar`) or suffixed
// (`scalar_function`); both mean the same thing.
static std::optional<std::string> normalize_function_type(const std::string& type) {
    if (type.empty()) return std::nullopt;
    std::string lower = type;
    for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    static constexpr std::string_view kSuffix = "_function";
    if (lower.size() > kSuffix.size() &&
        lower.compare(lower.size() - kSuffix.size(), kSuffix.size(), kSuffix) == 0) {
        lower.resize(lower.size() - kSuffix.size());
    }
    return lower;
}

std::string Dispatcher::encode_macro_info(const CatalogMacro& macro,
                                          const std::string& schema_name) {
    std::string defaults;
    if (!macro.defaults.empty()) {
        arrow::FieldVector fields;
        std::vector<std::shared_ptr<arrow::Array>> values;
        for (const auto& [name, value] : macro.defaults) {
            fields.push_back(arrow::field(name, arrow::int64(), /*nullable=*/false));
            arrow::Int64Builder builder;
            (void)builder.Append(value);
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            values.push_back(std::move(array));
        }
        defaults = wire::encode_ipc(arrow::RecordBatch::Make(arrow::schema(fields), 1, values));
    }

    std::string arguments;
    if (!macro.parameters.empty() || !macro.parameter_docs.empty()) {
        arrow::FieldVector fields;
        for (const auto& name : macro.parameters) {
            const bool has_default =
                std::any_of(macro.defaults.begin(), macro.defaults.end(),
                            [&](const auto& entry) { return entry.first == name; });
            auto field = arrow::field(name, has_default ? arrow::int64() : arrow::null(),
                                      /*nullable=*/true);
            const auto doc = std::find_if(
                macro.parameter_docs.begin(), macro.parameter_docs.end(),
                [&](const auto& entry) { return entry.first == name; });
            // Presence-only, as everywhere else: an empty doc is an absent key.
            if (doc != macro.parameter_docs.end() && !doc->second.empty()) {
                field = field->WithMetadata(
                    arrow::key_value_metadata({"vgi_doc"}, {doc->second}));
            }
            fields.push_back(std::move(field));
        }
        arguments = wire::encode_schema(arrow::schema(fields));
    }

    auto builder = wire::ResultBuilder(gen::MacroInfoSchema());
    builder.set_string("name", macro.name)
        .set_string("schema_name", schema_name)
        .set_enum("macro_type", macro.table_macro ? "table" : "scalar")
        .set_string_list("parameters", macro.parameters)
        .set_binary("parameter_default_values", defaults)
        .set_string("definition", macro.definition)
        .set_binary("arguments_schema", arguments);
    if (macro.comment) {
        builder.set_string("comment", *macro.comment);
    } else {
        builder.set_null("comment");
    }
    return wire::encode_ipc(builder.fill_defaults().finish());
}

vgi_rpc::Result Dispatcher::catalog_macro_get(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "schema_name");
    const auto name = wire::get_string(request.batch(), "name");
    std::vector<std::string> items;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& macro : schema->macros) {
            if (macro.name == name) items.push_back(encode_macro_info(macro, schema_name));
        }
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_macro_get"))
                        .set_binary_list("items", items)
                        .finish());
}

vgi_rpc::Result Dispatcher::catalog_index_get(const vgi_rpc::Request&) {
    return empty_items("catalog_index_get");
}

std::string Dispatcher::encode_schema_info(const std::string& owner,
                                           const CatalogSchema& schema,
                                           const CatalogSchema* contents) const {
    // `contents` is what this attachment will actually be shown, which is not
    // `schema` for a catalog whose table set varies by resolved data version:
    // counting the declared (empty) schema would promise zero tables and the
    // engine would never ask for them.
    if (!contents) contents = &schema;
    // The engine reads a zero as a promise, not an estimate: it then skips
    // both the bulk `catalog_schema_contents_*` call and every per-name lookup
    // for that kind. Undercounting therefore hides objects rather than costing
    // a round trip, which is why these are exact counts and not guesses.
    const auto size = [](const auto& registrations) {
        return static_cast<int64_t>(registrations.size());
    };

    auto builder = wire::ResultBuilder(gen::SchemaInfoSchema())
                       .set_string("name", schema.name)
                       .set_string_map("tags", schema.tags)
                       .set_binary("attach_opaque_data", owner)
                       .set_int64_map(
                           "estimated_object_count",
                           {{"view", static_cast<int64_t>(contents->views.size())},
                            {"macro", static_cast<int64_t>(contents->macros.size())},
                            {"table", static_cast<int64_t>(contents->tables.size())},
                            {"scalar_function", size(scalars_in_schema({owner, schema.name}))},
                            {"aggregate_function", size(aggregates_in_schema({owner, schema.name}))},
                            // Every kind the engine registers as a table
                            // function, which is three of ours: a table-in-out
                            // and a buffering sink are table functions to it.
                            {"table_function",
                             size(tables_in_schema({owner, schema.name})) +
                                 size(table_in_outs_in_schema({owner, schema.name})) +
                                 size(bufferings_in_schema({owner, schema.name}))},
                            {"index", 0}});
    if (schema.comment) {
        builder.set_string("comment", *schema.comment);
    } else {
        builder.set_null("comment");
    }
    return wire::encode_ipc(builder.fill_defaults().finish());
}

vgi_rpc::Result Dispatcher::catalog_schemas(const vgi_rpc::Request& request) {
    std::vector<std::string> items;

    const auto owner = attachment_of(request).catalog;
    const auto* model = find_catalog(owner);
    if (!model) model = &catalog();
    for (const auto& schema : model->schemas) {
        items.push_back(encode_schema_info(owner, *schema, schema_for(request, schema->name)));
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_schemas"))
                                      .set_binary_list("items", items)
                                      .finish());
}

vgi_rpc::Result Dispatcher::catalog_schema_get(const vgi_rpc::Request& request) {
    const auto wanted = wire::get_string(request.batch(), "name");
    std::vector<std::string> items;
    const auto owner = attachment_of(request).catalog;
    const auto* model = find_catalog(owner);
    if (!model) model = &catalog();
    for (const auto& schema : model->schemas) {
        if (schema->name != wanted) continue;
        items.push_back(encode_schema_info(owner, *schema, schema_for(request, wanted)));
    }
    // Zero or one item; the engine reads absence as "no such schema".
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_schema_get"))
                                      .set_binary_list("items", items)
                                      .finish());
}

std::string Dispatcher::encode_table_function_info(const TableFunction& fn,
                                                   const std::string& schema_name) {
    const auto metadata = fn.metadata();
    // A table function's output schema is settled at bind, so nothing useful
    // can be advertised here; an empty schema is how that is spelled.
    auto builder =
        wire::ResultBuilder(gen::FunctionInfoSchema())
            .set_string("name", fn.name())
            .set_string("schema_name", schema_name)
            .set_enum("function_type", enums::function_type::kTable)
            .set_binary("arguments", wire::encode_schema(build_arg_schema(fn.argument_specs())))
            .set_binary("output_schema", wire::encode_schema(arrow::schema({})))
            .set_enum("stability", stability_wire_value(metadata.stability))
            .set_enum("null_handling", metadata.null_handling == NullHandling::Special
                                           ? enums::null_handling::kSpecial
                                           : enums::null_handling::kDefault)
            .set_string("description", metadata.description)
            .set_examples("examples", metadata.examples)
            .set_string_list("categories", metadata.categories)
            .set_string_map("tags", metadata.tags)
            .set_string_list("required_settings", metadata.required_settings)
            .set_secret_lookups("required_secrets", secret_entries(metadata))
            .set_bool("projection_pushdown", metadata.projection_pushdown)
            .set_bool("filter_pushdown", metadata.filter_pushdown)
            .set_bool("sampling_pushdown", metadata.sampling_pushdown)
            .set_bool("late_materialization", metadata.late_materialization)
            .set_bool("supports_batch_index", metadata.supports_batch_index)
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", partition_kind_wire_value(metadata))
            .set_enum("order_dependent", enums::order_dependence::kNotOrderDependent)
            .set_enum("distinct_dependent", enums::distinct_dependence::kNotDistinctDependent);
    // Only when the function says so: the field is nullable, and writing a
    // value unconditionally would replace the engine's default for every
    // table function that never thought about ordering.
    if (!metadata.order_preservation.empty()) {
        builder.set_enum("order_preservation", metadata.order_preservation.c_str());
    }
    return wire::encode_ipc(builder.fill_defaults().finish());
}

std::string Dispatcher::encode_table_in_out_info(const TableInOutFunction& fn,
                                                 const std::string& schema_name) {
    const auto metadata = fn.metadata();
    // The engine sees a table-in-out as a table function that happens to take
    // a table argument; there is no separate wire kind for it.
    return wire::encode_ipc(
        wire::ResultBuilder(gen::FunctionInfoSchema())
            .set_string("name", fn.name())
            .set_string("schema_name", schema_name)
            .set_enum("function_type", enums::function_type::kTable)
            .set_binary("arguments", wire::encode_schema(build_arg_schema(fn.argument_specs())))
            .set_binary("output_schema", wire::encode_schema(arrow::schema({})))
            .set_enum("stability", stability_wire_value(metadata.stability))
            .set_enum("null_handling", metadata.null_handling == NullHandling::Special
                                           ? enums::null_handling::kSpecial
                                           : enums::null_handling::kDefault)
            .set_string("description", metadata.description)
            .set_examples("examples", metadata.examples)
            .set_string_list("categories", metadata.categories)
            .set_string_map("tags", metadata.tags)
            .set_string_list("required_settings", metadata.required_settings)
            .set_secret_lookups("required_secrets", secret_entries(metadata))
            .set_bool("projection_pushdown", metadata.projection_pushdown)
            .set_bool("filter_pushdown", metadata.filter_pushdown)
            .set_bool("sampling_pushdown", metadata.sampling_pushdown)
            .set_bool("has_finalize", fn.has_finish())
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", partition_kind_wire_value(metadata))
            .set_enum("order_dependent", enums::order_dependence::kNotOrderDependent)
            .set_enum("distinct_dependent", enums::distinct_dependence::kNotDistinctDependent)
            .fill_defaults()
            .finish());
}

std::string Dispatcher::encode_buffering_info(const TableBufferingFunction& fn,
                                              const std::string& schema_name) {
    const auto metadata = fn.metadata();
    return wire::encode_ipc(
        wire::ResultBuilder(gen::FunctionInfoSchema())
            .set_string("name", fn.name())
            .set_string("schema_name", schema_name)
            .set_enum("function_type", enums::function_type::kTableBuffering)
            .set_binary("arguments", wire::encode_schema(build_arg_schema(fn.argument_specs())))
            .set_binary("output_schema", wire::encode_schema(arrow::schema({})))
            .set_enum("stability", stability_wire_value(metadata.stability))
            .set_enum("null_handling", metadata.null_handling == NullHandling::Special
                                           ? enums::null_handling::kSpecial
                                           : enums::null_handling::kDefault)
            .set_string("description", metadata.description)
            .set_examples("examples", metadata.examples)
            .set_string_list("categories", metadata.categories)
            .set_string_map("tags", metadata.tags)
            .set_string_list("required_settings", metadata.required_settings)
            .set_secret_lookups("required_secrets", secret_entries(metadata))
            .set_bool("projection_pushdown", metadata.projection_pushdown)
            .set_bool("filter_pushdown", metadata.filter_pushdown)
            .set_bool("sampling_pushdown", metadata.sampling_pushdown)
            .set_bool("requires_input_batch_index", metadata.requires_input_batch_index)
            .set_bool("sink_order_dependent", metadata.sink_order_dependent)
            .set_bool("source_order_dependent", metadata.source_order_dependent)
            .set_bool("supports_batch_index", metadata.supports_batch_index)
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", partition_kind_wire_value(metadata))
            .set_enum("order_dependent", enums::order_dependence::kNotOrderDependent)
            .set_enum("distinct_dependent", enums::distinct_dependence::kNotDistinctDependent)
            .fill_defaults()
            .finish());
}

std::string Dispatcher::encode_aggregate_info(const AggregateFunction& fn,
                                              const std::string& schema_name) {
    const auto metadata = fn.metadata();
    return wire::encode_ipc(
        wire::ResultBuilder(gen::FunctionInfoSchema())
            .set_string("name", fn.name())
            .set_string("schema_name", schema_name)
            .set_enum("function_type", enums::function_type::kAggregate)
            .set_binary("arguments", wire::encode_schema(build_arg_schema(fn.argument_specs())))
            // An aggregate must advertise exactly one output field — the
            // engine rejects a zero-field schema outright. A declared return
            // type wins; otherwise ask bind with empty params, which answers
            // for any aggregate whose type does not depend on its input. One
            // that does throws, and the `vgi:any` marker defers the type.
            .set_binary("output_schema", wire::encode_schema(advertised_aggregate_schema(fn)))
            .set_enum("stability", stability_wire_value(metadata.stability))
            .set_enum("null_handling", metadata.null_handling == NullHandling::Special
                                           ? enums::null_handling::kSpecial
                                           : enums::null_handling::kDefault)
            .set_string("description", metadata.description)
            .set_examples("examples", metadata.examples)
            .set_string_list("categories", metadata.categories)
            .set_string_map("tags", metadata.tags)
            .set_string_list("required_settings", metadata.required_settings)
            .set_secret_lookups("required_secrets", secret_entries(metadata))
            .set_bool("projection_pushdown", metadata.projection_pushdown)
            .set_bool("filter_pushdown", metadata.filter_pushdown)
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", partition_kind_wire_value(metadata))
            .set_enum("order_dependent", enums::order_dependence::kNotOrderDependent)
            .set_enum("distinct_dependent", enums::distinct_dependence::kNotDistinctDependent)
            // Declared, or the engine never sends a window request at all and
            // an aggregate that implements `window` is simply never asked.
            .set_bool("supports_window", fn.supports_window())
            .set_bool("streaming_partitioned", fn.streaming_partitioned())
            .fill_defaults()
            .finish());
}

std::string Dispatcher::encode_function_info(const ScalarFunction& fn,
                                             const std::string& schema_name) {
    const auto metadata = fn.metadata();

    // Both are IPC-serialized *schemas*, not batches: a parameter list is
    // carried as fields plus metadata, and the output schema is what lets the
    // engine type the call site before any bind happens.
    const auto arguments = build_arg_schema(fn.argument_specs());
    const auto output = build_scalar_output_schema(metadata.return_type);

    return wire::encode_ipc(
        wire::ResultBuilder(gen::FunctionInfoSchema())
            .set_string("name", fn.name())
            .set_string("schema_name", schema_name)
            .set_enum("function_type", enums::function_type::kScalar)
            .set_binary("arguments", wire::encode_schema(arguments))
            .set_binary("output_schema", wire::encode_schema(output))
            .set_enum("stability", stability_wire_value(metadata.stability))
            .set_enum("null_handling", metadata.null_handling == NullHandling::Special
                                           ? enums::null_handling::kSpecial
                                           : enums::null_handling::kDefault)
            .set_string("description", metadata.description)
            .set_examples("examples", metadata.examples)
            .set_string_list("categories", metadata.categories)
            .set_string_map("tags", metadata.tags)
            .set_string_list("required_settings", metadata.required_settings)
            .set_secret_lookups("required_secrets", secret_entries(metadata))
            .set_bool("projection_pushdown", metadata.projection_pushdown)
            .set_bool("filter_pushdown", metadata.filter_pushdown)
            .set_bool("sampling_pushdown", metadata.sampling_pushdown)
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", partition_kind_wire_value(metadata))
            .set_enum("order_dependent", enums::order_dependence::kNotOrderDependent)
            .set_enum("distinct_dependent", enums::distinct_dependence::kNotDistinctDependent)
            .fill_defaults()
            .finish());
}

// Normalize the engine's function-type filter to the short lowercase form.
//
// Two spellings of the same idea travel on this protocol and they are not the
// same enum: `FunctionInfo.function_type` is `scalar`, while the filter the
// engine sends to catalog_schema_contents_functions is DuckDB's own
// `SCALAR_FUNCTION`. Comparing one against the other silently matches nothing,
// which surfaces as "Scalar Function with name X does not exist" — a catalog
// error that says nothing about the enum.
//
// Empty means no filter.
std::string Dispatcher::seal_attachment(const Attachment& attachment) {
    return attachment.catalog + '\0' + attachment.data_version + '\0' + attachment.alias +
           '\0' + attachment.id;
}

Dispatcher::Attachment Dispatcher::attachment_of(
    const std::shared_ptr<arrow::RecordBatch>& batch) const {
    Attachment attachment;
    attachment.catalog = catalog().name;
    if (!batch) return attachment;

    const auto sealed = wire::get_optional_binary(batch, "attach_opaque_data");
    if (!sealed || sealed->empty()) return attachment;

    std::vector<std::string> fields;
    for (size_t start = 0; start <= sealed->size();) {
        const auto separator = sealed->find('\0', start);
        if (separator == std::string::npos) {
            fields.push_back(sealed->substr(start));
            break;
        }
        fields.push_back(sealed->substr(start, separator - start));
        start = separator + 1;
    }
    // Anything that is not the four-field seal is not ours — an older client,
    // or a catalog we do not serve — and the primary is the honest answer.
    if (fields.size() != 4 || !find_catalog(fields[0])) return attachment;

    attachment.catalog = fields[0];
    attachment.data_version = fields[1];
    attachment.alias = fields[2];
    attachment.id = fields[3];
    return attachment;
}

Dispatcher::Attachment Dispatcher::attachment_of(const vgi_rpc::Request& request) const {
    return attachment_of(request.batch());
}

bool Dispatcher::advertised_to(const Scope& scope, const vgi_rpc::Request& request) const {
    // A function declared in some other catalog is not part of this
    // attachment's surface, however the attachment was named.
    return scope.catalog == attachment_of(request).catalog;
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_functions(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "name");
    const auto filter = normalize_function_type(wire::get_enum(request.batch(), "type"));

    std::vector<std::string> items;
    // Only what is declared in this schema *and* in this attachment's catalog.
    // Advertising everything under every schema would make a two-schema
    // collision look like one flat entry with two overloads, which is what the
    // engine then reports.
    const auto mine = [&](const Scope& scope) { return advertised_to(scope, request); };
    // Hidden names are still registered — the engine calls them to scan the
    // tables they back — but they are not part of the surface a user browses.
    const auto shown = [&](const std::string& name) { return !hidden(name); };
    if (!filter || *filter == enums::function_type::kScalar) {
        for (size_t i = 0; i < scalars_.size(); ++i) {
            if (scalar_scopes_[i].schema != schema_name || !mine(scalar_scopes_[i]) ||
                !shown(scalars_[i]->name())) {
                continue;
            }
            items.push_back(encode_function_info(*scalars_[i], schema_name));
        }
    }
    if (!filter || *filter == enums::function_type::kTable) {
        for (size_t i = 0; i < tables_.size(); ++i) {
            if (table_scopes_[i].schema != schema_name || !mine(table_scopes_[i]) ||
                !shown(tables_[i]->name())) {
                continue;
            }
            items.push_back(encode_table_function_info(*tables_[i], schema_name));
        }
        for (size_t i = 0; i < table_in_outs_.size(); ++i) {
            if (table_in_out_scopes_[i].schema != schema_name ||
                !mine(table_in_out_scopes_[i]) || !shown(table_in_outs_[i]->name())) {
                continue;
            }
            items.push_back(encode_table_in_out_info(*table_in_outs_[i], schema_name));
        }
        // Buffering functions are advertised under the *table* filter, not a
        // filter of their own. The engine only ever asks for scalar, table or
        // aggregate — `table_buffering` is what the record calls itself, not
        // something the engine knows to ask for — so listing them under their
        // own name means they are never returned and never resolve.
        for (size_t i = 0; i < bufferings_.size(); ++i) {
            if (buffering_scopes_[i].schema != schema_name || !mine(buffering_scopes_[i]) ||
                !shown(bufferings_[i]->name())) {
                continue;
            }
            items.push_back(encode_buffering_info(*bufferings_[i], schema_name));
        }
    }
    if (!filter || *filter == enums::function_type::kAggregate) {
        for (size_t i = 0; i < aggregates_.size(); ++i) {
            if (aggregate_scopes_[i].schema != schema_name || !mine(aggregate_scopes_[i]) ||
                !shown(aggregates_[i]->name())) {
                continue;
            }
            items.push_back(encode_aggregate_info(*aggregates_[i], schema_name));
        }
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_schema_contents_functions"))
                        .set_binary_list("items", items)
                        .finish());
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_tables(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "name");
    std::vector<std::string> items;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& table : schema->tables) {
            items.push_back(encode_table_info(table, schema_name));
        }
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_schema_contents_tables"))
                        .set_binary_list("items", items)
                        .finish());
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_views(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "name");
    std::vector<std::string> items;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& view : schema->views) {
            items.push_back(encode_view_info(view, schema_name));
        }
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_schema_contents_views"))
                        .set_binary_list("items", items)
                        .finish());
}

vgi_rpc::Result Dispatcher::catalog_table_scan_function_get(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "schema_name");
    const auto name = wire::get_string(request.batch(), "name");

    const CatalogTable* found = nullptr;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& table : schema->tables) {
            if (table.name == name) found = &table;
        }
    }
    if (!found) throw std::invalid_argument("no table '" + schema_name + "." + name + "'");

    const auto* version =
        resolve_version(*found, wire::get_optional_string(request.batch(), "at_unit"),
                        wire::get_optional_string(request.batch(), "at_value"));

    // Returned as raw IPC bytes, not wrapped in an items list: this method's
    // return type is `bytes`, so the payload *is* the ScanFunctionResult.
    auto scan =
        wire::ResultBuilder(gen::ScanFunctionResultSchema())
            .set_string("function_name",
                        version ? version->scan_function : found->scan_function)
            .set_binary("arguments", version ? version->scan_arguments : found->scan_arguments)
            .set_string_list("required_extensions", {})
            .finish();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(scan))
                                      .finish());
}

vgi_rpc::Result Dispatcher::catalog_table_scan_branches_get(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "schema_name");
    const auto name = wire::get_string(request.batch(), "name");

    const CatalogTable* found = nullptr;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& table : schema->tables) {
            if (table.name == name) found = &table;
        }
    }
    if (!found) throw std::invalid_argument("no table '" + schema_name + "." + name + "'");

    std::vector<std::string> branches;
    if (!found->branches) {
        // The single-branch default: a table that never mentioned branches is
        // one branch, its own scan function. Answering with an empty list
        // instead would read as "this table has no sources" — which is what a
        // table that *did* declare an empty list is saying.
        auto branch = wire::ResultBuilder(gen::ScanBranchSchema())
                          .set_string("function_name", found->scan_function)
                          .set_binary("arguments", found->scan_arguments)
                          .set_bool("writable", false)
                          .fill_defaults()
                          .finish();
        branches.push_back(wire::encode_ipc(branch));
    } else {
        for (const auto& source : *found->branches) {
            auto builder = wire::ResultBuilder(gen::ScanBranchSchema());
            builder.set_string("function_name", source.function_name)
                .set_binary("arguments", source.scan_arguments)
                .set_bool("writable", source.writable);
            const auto optional = [&](const char* field,
                                      const std::optional<std::string>& value) {
                if (value) {
                    builder.set_string(field, *value);
                } else {
                    builder.set_null(field);
                }
            };
            optional("branch_filter", source.branch_filter);
            optional("source_catalog", source.source_catalog);
            optional("source_schema", source.source_schema);
            optional("source_table", source.source_table);
            branches.push_back(wire::encode_ipc(builder.fill_defaults().finish()));
        }
    }

    auto result = wire::ResultBuilder(gen::ScanBranchesResultSchema())
                      .set_binary_list("branches", branches)
                      .set_string_list("required_extensions", found->required_extensions)
                      .finish();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(result))
                                      .finish());
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_macros(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "name");
    // The engine scans the two macro kinds in separate calls, and the kind it
    // wants is in `type`. Answering with all of them on a kind-scoped request
    // registers every macro twice, once per call.
    const auto filter = normalize_function_type(
        wire::get_optional_enum(request.batch(), "type").value_or(""));

    std::vector<std::string> items;
    if (const auto* schema = schema_for(request, schema_name)) {
        for (const auto& macro : schema->macros) {
            // The kind arrives either bare or as `scalar_macro`/`table_macro`.
            const bool wanted =
                !filter || (macro.table_macro ? *filter == "table" || *filter == "table_macro"
                                              : *filter == "scalar" || *filter == "scalar_macro");
            if (wanted) items.push_back(encode_macro_info(macro, schema_name));
        }
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_schema_contents_macros"))
                        .set_binary_list("items", items)
                        .finish());
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_indexes(const vgi_rpc::Request&) {
    return empty_items("catalog_schema_contents_indexes");
}

vgi_rpc::Result Dispatcher::catalog_copy_from_formats(const vgi_rpc::Request&) {
    // Both directions ride this one method — `direction` distinguishes them —
    // which is why its name mentions only "from".
    std::vector<std::string> items;
    for (const auto& writer : copy_to_) {
        const auto metadata = writer->metadata();
        auto builder = wire::ResultBuilder(gen::CopyFromFormatInfoSchema());
        builder.set_string("format_name", writer->format())
            .set_string("handler", writer->handler_name())
            .set_string("direction", "to")
            .set_string("description", metadata.description)
            .set_bool("ordered", writer->ordered())
            .set_binary("options",
                        wire::encode_schema(build_arg_schema(writer->argument_specs())))
            .set_string_map("tags", metadata.tags);
        if (auto comment = writer->comment()) {
            builder.set_string("comment", *comment);
        } else {
            builder.set_null("comment");
        }
        items.push_back(wire::encode_ipc(builder.fill_defaults().finish()));
    }
    for (const auto& reader : copy_from_) {
        const auto metadata = reader->metadata();
        auto builder = wire::ResultBuilder(gen::CopyFromFormatInfoSchema());
        builder.set_string("format_name", reader->format())
            .set_string("handler", reader->handler_name())
            .set_string("direction", "from")
            .set_string("description", metadata.description)
            .set_bool("ordered", false)
            .set_binary("options",
                        wire::encode_schema(build_arg_schema(reader->argument_specs())))
            .set_string_map("tags", metadata.tags);
        if (auto comment = reader->comment()) {
            builder.set_string("comment", *comment);
        } else {
            builder.set_null("comment");
        }
        items.push_back(wire::encode_ipc(builder.fill_defaults().finish()));
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_copy_from_formats"))
                        .set_binary_list("items", items)
                        .finish());
}

}  // namespace vgi
