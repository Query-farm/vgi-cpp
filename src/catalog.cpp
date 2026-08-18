// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The catalog half of the protocol: what the engine asks at ATTACH and while
// resolving names.

#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>

#include <algorithm>
#include <tuple>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string_view>

#include "arg_schema.h"
#include "dispatcher.h"
#include "enums.h"
#include "generated/vgi_protocol_schemas.hpp"
#include "methods.h"
#include "wire.h"

namespace vgi {
namespace {

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
bool Dispatcher::supports_time_travel() const {
    if (catalog_.supports_time_travel) return true;
    for (const auto& schema : catalog_.schemas) {
        for (const auto& table : schema->tables) {
            if (!table.time_travel.empty()) return true;
        }
    }
    return false;
}

std::vector<std::string> Dispatcher::encode_settings() const {
    static const auto schema = arrow::schema({
        arrow::field("name", arrow::utf8(), /*nullable=*/false),
        arrow::field("description", arrow::utf8(), /*nullable=*/false),
        arrow::field("type", arrow::binary(), /*nullable=*/false),
        arrow::field("default_value", arrow::binary(), /*nullable=*/true),
    });

    std::vector<std::string> entries;
    entries.reserve(catalog_.settings.size());
    for (const auto& setting : catalog_.settings) {
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
std::vector<std::string> Dispatcher::encode_secret_types() const {
    static const auto schema = arrow::schema({
        arrow::field("name", arrow::utf8(), /*nullable=*/false),
        arrow::field("description", arrow::utf8(), /*nullable=*/false),
        arrow::field("parameters_schema", arrow::binary(), /*nullable=*/false),
    });

    std::vector<std::string> entries;
    entries.reserve(catalog_.secret_types.size());
    for (const auto& secret : catalog_.secret_types) {
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

vgi_rpc::Result Dispatcher::catalog_attach(const vgi_rpc::Request& request) {
    // The request dataclass rides in one binary column as a self-describing
    // IPC stream; the params schema is only ever {request: binary}.
    auto attach = wire::get_ipc(request.batch(), "request");
    if (!attach) throw std::runtime_error("catalog_attach: empty request");

    // `name` is the ATTACH alias the user wrote, which need not match the
    // catalog this worker serves — the suite attaches one binary under several
    // names. It is echoed back through attach_opaque_data so later calls know
    // which attachment they belong to.
    const auto name = wire::get_string(attach, "name");

    // Version negotiation. An exact match against what this worker serves,
    // rather than a range check: a caller asking for something the worker
    // cannot serve should be told at ATTACH, not discover it mid-query.
    const auto requested_data = wire::get_optional_string(attach, "data_version_spec");
    const auto requested_impl = wire::get_optional_string(attach, "implementation_version");

    std::optional<std::string> resolved_data;
    if (!catalog_.supported_data_versions.empty()) {
        if (requested_data) {
            const auto& supported = catalog_.supported_data_versions;
            if (std::find(supported.begin(), supported.end(), *requested_data) ==
                supported.end()) {
                throw std::invalid_argument("Unsupported data_version_spec \"" +
                                            *requested_data + "\"; this worker serves one of " +
                                            join_quoted(supported));
            }
            resolved_data = requested_data;
        } else {
            resolved_data = catalog_.default_data_version;
        }
    }

    std::optional<std::string> resolved_impl;
    if (!catalog_.implementation_version.empty()) {
        const auto& supported = catalog_.supported_implementation_versions.empty()
                                    ? std::vector<std::string>{catalog_.implementation_version}
                                    : catalog_.supported_implementation_versions;
        if (requested_impl) {
            if (std::find(supported.begin(), supported.end(), *requested_impl) ==
                supported.end()) {
                throw std::invalid_argument("Unsupported implementation_version \"" +
                                            *requested_impl + "\"; this worker serves " +
                                            join_quoted(supported));
            }
            resolved_impl = requested_impl;
        } else {
            resolved_impl = catalog_.implementation_version;
        }
    }

    auto batch = wire::ResultBuilder(payload_schema_of("catalog_attach"))
                     // Opaque to the engine, which only stores and echoes it.
                     // The canonical worker seals a UUID plus catalog bytes; a
                     // worker with no cross-process state to protect needs
                     // only enough to identify the attachment.
                     .set_binary("attach_opaque_data", name)
                     .set_bool("supports_transactions", false)
                     .set_bool("supports_time_travel", supports_time_travel())
                     .set_bool("catalog_version_frozen", true)
                     .set_int64("catalog_version", 1)
                     .set_bool("attach_opaque_data_required", true)
                     .set_string("default_schema", "main")
                     .set_bool("supports_column_statistics", false)
                     .set_string("global_function_prefix", "")
                     .set_optional_string("resolved_data_version", resolved_data)
                     .set_optional_string("resolved_implementation_version", resolved_impl)
                     .set_binary_list("settings", encode_settings())
                     .set_binary_list("secret_types", encode_secret_types())
                     .fill_defaults()
                     .finish();
    return envelope(batch);
}

vgi_rpc::Result Dispatcher::catalog_version(const vgi_rpc::Request&) {
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_version"))
                                      .set_int64("version", 1)
                                      .finish());
}

void Dispatcher::catalog_detach(const vgi_rpc::Request&) {
    // Nothing is held per attachment yet. When something is, it is freed here.
}

vgi_rpc::Result Dispatcher::catalog_catalogs(const vgi_rpc::Request&) {
    // One catalog: the one this worker serves. A MetaWorker serving several
    // would list them all here.
    auto builder = wire::ResultBuilder(gen::CatalogInfoSchema());
    builder.set_string("name", catalog_.name);
    // Discovery, before any ATTACH: what this catalog *is*, not what a
    // particular attachment resolved to.
    builder.set_optional_string("implementation_version",
                                catalog_.implementation_version.empty()
                                    ? std::nullopt
                                    : std::optional<std::string>(
                                          catalog_.implementation_version));
    builder.set_optional_string("data_version_spec", catalog_.data_version_spec);
    builder.set_optional_string("source_url", catalog_.source_url.empty()
                                                  ? std::nullopt
                                                  : std::optional<std::string>(
                                                        catalog_.source_url));
    auto info = builder.fill_defaults().finish();
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_catalogs"))
                        .set_binary_list("items", {wire::encode_ipc(info)})
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
        if (has_clause && table.branches.empty()) {
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

std::string Dispatcher::encode_table_info(const CatalogTable& table,
                                          const std::string& schema_name,
                                          const TimeTravelVersion* version) {
    auto scan =
        wire::ResultBuilder(gen::ScanFunctionResultSchema())
            .set_string("function_name", version ? version->scan_function : table.scan_function)
            .set_binary("arguments", version ? version->scan_arguments : table.scan_arguments)
            .set_string_list("required_extensions", {})
            .finish();

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
        .set_bool("supports_column_statistics", false)
        .set_int64("cardinality_estimate", table.cardinality.value_or(-1))
        .set_int64("cardinality_max", table.cardinality.value_or(-1))
        .set_string_map("tags", table.tags)
        .set_string_list_list("required_filters", table.required_filters);
    if (table.comment) {
        builder.set_string("comment", *table.comment);
    } else {
        builder.set_null("comment");
    }
    return wire::encode_ipc(builder.fill_defaults().finish());
}

vgi_rpc::Result Dispatcher::catalog_table_get(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "schema_name");
    const auto name = wire::get_string(request.batch(), "name");

    const auto at_unit = wire::get_optional_string(request.batch(), "at_unit");
    const auto at_value = wire::get_optional_string(request.batch(), "at_value");

    std::vector<std::string> items;
    if (const auto* schema = catalog_.find_schema(schema_name)) {
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
    if (const auto* schema = catalog_.find_schema(schema_name)) {
        for (const auto& view : schema->views) {
            if (view.name == name) items.push_back(encode_view_info(view, schema_name));
        }
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_view_get"))
                        .set_binary_list("items", items)
                        .finish());
}

vgi_rpc::Result Dispatcher::catalog_macro_get(const vgi_rpc::Request&) {
    return empty_items("catalog_macro_get");
}

vgi_rpc::Result Dispatcher::catalog_index_get(const vgi_rpc::Request&) {
    return empty_items("catalog_index_get");
}

vgi_rpc::Result Dispatcher::catalog_schemas(const vgi_rpc::Request&) {
    std::vector<std::string> items;
    items.reserve(catalog_.schemas.size());
    for (const auto& schema_name : catalog_.schema_names()) {
        items.push_back(wire::encode_ipc(wire::ResultBuilder(gen::SchemaInfoSchema())
                                             .set_string("name", schema_name)
                                             .set_binary("attach_opaque_data", catalog_.name)
                                             .fill_defaults()
                                             .finish()));
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_schemas"))
                                      .set_binary_list("items", items)
                                      .finish());
}

vgi_rpc::Result Dispatcher::catalog_schema_get(const vgi_rpc::Request& request) {
    const auto wanted = wire::get_string(request.batch(), "name");
    std::vector<std::string> items;
    for (const auto& schema_name : catalog_.schema_names()) {
        if (schema_name != wanted) continue;
        items.push_back(wire::encode_ipc(wire::ResultBuilder(gen::SchemaInfoSchema())
                                             .set_string("name", schema_name)
                                             .set_binary("attach_opaque_data", catalog_.name)
                                             .fill_defaults()
                                             .finish()));
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
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", enums::partition_kind::kNotPartitioned)
            .set_enum("order_dependent", enums::order_dependence::kNotOrderDependent)
            .set_enum("distinct_dependent", enums::distinct_dependence::kNotDistinctDependent)
            .fill_defaults()
            .finish());
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
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", enums::partition_kind::kNotPartitioned)
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
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", enums::partition_kind::kNotPartitioned)
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
            .set_enum("partition_kind", metadata.partition_kind.empty()
                                            ? enums::partition_kind::kNotPartitioned
                                            : metadata.partition_kind.c_str())
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
            .set_bool("input_from_args", metadata.input_from_args)
            .set_enum("partition_kind", enums::partition_kind::kNotPartitioned)
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

vgi_rpc::Result Dispatcher::catalog_schema_contents_functions(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "name");
    const auto filter = normalize_function_type(wire::get_enum(request.batch(), "type"));

    std::vector<std::string> items;
    // Only what is declared in this schema. Advertising everything under every
    // schema would make a two-schema collision look like one flat entry with
    // two overloads, which is what the engine then reports.
    if (!filter || *filter == enums::function_type::kScalar) {
        for (const auto& fn : scalars_in_schema(schema_name)) {
            items.push_back(encode_function_info(*fn, schema_name));
        }
    }
    if (!filter || *filter == enums::function_type::kTable) {
        for (const auto& fn : tables_in_schema(schema_name)) {
            items.push_back(encode_table_function_info(*fn, schema_name));
        }
        for (const auto& fn : table_in_outs_in_schema(schema_name)) {
            items.push_back(encode_table_in_out_info(*fn, schema_name));
        }
        // Buffering functions are advertised under the *table* filter, not a
        // filter of their own. The engine only ever asks for scalar, table or
        // aggregate — `table_buffering` is what the record calls itself, not
        // something the engine knows to ask for — so listing them under their
        // own name means they are never returned and never resolve.
        for (const auto& fn : bufferings_in_schema(schema_name)) {
            items.push_back(encode_buffering_info(*fn, schema_name));
        }
    }
    if (!filter || *filter == enums::function_type::kAggregate) {
        for (const auto& fn : aggregates_in_schema(schema_name)) {
            items.push_back(encode_aggregate_info(*fn, schema_name));
        }
    }
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_schema_contents_functions"))
                        .set_binary_list("items", items)
                        .finish());
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_tables(const vgi_rpc::Request& request) {
    const auto schema_name = wire::get_string(request.batch(), "name");
    std::vector<std::string> items;
    if (const auto* schema = catalog_.find_schema(schema_name)) {
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
    if (const auto* schema = catalog_.find_schema(schema_name)) {
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
    if (const auto* schema = catalog_.find_schema(schema_name)) {
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
    if (const auto* schema = catalog_.find_schema(schema_name)) {
        for (const auto& table : schema->tables) {
            if (table.name == name) found = &table;
        }
    }
    if (!found) throw std::invalid_argument("no table '" + schema_name + "." + name + "'");

    std::vector<std::string> branches;
    if (found->branches.empty()) {
        // The single-branch default: a table with no declared branches is one
        // branch, its own scan function. Answering with an empty list instead
        // would read as "this table has no sources".
        auto branch = wire::ResultBuilder(gen::ScanBranchSchema())
                          .set_string("function_name", found->scan_function)
                          .set_binary("arguments", found->scan_arguments)
                          .set_bool("writable", false)
                          .fill_defaults()
                          .finish();
        branches.push_back(wire::encode_ipc(branch));
    } else {
        for (const auto& source : found->branches) {
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

vgi_rpc::Result Dispatcher::catalog_schema_contents_macros(const vgi_rpc::Request&) {
    return empty_items("catalog_schema_contents_macros");
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
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_copy_from_formats"))
                        .set_binary_list("items", items)
                        .finish());
}

}  // namespace vgi
