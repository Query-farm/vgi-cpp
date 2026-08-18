// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The catalog half of the protocol: what the engine asks at ATTACH and while
// resolving names.

#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>

#include <cctype>
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

    auto batch = wire::ResultBuilder(payload_schema_of("catalog_attach"))
                     // Opaque to the engine, which only stores and echoes it.
                     // The canonical worker seals a UUID plus catalog bytes; a
                     // worker with no cross-process state to protect needs
                     // only enough to identify the attachment.
                     .set_binary("attach_opaque_data", name)
                     .set_bool("supports_transactions", false)
                     .set_bool("supports_time_travel", false)
                     .set_bool("catalog_version_frozen", true)
                     .set_int64("catalog_version", 1)
                     .set_bool("attach_opaque_data_required", true)
                     .set_string("default_schema", "main")
                     .set_bool("supports_column_statistics", false)
                     .set_string("global_function_prefix", "")
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
    auto info = wire::ResultBuilder(gen::CatalogInfoSchema())
                    .set_string("name", catalog_.name)
                    .fill_defaults()
                    .finish();
    return envelope(wire::ResultBuilder(payload_schema_of("catalog_catalogs"))
                        .set_binary_list("items", {wire::encode_ipc(info)})
                        .finish());
}

// A worker with no tables answers every table lookup with "not found", which
// is an empty item list rather than an error.
//
// These have to be *answered*, not left unimplemented: the engine asks for a
// table before it will consider a function of the same name, so a worker that
// refuses the question fails every query against it — 52 tests in this suite,
// none of which are about tables.
vgi_rpc::Result Dispatcher::catalog_table_get(const vgi_rpc::Request&) {
    return empty_items("catalog_table_get");
}

vgi_rpc::Result Dispatcher::catalog_view_get(const vgi_rpc::Request&) {
    return empty_items("catalog_view_get");
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
    for (const auto& schema_name : catalog_.schemas) {
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
    for (const auto& schema_name : catalog_.schemas) {
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
            .set_enum("partition_kind", enums::partition_kind::kNotPartitioned)
            .set_enum("order_dependent", enums::order_dependence::kNotOrderDependent)
            .set_enum("distinct_dependent", enums::distinct_dependence::kNotDistinctDependent)
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

vgi_rpc::Result Dispatcher::catalog_schema_contents_tables(const vgi_rpc::Request&) {
    return empty_items("catalog_schema_contents_tables");
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_views(const vgi_rpc::Request&) {
    return empty_items("catalog_schema_contents_views");
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_macros(const vgi_rpc::Request&) {
    return empty_items("catalog_schema_contents_macros");
}

vgi_rpc::Result Dispatcher::catalog_schema_contents_indexes(const vgi_rpc::Request&) {
    return empty_items("catalog_schema_contents_indexes");
}

vgi_rpc::Result Dispatcher::catalog_copy_from_formats(const vgi_rpc::Request&) {
    return empty_items("catalog_copy_from_formats");
}

}  // namespace vgi
