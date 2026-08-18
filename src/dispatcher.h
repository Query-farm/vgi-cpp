// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <vgi_rpc/server.h>
#include <vgi_rpc/stream.h>

#include "vgi/catalog.h"
#include "vgi/function.h"
#include "vgi/table_function.h"
#include "vgi/aggregate.h"
#include "vgi/buffering.h"
#include "vgi/copy_from.h"
#include "vgi/copy_to.h"
#include "vgi/table_in_out.h"

namespace vgi {

// The `init` stream's header schema (`GlobalInitResponse`). Defined in
// function_dispatch.cpp; registration needs it, and so does the handler.
const std::shared_ptr<arrow::Schema>& global_init_response_schema();

// A fresh execution id. Opaque to the engine, which only echoes it back.
std::string next_execution_id();

// Owns the function registries and the catalog identity, and turns them into
// the RPC methods a VGI engine calls.
//
// Split from Worker because the HTTP transport needs the registries after the
// server has been built, and because the registries outlive any one
// connection while a Worker is a one-shot builder.
class Dispatcher {
public:
    void set_catalog(CatalogModel catalog) { catalog_ = std::move(catalog); }
    const CatalogModel& catalog() const noexcept { return catalog_; }
    CatalogModel& catalog() noexcept { return catalog_; }

    void register_scalar(std::shared_ptr<ScalarFunction> fn);
    void register_scalar_in(std::string catalog, std::string schema,
                            std::shared_ptr<ScalarFunction> fn);
    void register_table(std::shared_ptr<TableFunction> fn);
    void register_table_in(std::string catalog, std::string schema,
                           std::shared_ptr<TableFunction> fn);
    void register_table_in_out(std::shared_ptr<TableInOutFunction> fn);
    void register_table_in_out_in(std::string catalog, std::string schema,
                                  std::shared_ptr<TableInOutFunction> fn);
    void register_aggregate(std::shared_ptr<AggregateFunction> fn);
    void register_aggregate_in(std::string catalog, std::string schema,
                               std::shared_ptr<AggregateFunction> fn);
    void register_buffering(std::shared_ptr<TableBufferingFunction> fn);
    void register_copy_to(std::shared_ptr<CopyToFunction> writer);
    void register_copy_from(std::shared_ptr<CopyFromFunction> reader);
    void register_buffering_in(std::string catalog, std::string schema,
                               std::shared_ptr<TableBufferingFunction> fn);

    // Where a registered function is declared. Every registration has exactly
    // one; the default is the catalog's own name and `main`.
    struct Scope {
        std::string catalog;
        std::string schema;
    };

    // Register every VGI method on `builder`.
    void install(vgi_rpc::ServerBuilder& builder);

    using UnaryHandler = vgi_rpc::Result (Dispatcher::*)(const vgi_rpc::Request&);
    using VoidHandler = void (Dispatcher::*)(const vgi_rpc::Request&);

    // ── Handlers ──────────────────────────────────────────────────────────
    //
    // One member per protocol method, grouped into translation units by area
    // (catalog.cpp, function.cpp, …) rather than a single file: the surface is
    // 70 methods, and a handler is much easier to review beside its siblings
    // than in a 5,000-line switch.

    vgi_rpc::Result bind(const vgi_rpc::Request& request);
    vgi_rpc::Result table_function_cardinality(const vgi_rpc::Request& request);
    vgi_rpc::Result table_function_statistics(const vgi_rpc::Request& request);
    vgi_rpc::Result table_function_dynamic_to_string(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_bind(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_update(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_combine(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_finalize(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_destructor(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_streaming_open(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_streaming_chunk(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_streaming_close(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_window_init(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_window(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_window_batch(const vgi_rpc::Request& request);
    vgi_rpc::Result aggregate_window_destructor(const vgi_rpc::Request& request);
    vgi_rpc::Result table_buffering_process(const vgi_rpc::Request& request);
    vgi_rpc::Result table_buffering_combine(const vgi_rpc::Request& request);
    vgi_rpc::Result table_buffering_destructor(const vgi_rpc::Request& request);
    vgi_rpc::Stream init(const vgi_rpc::Request& request);

    vgi_rpc::Result catalog_attach(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schemas(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_get(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_functions(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_tables(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_views(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_macros(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_schema_contents_indexes(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_copy_from_formats(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_version(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_catalogs(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_table_get(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_table_column_statistics_get(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_table_scan_function_get(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_table_scan_branches_get(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_view_get(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_macro_get(const vgi_rpc::Request& request);
    vgi_rpc::Result catalog_index_get(const vgi_rpc::Request& request);
    void catalog_detach(const vgi_rpc::Request& request);

private:
    // Serialize one Info dataclass as the IPC bytes a list<binary> column
    // carries.
    static std::string encode_function_info(const ScalarFunction& fn,
                                            const std::string& schema_name);
    static std::string encode_table_function_info(const TableFunction& fn,
                                                  const std::string& schema_name);
    static std::string encode_table_in_out_info(const TableInOutFunction& fn,
                                                const std::string& schema_name);
    static std::string encode_aggregate_info(const AggregateFunction& fn,
                                             const std::string& schema_name);
    static std::string encode_buffering_info(const TableBufferingFunction& fn,
                                             const std::string& schema_name);
    std::vector<std::string> encode_settings() const;
    std::vector<std::string> encode_secret_types() const;
    bool supports_time_travel() const;
    static std::string encode_table_info(const CatalogTable& table,
                                         const std::string& schema_name,
                                         const TimeTravelVersion* version = nullptr);
    static std::string encode_view_info(const CatalogView& view,
                                        const std::string& schema_name);

    // Every registration under `name`, in registration order.
    //
    // A name is not a key: `type_info` is registered five times, once per
    // argument type, and the engine chooses between them at the call site.
    // Resolution therefore happens at bind, against the types the engine
    // resolved, not at lookup.
    std::vector<std::shared_ptr<ScalarFunction>> scalars_named(const std::string& name) const;
    // The registrations declared in `schema`, in registration order.
    std::vector<std::shared_ptr<ScalarFunction>> scalars_in_schema(
        const std::string& schema) const;
    std::vector<std::shared_ptr<TableFunction>> tables_in_schema(
        const std::string& schema) const;
    std::shared_ptr<TableFunction> find_table(const std::string& name,
                                              const std::string& schema) const;
    // With `params`, resolves overloads by argument type as scalars do.
    std::shared_ptr<TableFunction> find_table(const std::string& name,
                                              const std::string& schema,
                                              const BindParams* params) const;
    std::vector<std::shared_ptr<TableInOutFunction>> table_in_outs_in_schema(
        const std::string& schema) const;
    std::shared_ptr<TableInOutFunction> find_table_in_out(const std::string& name,
                                                          const std::string& schema) const;
    std::vector<std::shared_ptr<AggregateFunction>> aggregates_in_schema(
        const std::string& schema) const;
    std::shared_ptr<AggregateFunction> require_aggregate(const std::string& name,
                                                         const std::string& schema) const;
    std::vector<std::shared_ptr<TableBufferingFunction>> bufferings_in_schema(
        const std::string& schema) const;
    std::shared_ptr<TableBufferingFunction> find_buffering(const std::string& name,
                                                           const std::string& schema) const;
    std::shared_ptr<TableBufferingFunction> require_buffering(const std::string& name,
                                                              const std::string& schema) const;
    ProcessParams buffering_params(const std::shared_ptr<arrow::RecordBatch>& dto) const;
    vgi_rpc::Result window_result(const std::shared_ptr<arrow::RecordBatch>& dto, bool batched);

    // The overload of `name` that matches `params`, or a clear error.
    //
    // Failing here is a user-visible error: the engine advertised this
    // function from our own discovery answer, so being unable to resolve it
    // means the two disagree about what was advertised.
    std::shared_ptr<ScalarFunction> resolve_scalar(const std::string& name,
                                                   const BindParams& params) const;
    static void check_type_bounds(const ScalarFunction& fn, const BindParams& params);
    std::vector<SecretLookup> required_secrets_of(const std::string& name,
                                                  const BindParams& params) const;
    // Reject a constant argument outside the range its spec declares.
    //
    // Declaring a range is a discovery surface *and* a contract: a caller told
    // that `count` is `[0, +inf)` should be refused at bind rather than have a
    // negative value quietly produce an empty scan.
    static void check_arg_constraints(const std::string& function_name,
                                      const std::vector<ArgSpec>& specs,
                                      const Arguments& arguments);

    BindParams read_bind_request(const std::shared_ptr<arrow::RecordBatch>& bind_call) const;

    // The schema `name`, as this attachment sees it.
    //
    // A catalog whose table set varies by data version answers from the
    // version sealed into the request's `attach_opaque_data`; every other
    // catalog answers from its declared schemas. Null when there is no such
    // schema, which is how "no such name" is spelled to the engine.
    const CatalogSchema* schema_for(const vgi_rpc::Request& request,
                                    const std::string& name) const;

    CatalogModel catalog_;
    std::vector<std::shared_ptr<ScalarFunction>> scalars_;
    // Parallel to scalars_: where each one is declared.
    std::vector<Scope> scalar_scopes_;
    // name -> indices into scalars_, in registration order.
    std::unordered_map<std::string, std::vector<size_t>> scalar_by_name_;

    std::vector<std::shared_ptr<TableFunction>> tables_;
    std::vector<Scope> table_scopes_;
    std::unordered_map<std::string, std::vector<size_t>> table_by_name_;

    std::vector<std::shared_ptr<TableInOutFunction>> table_in_outs_;
    std::vector<Scope> table_in_out_scopes_;
    std::unordered_map<std::string, std::vector<size_t>> table_in_out_by_name_;

    std::vector<std::shared_ptr<AggregateFunction>> aggregates_;
    std::vector<Scope> aggregate_scopes_;
    std::unordered_map<std::string, std::vector<size_t>> aggregate_by_name_;

    std::vector<std::shared_ptr<CopyToFunction>> copy_to_;
    std::vector<std::shared_ptr<CopyFromFunction>> copy_from_;
    std::vector<std::shared_ptr<TableBufferingFunction>> bufferings_;
    std::vector<Scope> buffering_scopes_;
    std::unordered_map<std::string, std::vector<size_t>> buffering_by_name_;

    // execution id -> group id -> serialized state.
    //
    // Held here rather than in the function because an aggregate function is
    // shared across concurrent aggregations, and each has its own groups. The
    // engine mints the execution id at bind and echoes it on every later call.
    std::unordered_map<std::string, std::map<int64_t, std::string>> aggregate_states_;

};

}  // namespace vgi
