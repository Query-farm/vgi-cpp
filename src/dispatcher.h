// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <set>
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

// The framework's severity for one of ours.
vgi_rpc::LogLevel to_rpc_level(LogLevel level);

// A sink that writes a function's messages into `context`'s in-band channel.
std::function<void(LogLevel, const std::string&)> client_log_sink(
    vgi_rpc::CallContext* context);


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
    Dispatcher();

    void set_catalog(CatalogModel catalog);
    const CatalogModel& catalog() const noexcept { return *catalogs_.front(); }
    CatalogModel& catalog() noexcept { return *catalogs_.front(); }
    // A second (third, …) catalog this worker serves, created on first use.
    //
    // One process may serve several: the suite attaches two catalogs from one
    // binary and expects a call to reach the one its attachment named, which
    // nothing but the catalog identity can decide.
    CatalogModel& catalog(const std::string& name);
    const CatalogModel* find_catalog(const std::string& name) const;

    // Names kept out of the advertised function surface. See
    // `Worker::hide_function`.
    void hide_function(std::string name) { hidden_.insert(std::move(name)); }
    bool hidden(const std::string& name) const { return hidden_.count(name) != 0; }

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
    // The few handlers that need the call's own channel back to the client.
    // Kept separate rather than widening every signature: only a method that
    // actually logs has any use for it, and the rest read better without it.
    using UnaryContextHandler = vgi_rpc::Result (Dispatcher::*)(const vgi_rpc::Request&,
                                                                vgi_rpc::CallContext&);
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
    vgi_rpc::Result table_buffering_process(const vgi_rpc::Request& request,
                                            vgi_rpc::CallContext& context);
    vgi_rpc::Result table_buffering_combine(const vgi_rpc::Request& request,
                                            vgi_rpc::CallContext& context);
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
    vgi_rpc::Result catalog_transaction_begin(const vgi_rpc::Request& request);
    void catalog_transaction_commit(const vgi_rpc::Request& request);
    void catalog_transaction_rollback(const vgi_rpc::Request& request);

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
    std::vector<std::string> encode_settings(const CatalogModel& model) const;
    std::vector<std::string> encode_secret_types(const CatalogModel& model) const;
    bool supports_time_travel(const CatalogModel& model) const;
    static std::string encode_table_info(const CatalogTable& table,
                                         const std::string& schema_name,
                                         const TimeTravelVersion* version = nullptr);
    // One `SchemaInfo` entry, with the object counts the engine treats as a
    // hard guarantee: a declared zero lets it skip both the bulk listing and
    // every per-name lookup for that kind.
    // The `FunctionInfo` for each name this catalog publishes globally.
    std::vector<std::string> encode_global_functions(const CatalogModel& model) const;
    std::string encode_schema_info(const std::string& owner, const CatalogSchema& schema,
                                   const CatalogSchema* contents) const;
    static std::string encode_macro_info(const CatalogMacro& macro,
                                         const std::string& schema_name);
    static std::string encode_view_info(const CatalogView& view,
                                        const std::string& schema_name);

    // Every registration under `name`, in registration order.
    //
    // A name is not a key: `type_info` is registered five times, once per
    // argument type, and the engine chooses between them at the call site.
    // Resolution therefore happens at bind, against the types the engine
    // resolved, not at lookup.
    std::vector<std::shared_ptr<ScalarFunction>> scalars_named(const std::string& name,
                                                               const Scope& scope) const;
    // The registrations declared in `schema`, in registration order.
    std::vector<std::shared_ptr<ScalarFunction>> scalars_in_schema(const Scope& scope) const;
    std::vector<std::shared_ptr<TableFunction>> tables_in_schema(const Scope& scope) const;
    std::shared_ptr<TableFunction> find_table(const std::string& name, const Scope& scope) const;
    // With `params`, resolves overloads by argument type as scalars do.
    std::shared_ptr<TableFunction> find_table(const std::string& name, const Scope& scope,
                                              const BindParams* params) const;
    std::vector<std::shared_ptr<TableInOutFunction>> table_in_outs_in_schema(
        const Scope& scope) const;
    std::shared_ptr<TableInOutFunction> find_table_in_out(const std::string& name,
                                                          const Scope& scope) const;
    std::vector<std::shared_ptr<AggregateFunction>> aggregates_in_schema(const Scope& scope) const;
    std::shared_ptr<AggregateFunction> require_aggregate(const std::string& name,
                                                         const Scope& scope) const;
    std::vector<std::shared_ptr<TableBufferingFunction>> bufferings_in_schema(
        const Scope& scope) const;
    std::shared_ptr<TableBufferingFunction> find_buffering(const std::string& name,
                                                           const Scope& scope) const;
    std::shared_ptr<TableBufferingFunction> require_buffering(const std::string& name,
                                                              const Scope& scope) const;
    ProcessParams buffering_params(const std::shared_ptr<arrow::RecordBatch>& dto,
                                   vgi_rpc::CallContext* context) const;
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

    std::vector<ArgSpec> argument_specs_of(const std::string& name,
                                           const BindParams& params) const;

    BindParams read_bind_request(const std::shared_ptr<arrow::RecordBatch>& bind_call) const;

    // The three fields `attach_opaque_data` seals, as sealed at ATTACH.
    //
    // Sealed rather than sent as separate columns because most catalog calls
    // carry nothing else that identifies the attachment: the wire has one
    // opaque field for the worker's own use, and this is what it is for.
    struct Attachment {
        std::string catalog;
        std::string data_version;
        std::string alias;
        // Minted per ATTACH, so two sessions on the same catalog never share
        // state keyed on it. The alias cannot serve: two ATTACHes may use the
        // same one, and a user may attach the same catalog twice.
        std::string id;
    };
    static std::string seal_attachment(const Attachment& attachment);
    // The attachment a request belongs to. Falls back to the primary catalog
    // when there is no seal — the engine asks some of these questions before
    // any attachment exists.
    Attachment attachment_of(const vgi_rpc::Request& request) const;
    Attachment attachment_of(const std::shared_ptr<arrow::RecordBatch>& batch) const;

    // The schema `name`, as this attachment sees it.
    //
    // A catalog whose table set varies by data version answers from the
    // version sealed into the request's `attach_opaque_data`; every other
    // catalog answers from its declared schemas. Null when there is no such
    // schema, which is how "no such name" is spelled to the engine.
    const CatalogSchema* schema_for(const vgi_rpc::Request& request,
                                    const std::string& name) const;

    static Scope scope_of(const BindParams& params);
    static Scope scope_of(const ProcessParams& params);

    // Whether a function declared in `scope` belongs to the attachment this
    // request was made under.
    //
    // One binary may serve a second catalog — a reproducer "app" registered
    // with `register_*_in(<other catalog>, …)` — and its functions are not
    // part of the primary catalog's surface. Everything registered without an
    // explicit catalog is, whatever name the attachment used.
    bool advertised_to(const Scope& scope, const vgi_rpc::Request& request) const;

    // The first is the primary — the one a bare `register_*` and a bare
    // `catalog()` mean. Held indirectly so a reference handed out by
    // `catalog(name)` survives a later addition.
    std::vector<std::unique_ptr<CatalogModel>> catalogs_;
    std::set<std::string> hidden_;
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

};

}  // namespace vgi
