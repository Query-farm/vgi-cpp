// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "dispatcher.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#include "methods.h"

namespace vgi {

namespace {

// Re-declare a handler's answer under the envelope its method is registered
// with.
//
// Handlers build in the nullable `envelope_schema()`, while a method whose
// return is not optional declares `result` non-nullable -- and an Arrow IPC
// writer refuses a batch whose schema differs from its stream's, nullability
// included. Doing it here keeps that at one seam instead of at every handler.
// A null where the protocol says there cannot be one is the handler's bug, and
// is reported as such rather than written.
vgi_rpc::Result conform(vgi_rpc::Result result, const std::shared_ptr<arrow::Schema>& declared,
                        const std::string& method) {
    auto answer = result.annotated_batch();
    if (!answer.batch || answer.batch->schema()->Equals(*declared) ||
        !answer.batch->schema()->Equals(*envelope_schema())) {
        return result;
    }
    for (int i = 0; i < declared->num_fields(); ++i) {
        if (!declared->field(i)->nullable() && answer.batch->column(i)->null_count() > 0) {
            throw std::runtime_error(method + " answered null, but its return is not optional");
        }
    }
    answer.batch =
        arrow::RecordBatch::Make(declared, answer.batch->num_rows(), answer.batch->columns());
    return vgi_rpc::Result::from_annotated_batch(std::move(answer));
}

}  // namespace

void Dispatcher::register_scalar(std::shared_ptr<ScalarFunction> fn) {
    register_scalar_in(catalog().name, "main", std::move(fn));
}

void Dispatcher::register_table(std::shared_ptr<TableFunction> fn) {
    register_table_in(catalog().name, "main", std::move(fn));
}

void Dispatcher::register_table_in(std::string catalog, std::string schema,
                                   std::shared_ptr<TableFunction> fn) {
    register_table_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Dispatcher::register_table_in(std::string catalog, SchemaPath schema_path,
                                   std::shared_ptr<TableFunction> fn) {
    if (!fn) throw std::invalid_argument("register_table: null function");
    // Declaring a function in a schema creates both the schema and, when the
    // name is one this worker has not served before, the catalog.
    this->catalog(catalog).schema(schema_path);
    table_by_name_[fn->name()].push_back(tables_.size());
    table_scopes_.push_back({std::move(catalog), std::move(schema_path)});
    tables_.push_back(std::move(fn));
    forget_function_listings();
}

void Dispatcher::register_table_in_out(std::shared_ptr<TableInOutFunction> fn) {
    register_table_in_out_in(catalog().name, "main", std::move(fn));
}

void Dispatcher::register_table_in_out_in(std::string catalog, std::string schema,
                                          std::shared_ptr<TableInOutFunction> fn) {
    register_table_in_out_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Dispatcher::register_table_in_out_in(std::string catalog, SchemaPath schema_path,
                                          std::shared_ptr<TableInOutFunction> fn) {
    if (!fn) throw std::invalid_argument("register_table_in_out: null function");
    // Declaring a function in a schema creates both the schema and, when the
    // name is one this worker has not served before, the catalog.
    this->catalog(catalog).schema(schema_path);
    table_in_out_by_name_[fn->name()].push_back(table_in_outs_.size());
    table_in_out_scopes_.push_back({std::move(catalog), std::move(schema_path)});
    table_in_outs_.push_back(std::move(fn));
    forget_function_listings();
}

void Dispatcher::register_buffering(std::shared_ptr<TableBufferingFunction> fn) {
    register_buffering_in(catalog().name, "main", std::move(fn));
}

void Dispatcher::register_buffering_in(std::string catalog, std::string schema,
                                       std::shared_ptr<TableBufferingFunction> fn) {
    register_buffering_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Dispatcher::register_buffering_in(std::string catalog, SchemaPath schema_path,
                                       std::shared_ptr<TableBufferingFunction> fn) {
    if (!fn) throw std::invalid_argument("register_buffering: null function");
    // Declaring a function in a schema creates both the schema and, when the
    // name is one this worker has not served before, the catalog.
    this->catalog(catalog).schema(schema_path);
    buffering_by_name_[fn->name()].push_back(bufferings_.size());
    buffering_scopes_.push_back({std::move(catalog), std::move(schema_path)});
    bufferings_.push_back(std::move(fn));
    forget_function_listings();
}

void Dispatcher::register_aggregate(std::shared_ptr<AggregateFunction> fn) {
    register_aggregate_in(catalog().name, "main", std::move(fn));
}

void Dispatcher::register_aggregate_in(std::string catalog, std::string schema,
                                       std::shared_ptr<AggregateFunction> fn) {
    register_aggregate_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Dispatcher::register_aggregate_in(std::string catalog, SchemaPath schema_path,
                                       std::shared_ptr<AggregateFunction> fn) {
    if (!fn) throw std::invalid_argument("register_aggregate: null function");
    // Declaring a function in a schema creates both the schema and, when the
    // name is one this worker has not served before, the catalog.
    this->catalog(catalog).schema(schema_path);
    aggregate_by_name_[fn->name()].push_back(aggregates_.size());
    aggregate_scopes_.push_back({std::move(catalog), std::move(schema_path)});
    aggregates_.push_back(std::move(fn));
    forget_function_listings();
}

void Dispatcher::register_scalar_in(std::string catalog, std::string schema,
                                    std::shared_ptr<ScalarFunction> fn) {
    register_scalar_in(std::move(catalog), SchemaPath{std::move(schema)}, std::move(fn));
}

void Dispatcher::register_scalar_in(std::string catalog, SchemaPath schema_path,
                                    std::shared_ptr<ScalarFunction> fn) {
    if (!fn) throw std::invalid_argument("register_scalar: null function");
    // Declaring a function in a schema creates it: a worker should not have to
    // list the schema separately and keep the two in step.
    // Declaring a function in a schema creates both the schema and, when the
    // name is one this worker has not served before, the catalog.
    this->catalog(catalog).schema(schema_path);
    // Repeating a name is not an error but an overload: the fixtures register
    // `type_info` five times, one per argument type, and the engine picks by
    // the call site's types. Each registration is advertised separately.
    scalar_by_name_[fn->name()].push_back(scalars_.size());
    scalar_scopes_.push_back({std::move(catalog), std::move(schema_path)});
    scalars_.push_back(std::move(fn));
    forget_function_listings();
}

void Dispatcher::forget_function_listings() {
    std::lock_guard<std::mutex> lock(function_listings_mutex_);
    function_listings_.clear();
}

namespace {

// VGI_TRACE=1 logs every dispatched method to stderr.
//
// Worth keeping rather than reaching for a debugger each time: the engine
// decides which methods to call from what the worker advertises, so "why was
// my function never called" is usually answered by seeing which discovery
// method the engine asked and what it did next. stderr because stdout is the
// Arrow-IPC channel.
bool tracing() {
    static const bool on = [] {
        const char* value = std::getenv("VGI_TRACE");
        return value && *value && std::string(value) != "0";
    }();
    return on;
}

void trace(const std::string& method) {
    if (tracing()) std::fprintf(stderr, "[vgi] %s\n", method.c_str());
}

}  // namespace

// The primary always exists, so `catalog()` needs no null check and a worker
// that never calls `set_catalog` still serves a default-named one.
Dispatcher::Dispatcher() {
    catalogs_.push_back(std::make_unique<CatalogModel>());
}

void Dispatcher::set_catalog(CatalogModel catalog) {
    // Replaces the primary rather than adding one: `set_catalog` names what
    // this worker principally serves, and calling it twice is a correction,
    // not a second catalog.
    *catalogs_.front() = std::move(catalog);
}

CatalogModel& Dispatcher::catalog(const std::string& name) {
    for (auto& model : catalogs_) {
        if (model->name == name) return *model;
    }
    auto model = std::make_unique<CatalogModel>();
    model->name = name;
    catalogs_.push_back(std::move(model));
    return *catalogs_.back();
}

const CatalogModel* Dispatcher::find_catalog(const std::string& name) const {
    for (const auto& model : catalogs_) {
        if (model->name == name) return model.get();
    }
    return nullptr;
}

void Dispatcher::install(vgi_rpc::ServerBuilder& builder) {
    // Every method of the protocol is registered, including the ones with no
    // implementation yet.
    //
    // Registering the whole surface up front is deliberate. A method that is
    // absent and a method that is present but unimplemented fail in very
    // different ways: the first surfaces as `method_not_implemented` from the
    // RPC layer, which the engine may treat as an optional capability the
    // worker declined, and the query then fails somewhere else entirely. The
    // second says exactly which method was reached. It also makes
    // `vgi_rpc.Reflection.v1`'s `describe` an honest inventory of the
    // protocol surface.
    // Implemented handlers, by method name. Anything absent from this map is
    // still registered — see the note above — but refuses when called.
    const std::unordered_map<std::string, UnaryHandler> unary = {
        {"bind", &Dispatcher::bind},
        {"table_function_cardinality", &Dispatcher::table_function_cardinality},
        {"table_function_statistics", &Dispatcher::table_function_statistics},
        {"table_function_dynamic_to_string", &Dispatcher::table_function_dynamic_to_string},
        {"aggregate_bind", &Dispatcher::aggregate_bind},
        {"aggregate_update", &Dispatcher::aggregate_update},
        {"aggregate_combine", &Dispatcher::aggregate_combine},
        {"aggregate_finalize", &Dispatcher::aggregate_finalize},
        {"aggregate_destructor", &Dispatcher::aggregate_destructor},
        {"aggregate_streaming_open", &Dispatcher::aggregate_streaming_open},
        {"aggregate_streaming_chunk", &Dispatcher::aggregate_streaming_chunk},
        {"aggregate_streaming_close", &Dispatcher::aggregate_streaming_close},
        {"aggregate_window_init", &Dispatcher::aggregate_window_init},
        {"aggregate_window", &Dispatcher::aggregate_window},
        {"aggregate_window_batch", &Dispatcher::aggregate_window_batch},
        {"aggregate_window_destructor", &Dispatcher::aggregate_window_destructor},
        {"table_buffering_destructor", &Dispatcher::table_buffering_destructor},
        {"catalog_attach", &Dispatcher::catalog_attach},
        {"catalog_transaction_begin", &Dispatcher::catalog_transaction_begin},
        {"catalog_version", &Dispatcher::catalog_version},
        {"catalog_catalogs", &Dispatcher::catalog_catalogs},
        {"catalog_table_get", &Dispatcher::catalog_table_get},
        {"catalog_table_column_statistics_get", &Dispatcher::catalog_table_column_statistics_get},
        {"catalog_table_scan_function_get", &Dispatcher::catalog_table_scan_function_get},
        {"catalog_table_scan_branches_get", &Dispatcher::catalog_table_scan_branches_get},
        {"catalog_view_get", &Dispatcher::catalog_view_get},
        {"catalog_macro_get", &Dispatcher::catalog_macro_get},
        {"catalog_index_get", &Dispatcher::catalog_index_get},
        {"catalog_schemas", &Dispatcher::catalog_schemas},
        {"catalog_schema_get", &Dispatcher::catalog_schema_get},
        {"catalog_schema_contents_functions", &Dispatcher::catalog_schema_contents_functions},
        {"catalog_schema_contents_tables", &Dispatcher::catalog_schema_contents_tables},
        {"catalog_schema_contents_views", &Dispatcher::catalog_schema_contents_views},
        {"catalog_schema_contents_macros", &Dispatcher::catalog_schema_contents_macros},
        {"catalog_schema_contents_indexes", &Dispatcher::catalog_schema_contents_indexes},
        {"catalog_copy_from_formats", &Dispatcher::catalog_copy_from_formats},
    };
    // The handlers that also take the call's log channel.
    const std::unordered_map<std::string, UnaryContextHandler> unary_with_context = {
        {"table_function_plan", &Dispatcher::table_function_plan},
        {"table_buffering_process", &Dispatcher::table_buffering_process},
        {"table_buffering_combine", &Dispatcher::table_buffering_combine},
    };
    const std::unordered_map<std::string, VoidHandler> voids = {
        {"catalog_detach", &Dispatcher::catalog_detach},
        {"catalog_transaction_commit", &Dispatcher::catalog_transaction_commit},
        {"catalog_transaction_rollback", &Dispatcher::catalog_transaction_rollback},
    };

    for (const auto& spec : protocol_methods()) {
        const std::string name = spec.name;
        const auto& declared = declared_envelope_schema(spec);

        if (spec.kind == MethodKind::Stream) {
            // `init` is the only streaming method, and an exchange rather than
            // a producer: the engine pushes input batches and reads one output
            // batch back for each. Its input and output schemas are settled
            // per call by the preceding bind, so the ones declared here are
            // only placeholders — the factory returns the real pair.
            builder.add_exchange(
                name, spec.params, arrow::schema({}), arrow::schema({}),
                [this, name](const vgi_rpc::Request& req, vgi_rpc::CallContext& ctx) {
                    trace(name);
                    return this->init(req, ctx);
                },
                "", global_init_response_schema());
            continue;
        }

        // Two different refusals, and the distinction is user-visible.
        //
        // A `catalog_*_create` / `_drop` / `_rename` on a worker that serves a
        // read-only catalog is not an unimplemented method — it is a DDL
        // statement against something that does not accept DDL, and the engine
        // surfaces "catalog is read-only" to the user. Reporting it as
        // unimplemented instead sends them looking for a missing feature.
        const bool is_ddl =
            name.rfind("catalog_", 0) == 0 &&
            (name.find("_create") != std::string::npos || name.find("_drop") != std::string::npos ||
             name.find("_rename") != std::string::npos || name.find("_set") != std::string::npos ||
             name.find("_add") != std::string::npos || name.find("_change") != std::string::npos);
        const auto refuse = [name, is_ddl] {
            trace(name + (is_ddl ? " (read-only)" : " (unimplemented)"));
            if (is_ddl) {
                throw std::invalid_argument("catalog is read-only: " + name + " is not supported");
            }
            throw std::runtime_error("vgi-c++ has not implemented " + name + " yet");
        };

        if (spec.kind == MethodKind::Void) {
            if (auto it = voids.find(name); it != voids.end()) {
                auto handler = it->second;
                builder.add_void(
                    name, spec.params,
                    [this, handler, name](const vgi_rpc::Request& req, vgi_rpc::CallContext&) {
                        trace(name);
                        (this->*handler)(req);
                    });
                continue;
            }
            builder.add_void(
                name, spec.params,
                [refuse](const vgi_rpc::Request&, vgi_rpc::CallContext&) { refuse(); });
        } else {
            if (auto it = unary_with_context.find(name); it != unary_with_context.end()) {
                auto handler = it->second;
                builder.add_unary(name, spec.params, declared,
                                  [this, handler, name, declared](const vgi_rpc::Request& req,
                                                                  vgi_rpc::CallContext& ctx) {
                                      trace(name);
                                      return conform((this->*handler)(req, ctx), declared, name);
                                  });
                continue;
            }
            if (auto it = unary.find(name); it != unary.end()) {
                auto handler = it->second;
                builder.add_unary(name, spec.params, declared,
                                  [this, handler, name, declared](const vgi_rpc::Request& req,
                                                                  vgi_rpc::CallContext&) {
                                      trace(name);
                                      return conform((this->*handler)(req), declared, name);
                                  });
                continue;
            }
            builder.add_unary(
                name, spec.params, declared,
                [refuse](const vgi_rpc::Request&, vgi_rpc::CallContext&) -> vgi_rpc::Result {
                    refuse();
                    return vgi_rpc::Result::void_result();  // unreachable
                });
        }
    }
}

}  // namespace vgi
