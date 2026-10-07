// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "dispatcher.h"
#include "opaque_seal.h"

#include <algorithm>

#include <arrow/builder.h>
#include <vgi_rpc/arrow_utils.h>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <variant>

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

void trace_line(const std::string& line) {
    trace(line);
}

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

std::optional<std::vector<AttachOptionSpec>> Dispatcher::declared_attach_options(
    const std::string& name) const {
    if (const auto* model = find_catalog(name)) return model->attach_options;
    // A memory catalog declares no attach options.
    if (memory_catalogs_.count(name)) return std::vector<AttachOptionSpec>{};
    return std::nullopt;
}

vgi_rpc::Request Dispatcher::redeem_attach_ticket(const vgi_rpc::Request& request,
                                                  const vgi_rpc::CallContext& ctx) const {
    const auto attach = wire::get_ipc(request.batch(), "request");
    const auto restored =
        vgi::redeem_attach_ticket(attach, attach_ticket_key_, attach_ticket_principal(ctx.auth()));
    if (!restored) return request;
    // Only the method name is traced: never the ticket, never a restored option.
    trace("catalog_attach (attach ticket redeemed)");
    arrow::BinaryBuilder builder;
    VGI_RPC_THROW_NOT_OK(builder.Append(wire::encode_ipc(restored)));
    std::shared_ptr<arrow::Array> column = vgi_rpc::unwrap(builder.Finish());
    const auto& params = request.batch();
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (int i = 0; i < params->num_columns(); ++i) {
        columns.push_back(params->schema()->field(i)->name() == "request" ? column
                                                                          : params->column(i));
    }
    return vgi_rpc::Request(arrow::RecordBatch::Make(params->schema(), 1, std::move(columns)),
                            request.metadata());
}

void Dispatcher::install(vgi_rpc::ServerBuilder& builder) {
    // `vgi_attach_ticket` is the framework's: it is read as an attach ticket
    // before any catalog code runs, so a catalog declaring it could never see
    // its value. A startup error, naming the catalog.
    for (const auto& model : catalogs_) {
        for (const auto& option : model->attach_options) {
            if (is_reserved_attach_option(option.name)) {
                throw std::invalid_argument(
                    "catalog '" + model->name + "' declares attach option '" + option.name +
                    "', which uses the reserved name '" + kAttachTicketOption +
                    "': the framework reads it as an attach ticket before any catalog code "
                    "runs. Rename the option.");
            }
        }
    }

    // Every method of the protocol is registered, including the ones with no
    // implementation: the generated `VgiService` answers those UNIMPLEMENTED.
    //
    // Registering the whole surface up front is deliberate. It makes
    // `vgi_rpc.Reflection.v1`'s `describe` an honest inventory of the protocol
    // surface, and reports the reference's protocol hash: the table below is
    // generated from vgi-python's VgiProtocol, schemas and all, so no method can
    // be missing or carry a schema of its own.
    //
    // Nothing here names a method but `catalog_attach`. Every handler runs
    // behind the same boundary: the opaque values are opened, a `catalog_*`
    // call addressed to a memory catalog is that catalog's (vgi-go routes its
    // sub-catalogs the same way: by method name, then by attachment), and a
    // unary's answer is re-declared under the `result` its method is registered
    // with.
    const auto routed = [this](const std::string& name, const vgi_rpc::Request& req,
                               std::optional<vgi_rpc::Result>* out) {
        if (!route_memory_catalog(name, req, out)) return false;
        trace(name + " (memory catalog)");
        return true;
    };

    for (const auto& method : generated::VGI_METHODS) {
        const std::string name(method.name);
        const std::string doc(method.doc);

        if (const auto* stream = std::get_if<generated::VgiMethod::Stream>(&method.handler)) {
            // `init` is the only streaming method, and an exchange rather than
            // a producer: the engine pushes input batches and reads one output
            // batch back for each. Its input and output schemas are settled
            // per call by the preceding bind, so the ones declared here are
            // only placeholders — the factory returns the real pair.
            builder.add_exchange(
                name, method.params(), arrow::schema({}), arrow::schema({}),
                [this, name, serve = *stream](const vgi_rpc::Request& incoming,
                                              vgi_rpc::CallContext& ctx) {
                    opaque::Call call(opaque_key_, ctx.auth());
                    const auto req = call.open(incoming);
                    trace(name);
                    return (this->*serve)(req, ctx);
                },
                doc, method.header());
            continue;
        }

        if (const auto* void_handler = std::get_if<generated::VgiMethod::Void>(&method.handler)) {
            builder.add_void(
                name, method.params(),
                [this, name, routed, serve = *void_handler](const vgi_rpc::Request& incoming,
                                                            vgi_rpc::CallContext& ctx) {
                    opaque::Call call(opaque_key_, ctx.auth());
                    const auto req = call.open(incoming);
                    std::optional<vgi_rpc::Result> answer;
                    if (routed(name, req, &answer)) return;
                    trace(name);
                    (this->*serve)(req, ctx);
                },
                doc);
            continue;
        }

        // A `vgi_attach_ticket` is redeemed after the opaque-value boundary (it
        // seals the value this call answers with) and before routing to a
        // memory catalog or any catalog code, so the sealed catalog, not the
        // request's name, decides who serves it.
        const bool redeems_ticket = name == "catalog_attach";
        const auto& declared = method.result();
        builder.add_unary(
            name, method.params(), declared,
            [this, name, routed, redeems_ticket, declared,
             serve = std::get<generated::VgiMethod::Unary>(method.handler)](
                const vgi_rpc::Request& incoming, vgi_rpc::CallContext& ctx) {
                opaque::Call call(opaque_key_, ctx.auth());
                auto req = call.open(incoming);
                if (redeems_ticket) req = redeem_attach_ticket(req, ctx);
                std::optional<vgi_rpc::Result> answer;
                if (routed(name, req, &answer)) {
                    return conform(std::move(*answer), declared, name);
                }
                trace(name);
                return conform((this->*serve)(req, ctx), declared, name);
            },
            doc);
    }
}

// Every DDL method, refused as read-only (see dispatcher.h). The body is the
// same for all of them, hence the macro.
namespace {

[[noreturn]] void refuse_read_only(const char* method) {
    trace(std::string(method) + " (read-only)");
    throw std::invalid_argument(std::string("catalog is read-only: ") + method +
                                " is not supported");
}

}  // namespace

#define VGI_READ_ONLY(method)                                                 \
    void Dispatcher::method(const vgi_rpc::Request&, vgi_rpc::CallContext&) { \
        refuse_read_only(#method);                                            \
    }

VGI_READ_ONLY(catalog_create)
VGI_READ_ONLY(catalog_drop)
VGI_READ_ONLY(catalog_schema_create)
VGI_READ_ONLY(catalog_schema_drop)
VGI_READ_ONLY(catalog_table_create)
VGI_READ_ONLY(catalog_table_drop)
VGI_READ_ONLY(catalog_table_rename)
VGI_READ_ONLY(catalog_table_comment_set)
VGI_READ_ONLY(catalog_table_column_add)
VGI_READ_ONLY(catalog_table_column_drop)
VGI_READ_ONLY(catalog_table_column_rename)
VGI_READ_ONLY(catalog_table_column_comment_set)
VGI_READ_ONLY(catalog_table_column_default_set)
VGI_READ_ONLY(catalog_table_column_default_drop)
VGI_READ_ONLY(catalog_table_column_type_change)
VGI_READ_ONLY(catalog_table_not_null_set)
VGI_READ_ONLY(catalog_table_not_null_drop)
VGI_READ_ONLY(catalog_view_create)
VGI_READ_ONLY(catalog_view_drop)
VGI_READ_ONLY(catalog_view_rename)
VGI_READ_ONLY(catalog_view_comment_set)
VGI_READ_ONLY(catalog_macro_create)
VGI_READ_ONLY(catalog_macro_drop)
VGI_READ_ONLY(catalog_index_create)
VGI_READ_ONLY(catalog_index_drop)

#undef VGI_READ_ONLY

}  // namespace vgi
