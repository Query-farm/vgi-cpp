// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "dispatcher.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#include "methods.h"

namespace vgi {

void Dispatcher::register_scalar(std::shared_ptr<ScalarFunction> fn) {
    register_scalar_in(catalog_.name, "main", std::move(fn));
}

void Dispatcher::register_table(std::shared_ptr<TableFunction> fn) {
    register_table_in(catalog_.name, "main", std::move(fn));
}

void Dispatcher::register_table_in(std::string catalog, std::string schema,
                                   std::shared_ptr<TableFunction> fn) {
    if (!fn) throw std::invalid_argument("register_table: null function");
    if (std::find(catalog_.schemas.begin(), catalog_.schemas.end(), schema) ==
        catalog_.schemas.end()) {
        catalog_.schemas.push_back(schema);
    }
    table_by_name_[fn->name()].push_back(tables_.size());
    table_scopes_.push_back({std::move(catalog), std::move(schema)});
    tables_.push_back(std::move(fn));
}

void Dispatcher::register_scalar_in(std::string catalog, std::string schema,
                                    std::shared_ptr<ScalarFunction> fn) {
    if (!fn) throw std::invalid_argument("register_scalar: null function");
    // Declaring a function in a schema creates it: a worker should not have to
    // list the schema separately and keep the two in step.
    if (std::find(catalog_.schemas.begin(), catalog_.schemas.end(), schema) ==
        catalog_.schemas.end()) {
        catalog_.schemas.push_back(schema);
    }
    // Repeating a name is not an error but an overload: the fixtures register
    // `type_info` five times, one per argument type, and the engine picks by
    // the call site's types. Each registration is advertised separately.
    scalar_by_name_[fn->name()].push_back(scalars_.size());
    scalar_scopes_.push_back({std::move(catalog), std::move(schema)});
    scalars_.push_back(std::move(fn));
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
    // `__describe__` an honest inventory of the protocol surface.
    // Implemented handlers, by method name. Anything absent from this map is
    // still registered — see the note above — but refuses when called.
    const std::unordered_map<std::string, UnaryHandler> unary = {
        {"bind", &Dispatcher::bind},
        {"catalog_attach", &Dispatcher::catalog_attach},
        {"catalog_version", &Dispatcher::catalog_version},
        {"catalog_schemas", &Dispatcher::catalog_schemas},
        {"catalog_schema_get", &Dispatcher::catalog_schema_get},
        {"catalog_schema_contents_functions", &Dispatcher::catalog_schema_contents_functions},
        {"catalog_schema_contents_tables", &Dispatcher::catalog_schema_contents_tables},
        {"catalog_schema_contents_views", &Dispatcher::catalog_schema_contents_views},
        {"catalog_schema_contents_macros", &Dispatcher::catalog_schema_contents_macros},
        {"catalog_schema_contents_indexes", &Dispatcher::catalog_schema_contents_indexes},
        {"catalog_copy_from_formats", &Dispatcher::catalog_copy_from_formats},
    };
    const std::unordered_map<std::string, VoidHandler> voids = {
        {"catalog_detach", &Dispatcher::catalog_detach},
    };

    for (const auto& spec : protocol_methods()) {
        const std::string name = spec.name;

        if (spec.kind == MethodKind::Stream) {
            // `init` is the only streaming method, and an exchange rather than
            // a producer: the engine pushes input batches and reads one output
            // batch back for each. Its input and output schemas are settled
            // per call by the preceding bind, so the ones declared here are
            // only placeholders — the factory returns the real pair.
            builder.add_exchange(name, spec.params, arrow::schema({}), arrow::schema({}),
                                 [this, name](const vgi_rpc::Request& req,
                                              vgi_rpc::CallContext&) {
                                     trace(name);
                                     return this->init(req);
                                 },
                                 "", global_init_response_schema());
            continue;
        }

        // std::runtime_error, not a kinded error: "not implemented" is not
        // part of the wire contract, and a client should see it as the plain
        // RuntimeError it is rather than branch on it.
        const auto refuse = [name] {
            trace(name + " (unimplemented)");
            throw std::runtime_error("vgi-c++ has not implemented " + name + " yet");
        };

        if (spec.kind == MethodKind::Void) {
            if (auto it = voids.find(name); it != voids.end()) {
                auto handler = it->second;
                builder.add_void(name, spec.params,
                                 [this, handler, name](const vgi_rpc::Request& req,
                                                       vgi_rpc::CallContext&) {
                                     trace(name);
                                     (this->*handler)(req);
                                 });
                continue;
            }
            builder.add_void(name, spec.params,
                             [refuse](const vgi_rpc::Request&, vgi_rpc::CallContext&) {
                                 refuse();
                             });
        } else {
            if (auto it = unary.find(name); it != unary.end()) {
                auto handler = it->second;
                builder.add_unary(name, spec.params, envelope_schema(),
                                  [this, handler, name](const vgi_rpc::Request& req,
                                                        vgi_rpc::CallContext&) {
                                      trace(name);
                                      return (this->*handler)(req);
                                  });
                continue;
            }
            builder.add_unary(name, spec.params, envelope_schema(),
                              [refuse](const vgi_rpc::Request&,
                                       vgi_rpc::CallContext&) -> vgi_rpc::Result {
                                  refuse();
                                  return vgi_rpc::Result::void_result();  // unreachable
                              });
        }
    }
}

}  // namespace vgi
