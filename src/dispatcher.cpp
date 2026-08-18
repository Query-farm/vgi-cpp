// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "dispatcher.h"

#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#include "methods.h"

namespace vgi {

void Dispatcher::register_scalar(std::shared_ptr<ScalarFunction> fn) {
    if (!fn) throw std::invalid_argument("register_scalar: null function");
    const auto name = fn->name();
    // A duplicate is a programming error, not a runtime condition: the engine
    // resolves by name, so a second registration would silently shadow the
    // first at some later, much less obvious point.
    if (scalar_by_name_.count(name)) {
        throw std::invalid_argument("scalar function already registered: " + name);
    }
    scalar_by_name_.emplace(name, scalars_.size());
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
            // `init` is the only streaming method: an exchange whose input and
            // output schemas are settled per call by the preceding bind, so
            // they cannot be declared here.
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
