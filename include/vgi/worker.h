// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vgi/catalog.h"
#include "vgi/function.h"
#include "vgi/table_function.h"
#include "vgi/table_in_out.h"

namespace vgi {

class Dispatcher;

// A VGI worker: register functions, then run().
//
// `run()` parses argv and serves until the engine disconnects — stdio by
// default, `--unix <path>` for the pooled launcher, `--http` for a standalone
// server.  It does not return.
class Worker {
public:
    Worker();
    ~Worker();
    Worker(Worker&&) noexcept;
    Worker& operator=(Worker&&) noexcept;

    void set_catalog(CatalogModel catalog);
    void set_server_id(std::string id);

    // Register in the catalog's default schema (`main`).
    void register_scalar(std::shared_ptr<ScalarFunction> fn);

    // Register in a named schema of a named catalog.
    //
    // A function's identity is (catalog, schema, name), not name alone: the
    // same name may be declared in two schemas with different implementations,
    // and a schema-qualified call has to reach the right one rather than
    // resolving as an ambiguous overload. Declaring a schema here also creates
    // it if the catalog does not list it.
    void register_scalar_in(std::string catalog, std::string schema,
                            std::shared_ptr<ScalarFunction> fn);

    void register_table(std::shared_ptr<TableFunction> fn);
    void register_table_in(std::string catalog, std::string schema,
                           std::shared_ptr<TableFunction> fn);

    void register_table_in_out(std::shared_ptr<TableInOutFunction> fn);
    void register_table_in_out_in(std::string catalog, std::string schema,
                                  std::shared_ptr<TableInOutFunction> fn);

    // Serve, selecting the transport from argv.  Never returns.
    [[noreturn]] void run(int argc, char** argv);

private:
    std::unique_ptr<Dispatcher> disp_;
    std::string server_id_;
};

}  // namespace vgi
