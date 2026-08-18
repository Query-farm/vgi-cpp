// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vgi/catalog.h"
#include "vgi/function.h"
#include "vgi/table_function.h"
#include "vgi/aggregate.h"
#include "vgi/buffering.h"
#include "vgi/copy_from.h"
#include "vgi/copy_to.h"
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
    // Mutable access, so a worker can declare tables and views after
    // registering the functions that back them.
    CatalogModel& catalog();
    // A second catalog this worker serves, created on first use.
    //
    // One process may serve several, and which one a call belongs to is
    // decided by the attachment, not by the process: two catalogs may declare
    // the same schema and the same function name.
    CatalogModel& catalog(const std::string& name);
    void set_server_id(std::string id);

    // Keep `name` out of the function surface the catalog advertises.
    //
    // A function that exists only to back a catalog table is still registered
    // — the engine calls it by name to scan the table — but it is not
    // something a user should find in `duckdb_functions()`, and a worker whose
    // surface is a cross-language contract has to be able to say so.
    void hide_function(std::string name);

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

    void register_aggregate(std::shared_ptr<AggregateFunction> fn);
    void register_aggregate_in(std::string catalog, std::string schema,
                               std::shared_ptr<AggregateFunction> fn);

    void register_buffering(std::shared_ptr<TableBufferingFunction> fn);

    // A `COPY … TO (FORMAT …)` writer. Registers its handler as a buffering
    // function too, since that is the RPC path the engine drives it over.
    void register_copy_to(std::shared_ptr<CopyToFunction> writer);

    // A `COPY … FROM (FORMAT …)` reader. Registers its handler as a table
    // function too, since that is the RPC path the engine drives it over.
    void register_copy_from(std::shared_ptr<CopyFromFunction> reader);

    void register_buffering_in(std::string catalog, std::string schema,
                               std::shared_ptr<TableBufferingFunction> fn);

    // Serve, selecting the transport from argv.  Never returns.
    [[noreturn]] void run(int argc, char** argv);

private:
    std::unique_ptr<Dispatcher> disp_;
    std::string server_id_;
};

}  // namespace vgi
