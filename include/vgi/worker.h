// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vgi/catalog.h"
#include "vgi/function.h"

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

    void register_scalar(std::shared_ptr<ScalarFunction> fn);

    // Serve, selecting the transport from argv.  Never returns.
    [[noreturn]] void run(int argc, char** argv);

private:
    std::unique_ptr<Dispatcher> disp_;
    std::string server_id_;
};

}  // namespace vgi
