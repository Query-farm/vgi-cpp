// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/worker.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <vgi_rpc/server.h>

#include "dispatcher.h"
#include "generated/vgi_protocol_version.hpp"

namespace vgi {

namespace {
// The generated headers carry the namespace of their original consumer, the
// DuckDB extension.  Alias rather than post-process generated output.
namespace gen = ::duckdb::vgi::generated;
}  // namespace

Worker::Worker() : disp_(std::make_unique<Dispatcher>()) {}
Worker::~Worker() = default;
Worker::Worker(Worker&&) noexcept = default;
Worker& Worker::operator=(Worker&&) noexcept = default;

void Worker::set_catalog(CatalogModel catalog) {
    disp_->set_catalog(std::move(catalog));
}

void Worker::set_server_id(std::string id) { server_id_ = std::move(id); }

void Worker::register_scalar(std::shared_ptr<ScalarFunction> fn) {
    disp_->register_scalar(std::move(fn));
}

void Worker::run(int argc, char** argv) {
    vgi_rpc::ServerBuilder builder;
    builder.enable_describe("vgi")
        .protocol_version(std::string(gen::VGI_PROTOCOL_VERSION));
    if (!server_id_.empty()) builder.server_id(server_id_);
    disp_->install(builder);

    auto server = builder.build();

    // Transport from argv, matching the Rust and Python workers so one
    // wrapper script can drive any of them.
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--unix" && i + 1 < args.size()) {
            server->serve_unix(args[i + 1]);
            std::exit(0);
        }
        if (args[i] == "--http") {
            int port = 0;
            if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) {
                port = std::atoi(args[i + 1].c_str());
            }
            server->serve_http("127.0.0.1", port);
            std::exit(0);
        }
    }

    // stdout is the Arrow-IPC channel; anything a worker wants to say goes to
    // stderr or it corrupts the stream.
    server->run();
    std::exit(0);
}

}  // namespace vgi
