// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/worker.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/compute/initialize.h>
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

CatalogModel& Worker::catalog() { return disp_->catalog(); }

void Worker::set_server_id(std::string id) { server_id_ = std::move(id); }

void Worker::register_scalar(std::shared_ptr<ScalarFunction> fn) {
    disp_->register_scalar(std::move(fn));
}

void Worker::register_scalar_in(std::string catalog, std::string schema,
                                std::shared_ptr<ScalarFunction> fn) {
    disp_->register_scalar_in(std::move(catalog), std::move(schema), std::move(fn));
}

void Worker::register_table(std::shared_ptr<TableFunction> fn) {
    disp_->register_table(std::move(fn));
}

void Worker::register_table_in(std::string catalog, std::string schema,
                               std::shared_ptr<TableFunction> fn) {
    disp_->register_table_in(std::move(catalog), std::move(schema), std::move(fn));
}

void Worker::register_copy_to(std::shared_ptr<CopyToFunction> writer) {
    disp_->register_copy_to(std::move(writer));
}

void Worker::register_copy_from(std::shared_ptr<CopyFromFunction> reader) {
    disp_->register_copy_from(std::move(reader));
}

void Worker::register_buffering(std::shared_ptr<TableBufferingFunction> fn) {
    disp_->register_buffering(std::move(fn));
}

void Worker::register_buffering_in(std::string catalog, std::string schema,
                                   std::shared_ptr<TableBufferingFunction> fn) {
    disp_->register_buffering_in(std::move(catalog), std::move(schema), std::move(fn));
}

void Worker::register_aggregate(std::shared_ptr<AggregateFunction> fn) {
    disp_->register_aggregate(std::move(fn));
}

void Worker::register_aggregate_in(std::string catalog, std::string schema,
                                   std::shared_ptr<AggregateFunction> fn) {
    disp_->register_aggregate_in(std::move(catalog), std::move(schema), std::move(fn));
}

void Worker::register_table_in_out(std::shared_ptr<TableInOutFunction> fn) {
    disp_->register_table_in_out(std::move(fn));
}

void Worker::register_table_in_out_in(std::string catalog, std::string schema,
                                      std::shared_ptr<TableInOutFunction> fn) {
    disp_->register_table_in_out_in(std::move(catalog), std::move(schema), std::move(fn));
}

void Worker::run(int argc, char** argv) {
    // Arrow's compute kernels register themselves from a translation unit
    // nothing here references, so linking statically drops it and `add`,
    // `multiply` and friends are simply absent from the registry at runtime —
    // while `cast`, which lives elsewhere, keeps working. That asymmetry makes
    // it read like a missing feature flag rather than a linker artifact.
    if (auto status = arrow::compute::Initialize(); !status.ok()) {
        throw std::runtime_error("cannot initialize Arrow compute: " + status.ToString());
    }

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
