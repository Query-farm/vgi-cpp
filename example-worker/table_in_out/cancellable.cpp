// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// `slow_cancellable_inout(probe_path, data, sleep_ms := 50)` — a passthrough
// slow enough to be cancelled mid-stream.
//
// The sleep is the fixture: a query that finishes before the client can send a
// cancel proves nothing about cancellation. `probe_path` names the file the
// canonical fixture appends to from its on_cancel hook; this SDK has no such
// hook yet, so the argument is carried to keep the signature identical and
// nothing is written.

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <arrow/record_batch.h>

#include <vgi/worker.h>

namespace example {
namespace {

class SlowCancellableInOut : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "slow_cancellable_inout"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Slow table-in-out with on_cancel probe (test fixture)";
        md.categories = {"test"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto sleep = vgi::ArgSpec::named("sleep_ms", "int64", "Sleep per batch (ms)");
        sleep.default_value = "50";
        sleep.with_range(0, std::nullopt);
        return {vgi::ArgSpec::constant_arg("probe_path", 0, "varchar",
                                           "Path to append to when on_cancel fires"),
                vgi::ArgSpec::table("data", 1, "Input relation"),
                std::move(sleep)};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema) {
            throw std::invalid_argument("slow_cancellable_inout requires an input schema");
        }
        return params.input_schema;
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t sleep_ms = params.arguments.named_int64("sleep_ms").value_or(50);
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
        return {vgi::project_batch(batch, params.output_schema)};
    }
};

}  // namespace

void register_cancellable_inout(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<SlowCancellableInOut>());
}

}  // namespace example
