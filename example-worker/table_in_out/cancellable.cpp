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

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
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
                vgi::ArgSpec::table("data", 1, "Input relation"), std::move(sleep)};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema) {
            throw std::invalid_argument("slow_cancellable_inout requires an input schema");
        }
        return params.input_schema;
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t sleep_ms = params.arguments.named_int64("sleep_ms").value_or(50);
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
        return {vgi::project_batch(batch, params.output_schema)};
    }
};

// `slow_cancellable(probe_path, sleep_ms := 50, count := 1000000)` — the
// producer half of the same idea: one row per batch, slowly, so a client has
// time to cancel mid-stream.
class SlowCancellable : public vgi::TableFunction {
public:
    std::string name() const override { return "slow_cancellable"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Slow producer with an on_cancel file-writing probe (test fixture)";
        md.categories = {"test"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto sleep = vgi::ArgSpec::named("sleep_ms", "int64", "Sleep per batch (ms)");
        sleep.default_value = "50";
        auto count = vgi::ArgSpec::named("count", "int64", "Total rows to produce");
        count.default_value = "1000000";
        return {vgi::ArgSpec::constant_arg("probe_path", 0, "varchar",
                                           "Path to append to when on_cancel fires"),
                std::move(sleep), std::move(count)};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.named_int64("count").value_or(1000000)),
            std::max<int64_t>(0, params.arguments.named_int64("sleep_ms").value_or(50)));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t sleep_ms)
            : schema_(std::move(schema)), remaining_(rows), sleep_ms_(sleep_ms) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            // One row per batch and a sleep between them: a query that
            // finishes before the client can send a cancel proves nothing.
            if (sleep_ms_ > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
            }
            arrow::Int64Builder builder;
            (void)builder.Append(next_++);
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            --remaining_;
            return arrow::RecordBatch::Make(schema_, 1, {array});
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t sleep_ms_;
        int64_t next_ = 0;
    };
};

}  // namespace

void register_cancellable_inout(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<SlowCancellableInOut>());
    worker.register_table(std::make_shared<SlowCancellable>());
}

}  // namespace example
