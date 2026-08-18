// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The in-band logging fixture.
//
// A worker has nowhere else to put a message: stdout is the Arrow channel and
// the engine swallows stderr. What this probes is that the channel carries the
// framing the engine needs to attribute a message — level, connection,
// invocation and attachment — and not just the text.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

namespace example {
namespace {

std::shared_ptr<arrow::Schema> logging_schema() {
    static const auto schema =
        arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
    return schema;
}

// One row per batch, so a small count still produces several ticks — the
// messages bracket the stream and there has to be something between them.
class LogGen : public vgi::TableProducer {
public:
    explicit LogGen(int64_t count) : count_(count) {}

    void set_log(std::function<void(vgi::LogLevel, const std::string&)> log) override {
        log_ = std::move(log);
    }

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (index_ == 0) {
            log_(vgi::LogLevel::Info,
                 "Starting generation of " + std::to_string(count_) + " values");
        }
        if (index_ >= count_) {
            log_(vgi::LogLevel::Info, "Generation complete");
            return nullptr;
        }
        arrow::Int64Builder builder;
        (void)builder.Append(index_++);
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        return arrow::RecordBatch::Make(logging_schema(), 1, {array});
    }

private:
    int64_t count_;
    int64_t index_ = 0;
    // A no-op until the driver binds one, so the first tick's message is not a
    // special case.
    std::function<void(vgi::LogLevel, const std::string&)> log_ = [](vgi::LogLevel,
                                                                     const std::string&) {};
};

class LoggingGenerator : public vgi::TableFunction {
public:
    std::string name() const override { return "logging_generator"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Emits log messages during generation";
        md.categories = {"testing"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of values to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return logging_schema();
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<LogGen>(
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)));
    }
};

}  // namespace

void register_logging_fixtures(vgi::Worker& worker) {
    worker.register_table(std::make_shared<LoggingGenerator>());
}

}  // namespace example
