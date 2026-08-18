// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Table-buffering fixtures: sink the whole relation, then produce.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// The log namespace this fixture appends under. Scoped by execution id, so
// two concurrent queries never see each other's rows.
constexpr const char* kNamespace = "buf";

std::string encode_batch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(sink, batch->schema()).ValueOrDie();
    (void)writer->WriteRecordBatch(*batch);
    (void)writer->Close();
    return sink->Finish().ValueOrDie()->ToString();
}

std::shared_ptr<arrow::RecordBatch> decode_batch(const std::string& bytes) {
    if (bytes.empty()) return nullptr;
    auto buffer = arrow::Buffer::FromString(bytes);
    auto source = std::make_shared<arrow::io::BufferReader>(buffer);
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!reader.ok()) return nullptr;
    std::shared_ptr<arrow::RecordBatch> batch;
    (void)reader.ValueUnsafe()->ReadNext(&batch);
    return batch;
}

// Replays a scope's appended batches, one per tick.
class Replay : public vgi::TableProducer {
public:
    Replay(std::shared_ptr<vgi::FunctionStorage> storage, std::string scope,
           std::shared_ptr<arrow::Schema> output_schema)
        : storage_(std::move(storage)),
          scope_(std::move(scope)),
          output_schema_(std::move(output_schema)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        // One entry per tick rather than draining the log up front: the whole
        // point of buffering is that the relation may not fit in memory.
        auto entries = storage_->scan(scope_, kNamespace, "", after_id_, 1);
        if (entries.empty()) return nullptr;
        after_id_ = entries.front().first;
        auto batch = decode_batch(entries.front().second);
        if (!batch) return nullptr;
        // Project on the way out: the engine may have narrowed the output
        // schema after the batches were sunk.
        return output_schema_ ? vgi::project_batch(batch, output_schema_) : batch;
    }

private:
    std::shared_ptr<vgi::FunctionStorage> storage_;
    std::string scope_;
    std::shared_ptr<arrow::Schema> output_schema_;
    int64_t after_id_ = 0;
};

// `buffer_input(data)` — collects every input batch, then replays them.
//
// The simplest thing that exercises the whole sink/combine/source shape, and
// what most of the buffering tests drive. It keeps nothing in the function
// object: the engine runs the sink in several worker processes and the source
// in another, so anything remembered in memory is gone by finalize.
class BufferInput : public vgi::TableBufferingFunction {
public:
    // `failure` names the phase that should raise, for the fixtures that test
    // how the engine recovers from a worker error mid-COPY or mid-aggregation.
    enum class Failure { None, Combine, Finalize, Process };

    explicit BufferInput(std::string name, Failure failure = Failure::None)
        : name_(std::move(name)), failure_(failure) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Buffers every input batch and replays them at finalize";
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input table")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema) {
            throw std::invalid_argument(name_ + " requires an input schema");
        }
        return params.input_schema;
    }

    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>& batch) override {
        if (failure_ == Failure::Process) {
            throw std::invalid_argument("Intentional exception during process()");
        }
        if (batch && batch->num_rows() > 0) {
            params.storage->append(params.execution_id, kNamespace, "", encode_batch(batch));
        }
        // Every sink writes into the one execution-scoped log, so the state id
        // is the execution id and combine has nothing to reconcile.
        return params.execution_id;
    }

    std::vector<std::string> combine(const vgi::ProcessParams& params,
                                     const std::vector<std::string>& state_ids) override {
        (void)state_ids;
        if (failure_ == Failure::Combine) {
            throw std::invalid_argument("Intentional exception during combine()");
        }
        return {params.execution_id};
    }

    std::unique_ptr<vgi::TableProducer> finalize_producer(
        const vgi::ProcessParams& params, const std::string& finalize_state_id) override {
        if (failure_ == Failure::Finalize) {
            throw std::invalid_argument("Intentional exception during finalize()");
        }
        const auto scope = finalize_state_id.empty() ? params.execution_id : finalize_state_id;
        return std::make_unique<Replay>(params.storage, scope, params.output_schema);
    }

private:
    std::string name_;
    Failure failure_;
};

}  // namespace

void register_buffering(vgi::Worker& worker) {
    worker.register_buffering(std::make_shared<BufferInput>("buffer_input"));
    worker.register_buffering(std::make_shared<BufferInput>("echo_buffering"));
    worker.register_buffering(std::make_shared<BufferInput>("ordered_buffer_input"));
    worker.register_buffering(std::make_shared<BufferInput>("batch_index_buffer_input"));
    worker.register_buffering(std::make_shared<BufferInput>("buffer_emit_wide"));
    worker.register_buffering(std::make_shared<BufferInput>("ordered_source"));
    worker.register_buffering(std::make_shared<BufferInput>("large_state"));
    worker.register_buffering(std::make_shared<BufferInput>("slow_cancellable_buffering"));

    // Failure fixtures: each raises in one phase, so the tests can check that
    // the pool recovers rather than wedging — a later query must still work.
    worker.register_buffering(
        std::make_shared<BufferInput>("crash_on_combine", BufferInput::Failure::Combine));
    worker.register_buffering(
        std::make_shared<BufferInput>("crash_on_finalize", BufferInput::Failure::Finalize));
    worker.register_buffering(
        std::make_shared<BufferInput>("crash_on_process", BufferInput::Failure::Process));
}

}  // namespace example
