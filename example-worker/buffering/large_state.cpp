// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// `large_state(data)` — buffer a megabyte per input batch and report the total.
//
// What this probes that the other buffering fixtures do not is *size*: a
// hundred-thousand-row input sinks ~50 MB of state, so the combine that reads
// it back and the finalize response that carries the answer both cross Arrow
// IPC's chunking threshold. The payload is deliberately opaque zero bytes —
// the assertion is on the byte count, not on anything decoded from it.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Bytes appended per process() call.
constexpr size_t kChunkBytes = 1024 * 1024;

// The sink's payload log, and the single row combine leaves for finalize.
// Two namespaces because finalize must not re-read the megabytes.
constexpr const char* kPayloadNamespace = "large";
constexpr const char* kResultNamespace = "buf";

std::string encode_batch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(sink, batch->schema()).ValueOrDie();
    (void)writer->WriteRecordBatch(*batch);
    (void)writer->Close();
    return sink->Finish().ValueOrDie()->ToString();
}

std::shared_ptr<arrow::RecordBatch> decode_batch(const std::string& bytes) {
    if (bytes.empty()) return nullptr;
    auto source = std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(bytes));
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!reader.ok()) return nullptr;
    std::shared_ptr<arrow::RecordBatch> batch;
    (void)reader.ValueUnsafe()->ReadNext(&batch);
    return batch;
}

// One row carrying `total` in every bound output column.
std::shared_ptr<arrow::RecordBatch> total_row(const std::shared_ptr<arrow::Schema>& schema,
                                              int64_t total) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(static_cast<size_t>(schema->num_fields()));
    for (int i = 0; i < schema->num_fields(); ++i) {
        arrow::Int64Builder builder;
        (void)builder.Append(total);
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        columns.push_back(cast_to(array, schema->field(i)->type()));
    }
    return arrow::RecordBatch::Make(schema, 1, columns);
}

// Hands back combine's single row, one batch then end-of-stream.
class DrainResult : public vgi::TableProducer {
public:
    DrainResult(std::shared_ptr<vgi::FunctionStorage> storage, std::string scope)
        : storage_(std::move(storage)), scope_(std::move(scope)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        auto entries = storage_->scan(scope_, kResultNamespace, "", after_id_, 1);
        if (entries.empty()) return nullptr;
        after_id_ = entries.front().first;
        return decode_batch(entries.front().second);
    }

private:
    std::shared_ptr<vgi::FunctionStorage> storage_;
    std::string scope_;
    int64_t after_id_ = 0;
};

class LargeState : public vgi::TableBufferingFunction {
public:
    std::string name() const override { return "large_state"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Buffers ~1 MB per input batch into state (IPC test)";
        md.categories = {"test", "memory"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input table")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema) {
            throw std::invalid_argument("large_state requires an input schema");
        }
        return params.input_schema;
    }

    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>&) override {
        params.storage->append(params.execution_id, kPayloadNamespace, "",
                               std::string(kChunkBytes, '\0'));
        return params.execution_id;
    }

    std::vector<std::string> combine(const vgi::ProcessParams& params,
                                     const std::vector<std::string>&) override {
        // Reduced here rather than in finalize so the megabytes are read once,
        // on the worker that combines, and only a single row travels onwards.
        int64_t total = 0;
        for (const auto& [id, payload] :
             params.storage->scan(params.execution_id, kPayloadNamespace, "", 0, SIZE_MAX)) {
            (void)id;
            total += static_cast<int64_t>(payload.size());
        }
        params.storage->append(params.execution_id, kResultNamespace, "",
                               encode_batch(total_row(params.output_schema, total)));
        return {params.execution_id};
    }

    std::unique_ptr<vgi::TableProducer> finalize_producer(
        const vgi::ProcessParams& params, const std::string& finalize_state_id) override {
        const auto scope = finalize_state_id.empty() ? params.execution_id : finalize_state_id;
        return std::make_unique<DrainResult>(params.storage, scope);
    }
};

}  // namespace

void register_large_state(vgi::Worker& worker) {
    worker.register_buffering(std::make_shared<LargeState>());
}

}  // namespace example
