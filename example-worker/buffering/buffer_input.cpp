// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Table-buffering fixtures: sink the whole relation, then produce.

#include <algorithm>
#include <chrono>
#include <thread>
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
    enum class Failure { None, Combine, Finalize, Process, Hang };

    // `by_batch_index` asks the engine to stamp each input chunk with its
    // source index, and reassembles the input's order from it in combine.
    BufferInput(std::string name, Failure failure = Failure::None, bool by_batch_index = false)
        : name_(std::move(name)), failure_(failure), by_batch_index_(by_batch_index) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = by_batch_index_
                             ? "buffer_input variant using batch_index to reconstruct order"
                             : "Collects all input batches and emits during finalization";
        if (by_batch_index_) {
            md.categories = {"test", "ordering"};
            md.requires_input_batch_index = true;
        }
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
        if (failure_ == Failure::Hang) {
            // Slept in short spans rather than one long one so the process
            // still answers a signal promptly.
            for (;;) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (batch && batch->num_rows() > 0) {
            if (by_batch_index_) {
                // Absent here means the engine never stamped it, which is a
                // broken declaration rather than a source that has no order —
                // the engine synthesizes an index for the sources that cannot
                // supply one.
                if (!params.input_batch_index) {
                    throw std::runtime_error(
                        name_ +
                        ".process() received no batch_index; requires_input_batch_index is "
                        "not reaching the engine");
                }
                // Written to a staging namespace, because the order they
                // arrive in is exactly what has to be undone.
                params.storage->append(params.execution_id, kUnsortedNamespace, "",
                                       encode_indexed(*params.input_batch_index,
                                                      encode_batch(batch)));
            } else {
                params.storage->append(params.execution_id, kNamespace, "", encode_batch(batch));
            }
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
        if (by_batch_index_) {
            // Combine is the one call that sees every sink's output, so it is
            // the only place the input's order can be put back.
            auto staged =
                params.storage->scan(params.execution_id, kUnsortedNamespace, "", 0, SIZE_MAX);
            std::vector<std::pair<int64_t, std::string>> ordered;
            ordered.reserve(staged.size());
            for (const auto& [id, blob] : staged) {
                (void)id;
                ordered.push_back(decode_indexed(blob));
            }
            std::sort(ordered.begin(), ordered.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [index, bytes] : ordered) {
                (void)index;
                params.storage->append(params.execution_id, kNamespace, "", bytes);
            }
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
    // A little-endian index prefix, so a staged entry carries the order it
    // belongs in without a second key.
    static std::string encode_indexed(int64_t index, const std::string& payload) {
        std::string bytes(sizeof(index), '\0');
        for (size_t i = 0; i < sizeof(index); ++i) {
            bytes[i] = static_cast<char>((static_cast<uint64_t>(index) >> (8 * i)) & 0xFF);
        }
        return bytes + payload;
    }

    static std::pair<int64_t, std::string> decode_indexed(const std::string& blob) {
        if (blob.size() < sizeof(int64_t)) return {0, {}};
        uint64_t index = 0;
        for (size_t i = 0; i < sizeof(int64_t); ++i) {
            index |= static_cast<uint64_t>(static_cast<unsigned char>(blob[i])) << (8 * i);
        }
        return {static_cast<int64_t>(index), blob.substr(sizeof(int64_t))};
    }

    // Where the batches wait between arriving and being ordered.
    static constexpr const char* kUnsortedNamespace = "buf.unsorted";

    std::string name_;
    Failure failure_;
    bool by_batch_index_;
};

}  // namespace

void register_buffering(vgi::Worker& worker) {
    worker.register_buffering(std::make_shared<BufferInput>("buffer_input"));
    worker.register_buffering(std::make_shared<BufferInput>("echo_buffering"));
    worker.register_buffering(std::make_shared<BufferInput>("ordered_buffer_input"));
    worker.register_buffering(std::make_shared<BufferInput>(
        "batch_index_buffer_input", BufferInput::Failure::None, /*by_batch_index=*/true));
    worker.register_buffering(std::make_shared<BufferInput>("slow_cancellable_buffering"));
    // Never returns from process(), so a client can only get out of it by
    // cancelling — which is the whole fixture.
    worker.register_buffering(
        std::make_shared<BufferInput>("hang_on_process", BufferInput::Failure::Hang));

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
