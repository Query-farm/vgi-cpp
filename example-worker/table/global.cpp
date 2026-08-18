// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The table and table-buffering halves of the global-registration probes.
//
// A globally published function is an alias the engine resolves back to a
// schema-resident registration, so each of these is an ordinary function that
// happens to be named in `CatalogModel::global_functions`. They tag their
// output with their own name so a test can tell that the published alias
// reached the function it was supposed to, and not some neighbour.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <vgi/worker.h>

namespace example {
namespace {

std::shared_ptr<arrow::Schema> global_table_schema() {
    static const auto schema = arrow::schema(
        {arrow::field("n", arrow::int64(), true), arrow::field("label", arrow::utf8(), true)});
    return schema;
}

class OneShot : public vgi::TableProducer {
public:
    explicit OneShot(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        auto batch = batch_;
        batch_ = nullptr;
        return batch;
    }

private:
    std::shared_ptr<arrow::RecordBatch> batch_;
};

class GlobalTable : public vgi::TableFunction {
public:
    std::string name() const override { return "global_table"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Global-registration probe (table)";
        md.categories = {"test", "global"};
        md.examples = {{"SELECT * FROM vgi_example_global_table()",
                        "Table probe published into system.main", std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return global_table_schema();
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override {
        return {kRows, kRows};
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        arrow::Int64Builder numbers;
        arrow::StringBuilder labels;
        for (int64_t i = 0; i < kRows; ++i) {
            (void)numbers.Append(i);
            (void)labels.Append("global_table:" + std::to_string(i));
        }
        std::vector<std::shared_ptr<arrow::Array>> columns(2);
        (void)numbers.Finish(&columns[0]);
        (void)labels.Finish(&columns[1]);

        // Built in the bound order, since the engine may have narrowed it.
        std::vector<std::shared_ptr<arrow::Array>> ordered;
        for (const auto& field : params.output_schema->fields()) {
            ordered.push_back(field->name() == "label" ? columns[1] : columns[0]);
        }
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, kRows, ordered));
    }

private:
    static constexpr int64_t kRows = 3;
};

// The buffering probe: an echo, since what is being tested is that the alias
// reaches a sink/source pair at all.
class GlobalBuffered : public vgi::TableBufferingFunction {
public:
    std::string name() const override { return "global_buffered"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Global-registration probe (table_buffering)";
        md.categories = {"test", "global"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input table")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema) {
            throw std::invalid_argument("global_buffered requires an input schema");
        }
        return params.input_schema;
    }

    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>& batch) override {
        if (batch && batch->num_rows() > 0) {
            params.storage->append(params.execution_id, kNamespace, "", encode(batch));
        }
        return params.execution_id;
    }

    std::vector<std::string> combine(const vgi::ProcessParams& params,
                                     const std::vector<std::string>&) override {
        return {params.execution_id};
    }

    std::unique_ptr<vgi::TableProducer> finalize_producer(
        const vgi::ProcessParams& params, const std::string& finalize_state_id) override {
        const auto scope = finalize_state_id.empty() ? params.execution_id : finalize_state_id;
        return std::make_unique<Replay>(params.storage, scope, params.output_schema);
    }

private:
    static constexpr const char* kNamespace = "global.buffered";

    static std::string encode(const std::shared_ptr<arrow::RecordBatch>& batch) {
        auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
        auto writer = arrow::ipc::MakeStreamWriter(sink, batch->schema()).ValueOrDie();
        (void)writer->WriteRecordBatch(*batch);
        (void)writer->Close();
        return sink->Finish().ValueOrDie()->ToString();
    }

    static std::shared_ptr<arrow::RecordBatch> decode(const std::string& bytes) {
        if (bytes.empty()) return nullptr;
        auto buffer = arrow::Buffer::FromString(bytes);
        auto source = std::make_shared<arrow::io::BufferReader>(buffer);
        auto reader = arrow::ipc::RecordBatchStreamReader::Open(source);
        if (!reader.ok()) return nullptr;
        std::shared_ptr<arrow::RecordBatch> batch;
        (void)reader.ValueUnsafe()->ReadNext(&batch);
        return batch;
    }

    class Replay : public vgi::TableProducer {
    public:
        Replay(std::shared_ptr<vgi::FunctionStorage> storage, std::string scope,
               std::shared_ptr<arrow::Schema> output_schema)
            : storage_(std::move(storage)),
              scope_(std::move(scope)),
              output_schema_(std::move(output_schema)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            // One entry per tick, not the whole log: the relation this
            // buffered may not fit in memory.
            auto entries = storage_->scan(scope_, kNamespace, "", after_id_, 1);
            if (entries.empty()) return nullptr;
            after_id_ = entries.front().first;
            auto batch = decode(entries.front().second);
            return batch ? vgi::project_batch(batch, output_schema_) : nullptr;
        }

    private:
        std::shared_ptr<vgi::FunctionStorage> storage_;
        std::string scope_;
        std::shared_ptr<arrow::Schema> output_schema_;
        int64_t after_id_ = 0;
    };
};

}  // namespace

void register_global_probes(vgi::Worker& worker) {
    worker.register_table(std::make_shared<GlobalTable>());
    worker.register_buffering(std::make_shared<GlobalBuffered>());
}

}  // namespace example
