// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Same-name-in-two-schemas fixtures for the two *exchange* shapes.
//
// The scalar analogue lives in `scalar/same_name.cpp`. These two are not
// redundant with it: an exchange-mode bind reaches the worker through a
// different call site in the engine, one that has to name the owning schema on
// the bind request or the collision is unresolvable. Each implementation tags
// its rows with its own schema, so a mis-routed bind reads as the wrong tag
// rather than as a plausible answer.
//
// The buffering half tags in the *sink* phase deliberately: the sink and the
// source acquire independent connections, and tagging in finalize would leave
// the sink's resolution untested.

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Shared across the two schemas — the collision is the point.
constexpr const char* kTransformName = "test_same_name_transform";
constexpr const char* kBufferedName = "test_same_name_buffered";

// The buffering half's staging log, scoped by execution id.
constexpr const char* kNamespace = "same_name";

std::shared_ptr<arrow::Schema> tag_schema() {
    return arrow::schema({arrow::field("tag", arrow::utf8(), /*nullable=*/true)});
}

// `<schema>:<value>` for every row of the first input column.
std::shared_ptr<arrow::RecordBatch> tag_rows(const std::string& schema,
                                             const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto values = cast_to(batch->column(0), arrow::utf8());
    const auto& text = static_cast<const arrow::StringArray&>(*values);
    arrow::StringBuilder out;
    (void)out.Reserve(text.length());
    for (int64_t i = 0; i < text.length(); ++i) {
        if (text.IsNull(i)) {
            (void)out.AppendNull();
        } else {
            (void)out.Append(schema + ":" + text.GetString(i));
        }
    }
    std::shared_ptr<arrow::Array> array;
    (void)out.Finish(&array);
    return arrow::RecordBatch::Make(tag_schema(), array->length(), {array});
}

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

class SameNameTransform : public vgi::TableInOutFunction {
public:
    explicit SameNameTransform(std::string schema) : schema_(std::move(schema)) {}

    std::string name() const override { return kTransformName; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Schema-disambiguation probe; the " + schema_ + "-schema table-in-out";
        md.examples = {
            {"SELECT * FROM example." + schema_ + "." + kTransformName + "((SELECT 1 AS n))",
             "Returns '" + schema_ + ":1'", std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return tag_schema();
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams&,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return {tag_rows(schema_, batch)};
    }

private:
    std::string schema_;
};

// Replays the tagged batches the sink staged, one per tick.
class Replay : public vgi::TableProducer {
public:
    Replay(std::shared_ptr<vgi::FunctionStorage> storage, std::string scope)
        : storage_(std::move(storage)), scope_(std::move(scope)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        auto entries = storage_->scan(scope_, kNamespace, "", after_id_, 1);
        if (entries.empty()) return nullptr;
        after_id_ = entries.front().first;
        return decode_batch(entries.front().second);
    }

private:
    std::shared_ptr<vgi::FunctionStorage> storage_;
    std::string scope_;
    int64_t after_id_ = 0;
};

class SameNameBuffered : public vgi::TableBufferingFunction {
public:
    explicit SameNameBuffered(std::string schema) : schema_(std::move(schema)) {}

    std::string name() const override { return kBufferedName; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Schema-disambiguation probe; the " + schema_ + "-schema buffered function";
        md.examples = {
            {"SELECT * FROM example." + schema_ + "." + kBufferedName + "((SELECT 1 AS n))",
             "Returns '" + schema_ + ":1'", std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return tag_schema();
    }

    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>& batch) override {
        if (batch && batch->num_rows() > 0) {
            params.storage->append(params.execution_id, kNamespace, "",
                                   encode_batch(tag_rows(schema_, batch)));
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
        return std::make_unique<Replay>(params.storage, scope);
    }

private:
    std::string schema_;
};

}  // namespace

void register_same_name_exchange(vgi::Worker& worker) {
    // The primary catalog, not a hardcoded name: one binary stands in for
    // several fixtures, and these belong to whichever it is serving.
    const auto& primary = worker.catalog().name;
    worker.register_table_in_out_in(primary, "main", std::make_shared<SameNameTransform>("main"));
    worker.register_table_in_out_in(primary, "data", std::make_shared<SameNameTransform>("data"));
    worker.register_buffering_in(primary, "main", std::make_shared<SameNameBuffered>("main"));
    worker.register_buffering_in(primary, "data", std::make_shared<SameNameBuffered>("data"));
}

}  // namespace example
