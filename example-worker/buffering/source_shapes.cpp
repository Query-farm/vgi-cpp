// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Two fixtures that probe the *Source* half of a buffering function — the
// phase that drains what the sink collected — rather than what it collected.
//
// `ordered_source` asks in what order several finalize state ids are drained;
// `buffer_emit_wide` asks how wide a single finalize batch may be. Both ignore
// their input entirely, so nothing about the sink can colour the answer.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

namespace example {
namespace {

// Finalize state ids `ordered_source` hands back, and the value each carries.
constexpr int64_t kOrderedRows = 16;

// Big-endian so the byte order matches the numeric order, which is what makes
// a mis-sorted drain visible as out-of-order output rather than as noise.
std::string encode_index(int64_t value) {
    std::string bytes(4, '\0');
    for (int i = 0; i < 4; ++i) {
        bytes[static_cast<size_t>(i)] = static_cast<char>((value >> (8 * (3 - i))) & 0xFF);
    }
    return bytes;
}

int64_t decode_index(const std::string& bytes) {
    int64_t value = 0;
    for (const char byte : bytes) {
        value = (value << 8) | static_cast<unsigned char>(byte);
    }
    return value;
}

// One batch, then end-of-stream.
class Once : public vgi::TableProducer {
public:
    explicit Once(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        auto batch = batch_;
        batch_ = nullptr;
        return batch;
    }

private:
    std::shared_ptr<arrow::RecordBatch> batch_;
};

std::shared_ptr<arrow::RecordBatch> int64_batch(const std::shared_ptr<arrow::Schema>& schema,
                                                int64_t first, int64_t count) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(count);
    for (int64_t i = 0; i < count; ++i) (void)builder.Append(first + i);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return arrow::RecordBatch::Make(schema, count, {array});
}

// `ordered_source(data)` — sixteen finalize state ids, one row each.
//
// The input is ignored on purpose: the assertion is that the Source phase
// drains the ids in the order combine() returned them, and any dependence on
// how the sink partitioned the rows would blur that.
class OrderedSource : public vgi::TableBufferingFunction {
public:
    std::string name() const override { return "ordered_source"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Emits a fixed 0..15 sequence via source_order_dependent=True; input is ignored";
        md.categories = {"test", "ordering"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input table (rows ignored)")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("v", arrow::int64(), /*nullable=*/true)});
    }

    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>&) override {
        return params.execution_id;
    }

    std::vector<std::string> combine(const vgi::ProcessParams&,
                                     const std::vector<std::string>&) override {
        std::vector<std::string> ids;
        ids.reserve(static_cast<size_t>(kOrderedRows));
        for (int64_t i = 0; i < kOrderedRows; ++i) ids.push_back(encode_index(i));
        return ids;
    }

    std::unique_ptr<vgi::TableProducer> finalize_producer(
        const vgi::ProcessParams& params, const std::string& finalize_state_id) override {
        return std::make_unique<Once>(
            int64_batch(params.output_schema, decode_index(finalize_state_id), 1));
    }
};

// `buffer_emit_wide(rows, data)` — one finalize batch of `rows` rows.
//
// A sink fixture's batches are already capped at DuckDB's vector size, so only
// a function that *builds* its output batch can ask whether the Source path
// carries one wider than that.
class BufferEmitWide : public vgi::TableBufferingFunction {
public:
    std::string name() const override { return "buffer_emit_wide"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Emit a single finalize batch of N rows (vector-size repro)";
        md.categories = {"test", "buffer"};
        md.examples = {{"SELECT count(*) FROM buffer_emit_wide(10000, (SELECT 1))",
                        "Emit a single 10000-row batch from the Source phase", std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto rows = vgi::ArgSpec::constant_arg("rows", 0, "int64",
                                               "Number of rows to emit in one finalize batch");
        rows.with_range(0, std::nullopt);
        return {std::move(rows), vgi::ArgSpec::table("data", 1, "Input table (content ignored)")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
    }

    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>&) override {
        return params.execution_id;
    }

    std::vector<std::string> combine(const vgi::ProcessParams& params,
                                     const std::vector<std::string>&) override {
        return {params.execution_id};
    }

    std::unique_ptr<vgi::TableProducer> finalize_producer(const vgi::ProcessParams& params,
                                                          const std::string&) override {
        const int64_t rows = params.arguments.const_int64(0).value_or(0);
        return std::make_unique<Once>(int64_batch(params.output_schema, 0, rows));
    }
};

}  // namespace

void register_source_shapes(vgi::Worker& worker) {
    worker.register_buffering(std::make_shared<OrderedSource>());
    worker.register_buffering(std::make_shared<BufferEmitWide>());
}

}  // namespace example
