// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The `projection_repro` app: four scans over one wide schema, shaped like
// vgi-kafka's `kafka_consume`, that pin value correctness *across* projection
// pushdown.
//
// Three of the four deliberately ignore the narrowed output schema and emit
// all twelve columns; only `proj_repro_strict` builds what it was asked for.
// That asymmetry is the fixture: the engine has to map the columns it wanted
// onto the ones it got, and the bug this reproduces was that mapping reading
// from the wrong wire position — an all-NULL column coming back non-NULL.
//
// They live in their own catalog rather than in `example` because a naive scan
// that over-emits is not something the example catalog should advertise.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/util.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

constexpr char kCatalog[] = "projection_repro";

std::shared_ptr<arrow::DataType> header_type() {
    return arrow::list(arrow::struct_({arrow::field("k", arrow::utf8(), /*nullable=*/true),
                                       arrow::field("v", arrow::binary(), /*nullable=*/true)}));
}

std::shared_ptr<arrow::Schema> wide_schema() {
    return arrow::schema({
        arrow::field("topic", arrow::utf8(), /*nullable=*/false),
        arrow::field("partition", arrow::int32(), /*nullable=*/false),
        arrow::field("offset", arrow::int64(), /*nullable=*/false),
        arrow::field("timestamp", arrow::timestamp(arrow::TimeUnit::MILLI, "UTC"),
                     /*nullable=*/true),
        arrow::field("timestamp_type", arrow::utf8(), /*nullable=*/true),
        arrow::field("key", arrow::binary(), /*nullable=*/true),
        arrow::field("key_string", arrow::utf8(), /*nullable=*/true),
        arrow::field("key_schema_id", arrow::int32(), /*nullable=*/true),
        arrow::field("value", arrow::binary(), /*nullable=*/true),
        arrow::field("value_string", arrow::utf8(), /*nullable=*/true),
        arrow::field("value_schema_id", arrow::int32(), /*nullable=*/true),
        arrow::field("headers", header_type(), /*nullable=*/false),
    });
}

// One row of empty headers per row, built against `type` rather than against
// `header_type()`: a projection that reaches into the struct narrows the list's
// value type, and a list whose children disagree with its own type corrupts
// whoever reads it next.
std::shared_ptr<arrow::Array> empty_headers(const std::shared_ptr<arrow::DataType>& type,
                                            int64_t n) {
    arrow::Int32Builder offsets;
    (void)offsets.Reserve(n + 1);
    for (int64_t i = 0; i <= n; ++i) (void)offsets.Append(0);
    std::shared_ptr<arrow::Array> offset_array;
    (void)offsets.Finish(&offset_array);

    const auto& list = static_cast<const arrow::ListType&>(*type);
    auto values = arrow::MakeEmptyArray(list.value_type());
    if (!values.ok()) throw std::runtime_error("headers: " + values.status().message());
    auto array = arrow::ListArray::FromArrays(*offset_array, *values.ValueUnsafe());
    if (!array.ok()) throw std::runtime_error("headers: " + array.status().message());
    return array.MoveValueUnsafe();
}

std::shared_ptr<arrow::Array> build_column(const arrow::Field& field, int64_t start, int64_t n) {
    const auto& name = field.name();
    if (name == "topic" || name == "key_string" || name == "value_string") {
        arrow::StringBuilder out;
        (void)out.Reserve(n);
        for (int64_t i = start; i < start + n; ++i) {
            if (name == "topic") {
                (void)out.Append("demo_topic");
            } else {
                (void)out.Append((name == "key_string" ? "k" : "v") + std::to_string(i));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return array;
    }
    if (name == "key" || name == "value") {
        arrow::BinaryBuilder out;
        (void)out.Reserve(n);
        for (int64_t i = start; i < start + n; ++i) {
            (void)out.Append((name == "key" ? "k" : "v") + std::to_string(i));
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return array;
    }
    if (name == "partition") {
        arrow::Int32Builder out;
        (void)out.Reserve(n);
        for (int64_t i = start; i < start + n; ++i) {
            (void)out.Append(static_cast<int32_t>(i % 4));
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return array;
    }
    if (name == "offset") {
        arrow::Int64Builder out;
        (void)out.Reserve(n);
        for (int64_t i = start; i < start + n; ++i) (void)out.Append(i);
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return array;
    }
    if (name == "headers") return empty_headers(field.type(), n);

    // timestamp, timestamp_type, key_schema_id and value_schema_id are NULL for
    // every row — which is the whole point of the reproducer.
    auto nulls = arrow::MakeArrayOfNull(field.type(), n);
    if (!nulls.ok()) throw std::runtime_error(name + ": " + nulls.status().message());
    return nulls.MoveValueUnsafe();
}

std::shared_ptr<arrow::RecordBatch> wide_batch(const std::shared_ptr<arrow::Schema>& schema,
                                               int64_t start, int64_t n) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(static_cast<size_t>(schema->num_fields()));
    for (const auto& field : schema->fields()) columns.push_back(build_column(*field, start, n));
    return arrow::RecordBatch::Make(schema, n, columns);
}

class WideProducer : public vgi::TableProducer {
public:
    WideProducer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t chunk)
        : schema_(std::move(schema)), rows_(rows), chunk_(chunk) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (position_ >= rows_) return nullptr;
        const int64_t n = std::min(chunk_, rows_ - position_);
        auto batch = wide_batch(schema_, position_, n);
        position_ += n;
        return batch;
    }

private:
    std::shared_ptr<arrow::Schema> schema_;
    int64_t rows_;
    int64_t chunk_;
    int64_t position_ = 0;
};

// The four reproducer scans differ only in how much they emit per tick, across
// how many workers, and whether they honour the narrowed schema.
class ProjRepro : public vgi::TableFunction {
public:
    struct Shape {
        std::string name;
        std::string description;
        // 0 means "the whole scan in one batch".
        int64_t chunk = 0;
        int64_t workers = 1;
        // Emit exactly the columns the engine asked for, rather than all
        // twelve.
        bool strict = false;
    };

    explicit ProjRepro(Shape shape) : shape_(std::move(shape)) {}

    std::string name() const override { return shape_.name; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = shape_.description;
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("n", 0, "int64", "Number of rows to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return wide_schema();
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return shape_.workers; }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const int64_t rows = std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        return std::make_unique<WideProducer>(shape_.strict ? params.output_schema : wide_schema(),
                                              rows, shape_.chunk > 0 ? shape_.chunk : rows + 1);
    }

private:
    Shape shape_;
};

}  // namespace

void register_projection_repro(vgi::Worker& worker) {
    const auto add = [&worker](ProjRepro::Shape shape) {
        worker.register_table_in(kCatalog, "main", std::make_shared<ProjRepro>(std::move(shape)));
    };
    add({"proj_repro_strict", "projection-pushdown reproducer (strict params.output_schema)", 0, 1,
         true});
    add({"proj_repro_full_schema", "projection-pushdown reproducer (emits full FIXED_SCHEMA)"});
    add({"proj_repro_chunked",
         "projection-pushdown reproducer (multi-tick, full FIXED_SCHEMA)", 2});
    add({"proj_repro_multi_worker",
         "projection-pushdown reproducer (4 workers, multi-tick, full FIXED_SCHEMA)", 2, 4});
}

}  // namespace example
