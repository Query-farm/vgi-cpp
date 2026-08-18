// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Table fixtures beyond the plain sequence: projection pushdown, constant
// columns, and a fixed large table.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

vgi::FunctionMetadata generator_metadata(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = {"generator", "utility"};
    return md;
}

// `projected_data(count)` — four columns, each derived differently.
//
// The whole point is projection pushdown: the producer builds only the columns
// the *bound output schema* names, so a test can tell which columns the engine
// actually asked for. Building all four and discarding would look identical
// from SQL and prove nothing.
class ProjectedData : public vgi::TableFunction {
public:
    std::string name() const override { return "projected_data"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Generates data with 4 columns, supporting projection pushdown");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("id", arrow::int64(), true),
                              arrow::field("name", arrow::utf8(), true),
                              arrow::field("value", arrow::float64(), true),
                              arrow::field("extra", arrow::int64(), true)});
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
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows)
            : schema_(std::move(schema)), remaining_(rows) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t n = std::min<int64_t>(remaining_, kBatchSize);
            const int64_t start = current_;

            std::vector<std::shared_ptr<arrow::Array>> columns;
            columns.reserve(static_cast<size_t>(schema_->num_fields()));
            for (int i = 0; i < schema_->num_fields(); ++i) {
                columns.push_back(build(schema_->field(i)->name(), start, n));
            }
            current_ += n;
            remaining_ -= n;
            return arrow::RecordBatch::Make(schema_, n, columns);
        }

    private:
        static constexpr int64_t kBatchSize = 1000;

        static std::shared_ptr<arrow::Array> build(const std::string& column, int64_t start,
                                                   int64_t n) {
            if (column == "name") {
                arrow::StringBuilder builder;
                (void)builder.Reserve(n);
                for (int64_t i = start; i < start + n; ++i) {
                    (void)builder.Append("item_" + std::to_string(i));
                }
                std::shared_ptr<arrow::Array> array;
                (void)builder.Finish(&array);
                return array;
            }
            if (column == "value") {
                arrow::DoubleBuilder builder;
                (void)builder.Reserve(n);
                for (int64_t i = start; i < start + n; ++i) {
                    (void)builder.Append(static_cast<double>(i) * 1.5);
                }
                std::shared_ptr<arrow::Array> array;
                (void)builder.Finish(&array);
                return array;
            }
            // `id` and anything unrecognized are the row index; `extra` is its
            // square, which makes a mixed-up projection visible.
            arrow::Int64Builder builder;
            (void)builder.Reserve(n);
            for (int64_t i = start; i < start + n; ++i) {
                (void)builder.Append(column == "extra" ? i * i : i);
            }
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            return array;
        }

        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t current_ = 0;
    };
};

// `ten_thousand_table()` — a fixed 10,000-row scan, for tests about volume
// rather than content.
class TenThousand : public vgi::TableFunction {
public:
    std::string name() const override { return "ten_thousand_table"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Emits exactly 10,000 rows");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override {
        return {kRows, kRows};
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema);
    }

private:
    static constexpr int64_t kRows = 10000;

    class Producer : public vgi::TableProducer {
    public:
        explicit Producer(std::shared_ptr<arrow::Schema> schema) : schema_(std::move(schema)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (emitted_ >= kRows) return nullptr;
            const int64_t n = std::min<int64_t>(kRows - emitted_, 2048);
            arrow::Int64Builder builder;
            (void)builder.Reserve(n);
            for (int64_t i = emitted_; i < emitted_ + n; ++i) (void)builder.Append(i);
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            emitted_ += n;
            return arrow::RecordBatch::Make(schema_, n, {array});
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t emitted_ = 0;
    };
};

// Repeat a one-row array `count` times, preserving its exact type — which is
// what makes the DuckDB-lossless extension types survive the round trip.
std::shared_ptr<arrow::Array> repeat_value(const std::shared_ptr<arrow::Array>& value,
                                           int64_t count) {
    arrow::UInt32Builder indices;
    (void)indices.Reserve(count);
    for (int64_t i = 0; i < count; ++i) (void)indices.Append(0);
    std::shared_ptr<arrow::Array> index_array;
    (void)indices.Finish(&index_array);

    auto taken = arrow::compute::Take(*value, *index_array);
    if (!taken.ok()) throw std::runtime_error("repeat: " + taken.status().message());
    return taken.MoveValueUnsafe();
}

// A producer that emits one prepared batch and stops.
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

// `constant_columns(count, values...)` — one column per vararg, every row the
// same value. The schema is dynamic: it comes from the arguments, so the
// column *types* are whatever the caller passed.
class ConstantColumns : public vgi::TableFunction {
public:
    std::string name() const override { return "constant_columns"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Generates rows with constant values from varargs");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto values = vgi::ArgSpec::any_column("values", 1, "Values to fill each column");
        values.with_varargs();
        values.constant = true;
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
                values};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        for (size_t i = 1; i < params.arguments.positional_count(); ++i) {
            auto value = params.arguments.positional(i);
            if (!value) continue;
            fields.push_back(arrow::field("col_" + std::to_string(i - 1), value->type(),
                                          /*nullable=*/true));
        }
        return arrow::schema(std::move(fields));
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
        const int64_t count =
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        std::vector<std::shared_ptr<arrow::Array>> columns;
        for (size_t i = 1; i < params.arguments.positional_count(); ++i) {
            if (auto value = params.arguments.positional(i)) {
                columns.push_back(repeat_value(value, count));
            }
        }
        if (columns.empty()) return std::make_unique<OneShot>(nullptr);
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, count, columns));
    }
};

// `repeat_value(count, values...)` — like constant_columns, but the columns
// are named v0, v1, … and the element type is fixed per overload.
class RepeatValue : public vgi::TableFunction {
public:
    RepeatValue(std::string type_name, std::shared_ptr<arrow::DataType> type)
        : type_name_(std::move(type_name)), type_(std::move(type)) {}

    std::string name() const override { return "repeat_value"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Repeat values for N rows");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto values =
            vgi::ArgSpec::constant_arg("values", 1, type_name_, "Values to repeat");
        values.with_varargs();
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows"), values};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        for (size_t i = 1; i < params.arguments.positional_count(); ++i) {
            fields.push_back(
                arrow::field("v" + std::to_string(i - 1), type_, /*nullable=*/true));
        }
        return arrow::schema(std::move(fields));
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const int64_t count =
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        std::vector<std::shared_ptr<arrow::Array>> columns;
        for (size_t i = 1; i < params.arguments.positional_count(); ++i) {
            auto value = params.arguments.positional(i);
            if (!value) continue;
            columns.push_back(repeat_value(cast_to(value, type_), count));
        }
        if (columns.empty()) return std::make_unique<OneShot>(nullptr);
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, count, columns));
    }

private:
    std::string type_name_;
    std::shared_ptr<arrow::DataType> type_;
};

}  // namespace

void register_more_tables(vgi::Worker& worker) {
    worker.register_table(std::make_shared<ConstantColumns>());
    worker.register_table(std::make_shared<RepeatValue>("int64", arrow::int64()));
    worker.register_table(std::make_shared<RepeatValue>("varchar", arrow::utf8()));
    worker.register_table(std::make_shared<ProjectedData>());
    worker.register_table(std::make_shared<TenThousand>());
}

}  // namespace example
