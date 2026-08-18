// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Series generators: the same name at three arities, and a float variant.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

vgi::FunctionMetadata series_metadata(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = {"generator"};
    return md;
}

// Emits a precomputed run of int64 values in chunks.
class Values : public vgi::TableProducer {
public:
    Values(std::shared_ptr<arrow::Schema> schema, std::vector<int64_t> values,
           int64_t batch_size)
        : schema_(std::move(schema)), values_(std::move(values)), batch_size_(batch_size) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (offset_ >= static_cast<int64_t>(values_.size())) return nullptr;
        const int64_t end =
            std::min<int64_t>(offset_ + batch_size_, static_cast<int64_t>(values_.size()));
        arrow::Int64Builder builder;
        (void)builder.Reserve(end - offset_);
        for (int64_t i = offset_; i < end; ++i) {
            (void)builder.Append(values_[static_cast<size_t>(i)]);
        }
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        auto batch = arrow::RecordBatch::Make(schema_, end - offset_, {array});
        offset_ = end;
        return batch;
    }

private:
    std::shared_ptr<arrow::Schema> schema_;
    std::vector<int64_t> values_;
    int64_t batch_size_;
    int64_t offset_ = 0;
};

// Emits one prepared batch and stops.
class OneBatch : public vgi::TableProducer {
public:
    explicit OneBatch(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        auto batch = batch_;
        batch_ = nullptr;
        return batch;
    }

private:
    std::shared_ptr<arrow::RecordBatch> batch_;
};

// `make_series` at three arities — count, [start, stop), and [start, stop) by
// step. Three registrations of one name, resolved by argument count.
class MakeSeries : public vgi::TableFunction {
public:
    enum class Arity { Count, Range, Step };

    explicit MakeSeries(Arity arity) : arity_(arity) {}

    std::string name() const override { return "make_series"; }

    vgi::FunctionMetadata metadata() const override {
        switch (arity_) {
            case Arity::Count:
                return series_metadata("Generate integers from 0 to count-1");
            case Arity::Range:
                return series_metadata("Generate integers from start to stop-1");
            case Arity::Step:
                break;
        }
        return series_metadata("Generate integers from start to stop-1 with step");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        switch (arity_) {
            case Arity::Count: {
                auto count = vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of values");
                count.ge = 0;
                return {count};
            }
            case Arity::Range:
                return {vgi::ArgSpec::constant_arg("start", 0, "int64", "Start (inclusive)"),
                        vgi::ArgSpec::constant_arg("stop", 1, "int64", "Stop (exclusive)")};
            case Arity::Step:
                break;
        }
        auto step = vgi::ArgSpec::constant_arg("step", 2, "int64", "Step");
        step.ge = 1;
        return {vgi::ArgSpec::constant_arg("start", 0, "int64", "Start (inclusive)"),
                vgi::ArgSpec::constant_arg("stop", 1, "int64", "Stop (exclusive)"),
                step};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("value", arrow::int64(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto& args = params.arguments;
        std::vector<int64_t> values;
        switch (arity_) {
            case Arity::Count: {
                const int64_t count = args.const_int64(0).value_or(0);
                for (int64_t i = 0; i < count; ++i) values.push_back(i);
                break;
            }
            case Arity::Range: {
                for (int64_t i = args.const_int64(0).value_or(0),
                             stop = args.const_int64(1).value_or(0);
                     i < stop; ++i) {
                    values.push_back(i);
                }
                break;
            }
            case Arity::Step: {
                // Clamped to at least 1: a zero or negative step would not
                // terminate, and rejecting it at bind is the caller's problem
                // to have been told about, not a reason to hang here.
                const int64_t step = std::max<int64_t>(1, args.const_int64(2).value_or(1));
                for (int64_t i = args.const_int64(0).value_or(0),
                             stop = args.const_int64(1).value_or(0);
                     i < stop; i += step) {
                    values.push_back(i);
                }
                break;
            }
        }
        return std::make_unique<Values>(params.output_schema, std::move(values), 1024);
    }

private:
    Arity arity_;
};

// `double_sequence(count, batch_size := 1000, increment := 1.0)` — the float
// counterpart of `sequence`, so the float path through bind and the wire is
// exercised too.
class DoubleSequence : public vgi::TableFunction {
public:
    std::string name() const override { return "double_sequence"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = series_metadata("Generates a sequence of floating-point numbers from 0 to n-1");
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    // The float counterpart of `sequence`'s bounds, and the reason the two
    // exist side by side: a float bound has to arrive as a float or it will
    // not prune a float filter.
    std::optional<std::vector<vgi::ColumnStatistics>> statistics(
        const vgi::ProcessParams& params) const override {
        const auto count = params.arguments.const_int64(0);
        if (!count || *count <= 0) return std::vector<vgi::ColumnStatistics>{};
        const double increment = params.arguments.named_double("increment").value_or(1.0);

        vgi::ColumnStatistics stat;
        stat.column_name = "n";
        stat.min = vgi::StatValue::floating(0.0);
        stat.max = vgi::StatValue::floating(static_cast<double>(*count - 1) * increment);
        stat.has_null = false;
        stat.has_not_null = true;
        stat.distinct_count = *count;
        return std::vector<vgi::ColumnStatistics>{std::move(stat)};
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows"),
                vgi::ArgSpec::named("batch_size", "int64", "Batch size"),
                vgi::ArgSpec::named("increment", "float64", "Step between values")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::float64(), /*nullable=*/true)});
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
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)),
            std::max<int64_t>(1, params.arguments.named_int64("batch_size").value_or(1000)),
            params.arguments.named_double("increment").value_or(1.0));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size,
                 double increment)
            : schema_(std::move(schema)),
              remaining_(rows),
              batch_size_(batch_size),
              increment_(increment) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t n = std::min(remaining_, batch_size_);
            arrow::DoubleBuilder builder;
            (void)builder.Reserve(n);
            for (int64_t i = index_; i < index_ + n; ++i) {
                (void)builder.Append(static_cast<double>(i) * increment_);
            }
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            index_ += n;
            remaining_ -= n;
            return arrow::RecordBatch::Make(schema_, n, {array});
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        double increment_;
        int64_t index_ = 0;
    };
};

// `make_series(csv)` and `make_series(step)` — same arity as `make_series(count)`,
// resolved by the argument's *type* rather than by how many there are.
class MakeSeriesCsv : public vgi::TableFunction {
public:
    std::string name() const override { return "make_series"; }

    vgi::FunctionMetadata metadata() const override {
        return series_metadata("Parse comma-separated integers into rows");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("values", 0, "varchar", "Comma-separated integers")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("value", arrow::int64(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        std::vector<int64_t> values;
        const auto csv = params.arguments.const_string(0).value_or("");
        for (size_t start = 0; start <= csv.size();) {
            const auto comma = csv.find(',', start);
            const auto field = csv.substr(start, comma == std::string::npos ? std::string::npos
                                                                            : comma - start);
            // Unparseable fields are skipped rather than failing the scan:
            // this is a formatting fixture, not a validator.
            try {
                values.push_back(std::stoll(field));
            } catch (const std::exception&) {
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        return std::make_unique<Values>(params.output_schema, std::move(values), 1024);
    }
};

class MakeSeriesFloat : public vgi::TableFunction {
public:
    std::string name() const override { return "make_series"; }

    vgi::FunctionMetadata metadata() const override {
        return series_metadata("Generate 10 float values with given step size");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("step", 0, "float64", "Step size between values")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("value", arrow::float64(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const double step = params.arguments.const_double(0).value_or(1.0);
        arrow::DoubleBuilder builder;
        (void)builder.Reserve(kValues);
        for (int64_t i = 0; i < kValues; ++i) (void)builder.Append(static_cast<double>(i) * step);
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        return std::make_unique<OneBatch>(
            arrow::RecordBatch::Make(params.output_schema, kValues, {array}));
    }

private:
    static constexpr int64_t kValues = 10;
};

// `make_pairs` at one arity and three type shapes: (int, int), (varchar,
// varchar), and the mixed (int, varchar).
class MakePairs : public vgi::TableFunction {
public:
    enum class Shape { Int, Str, IntStr };

    explicit MakePairs(Shape shape) : shape_(shape) {}

    std::string name() const override { return "make_pairs"; }

    vgi::FunctionMetadata metadata() const override {
        switch (shape_) {
            case Shape::Str:
                return series_metadata("Generate string pairs with prefix and suffix");
            case Shape::IntStr:
                return series_metadata("Generate mixed int/string pairs");
            case Shape::Int: break;
        }
        return series_metadata("Generate integer pairs (i, i*2)");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        switch (shape_) {
            case Shape::Str:
                return {vgi::ArgSpec::constant_arg("prefix", 0, "varchar", "Prefix"),
                        vgi::ArgSpec::constant_arg("suffix", 1, "varchar", "Suffix")};
            case Shape::IntStr:
                return {vgi::ArgSpec::constant_arg("start", 0, "int64", "Start"),
                        vgi::ArgSpec::constant_arg("label", 1, "varchar", "Label")};
            case Shape::Int: break;
        }
        return {vgi::ArgSpec::constant_arg("start", 0, "int64", "Start"),
                vgi::ArgSpec::constant_arg("stop", 1, "int64", "Stop")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        switch (shape_) {
            case Shape::Str:
                return arrow::schema({arrow::field("a", arrow::utf8(), true),
                                      arrow::field("b", arrow::utf8(), true)});
            case Shape::IntStr:
                return arrow::schema({arrow::field("a", arrow::int64(), true),
                                      arrow::field("b", arrow::utf8(), true)});
            case Shape::Int: break;
        }
        return arrow::schema({arrow::field("a", arrow::int64(), true),
                              arrow::field("b", arrow::int64(), true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto& args = params.arguments;
        std::vector<std::shared_ptr<arrow::Array>> columns;
        int64_t rows = 0;

        if (shape_ == Shape::Int) {
            const int64_t start = args.const_int64(0).value_or(0);
            const int64_t stop = args.const_int64(1).value_or(0);
            arrow::Int64Builder first;
            arrow::Int64Builder second;
            for (int64_t i = start; i < stop; ++i, ++rows) {
                (void)first.Append(i);
                (void)second.Append(i * 2);
            }
            columns.resize(2);
            (void)first.Finish(&columns[0]);
            (void)second.Finish(&columns[1]);
        } else if (shape_ == Shape::Str) {
            const auto prefix = args.const_string(0).value_or("");
            const auto suffix = args.const_string(1).value_or("");
            arrow::StringBuilder first;
            arrow::StringBuilder second;
            for (int64_t i = 0; i < kStringRows; ++i, ++rows) {
                (void)first.Append(prefix + std::to_string(i));
                (void)second.Append(suffix + std::to_string(i));
            }
            columns.resize(2);
            (void)first.Finish(&columns[0]);
            (void)second.Finish(&columns[1]);
        } else {
            const int64_t start = args.const_int64(0).value_or(0);
            const auto label = args.const_string(1).value_or("");
            arrow::Int64Builder first;
            arrow::StringBuilder second;
            for (int64_t i = 0; i < kStringRows; ++i, ++rows) {
                (void)first.Append(start + i);
                (void)second.Append(label + std::to_string(i));
            }
            columns.resize(2);
            (void)first.Finish(&columns[0]);
            (void)second.Finish(&columns[1]);
        }

        return std::make_unique<OneBatch>(
            arrow::RecordBatch::Make(params.output_schema, rows, columns));
    }

private:
    // The string shapes take no count, so the row count is the fixture's.
    static constexpr int64_t kStringRows = 5;

    Shape shape_;
};

}  // namespace

void register_series(vgi::Worker& worker) {
    worker.register_table(std::make_shared<MakeSeries>(MakeSeries::Arity::Count));
    worker.register_table(std::make_shared<MakeSeries>(MakeSeries::Arity::Range));
    worker.register_table(std::make_shared<MakeSeries>(MakeSeries::Arity::Step));
    worker.register_table(std::make_shared<MakeSeriesCsv>());
    worker.register_table(std::make_shared<MakeSeriesFloat>());
    worker.register_table(std::make_shared<MakePairs>(MakePairs::Shape::Int));
    worker.register_table(std::make_shared<MakePairs>(MakePairs::Shape::Str));
    worker.register_table(std::make_shared<MakePairs>(MakePairs::Shape::IntStr));
    worker.register_table(std::make_shared<DoubleSequence>());
}

}  // namespace example
