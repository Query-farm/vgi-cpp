// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Table fixtures beyond the plain sequence: projection pushdown, constant
// columns, a fixed large table, and the two that echo the optimizer hints
// DuckDB folded into the scan.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/util.h>
#include <arrow/compute/api.h>
#include <arrow/scalar.h>

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

// `ten_thousand()` — a fixed 10,000-row scan, for tests about volume rather
// than content. The `ten_thousand_table` catalog table is backed by it.
class TenThousand : public vgi::TableFunction {
public:
    std::string name() const override { return "ten_thousand"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Generates 10000 integers from 0 to 9999");
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
            // DuckDB's lossless types (HUGEINT, UUID, …) arrive as a plain
            // storage type plus an `ARROW:extension:name` entry on the field;
            // rebuilding the column from the type alone hands them back as
            // BLOBs.
            auto field = params.arguments.positional_field(i);
            fields.push_back(arrow::field("col_" + std::to_string(i - 1), value->type(),
                                          /*nullable=*/true,
                                          field ? field->metadata() : nullptr));
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

int64_t echo_count(const vgi::ProcessParams& params) {
    return std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
}

int64_t echo_batch_size(const vgi::ProcessParams& params) {
    return std::max<int64_t>(1, params.arguments.named_int64("batch_size").value_or(2048));
}

// Exact rather than estimated: the row count is the argument.
vgi::TableCardinality count_cardinality(const vgi::ProcessParams& params) {
    vgi::TableCardinality estimate;
    if (auto count = params.arguments.const_int64(0)) {
        estimate.estimate = *count;
        estimate.max = *count;
    }
    return estimate;
}

// The two pushdown-hint echo fixtures: `order_echo` and `sample_echo`.
//
// Neither one honours the hint it receives — they exist to prove the hint
// *arrived*. Echoing it into a column is the only way a SQL-level test can see
// what DuckDB's optimizers decided to fold into the scan, which is a function
// of the plan shape and not of the query text.
std::shared_ptr<arrow::Array> constant_column(const std::shared_ptr<arrow::Scalar>& value,
                                              int64_t size) {
    auto array = arrow::MakeArrayFromScalar(*value, size);
    if (!array.ok()) throw std::runtime_error("hint echo: " + array.status().message());
    return array.MoveValueUnsafe();
}

class HintEcho : public vgi::TableProducer {
public:
    using Constants = std::map<std::string, std::shared_ptr<arrow::Scalar>>;

    HintEcho(std::shared_ptr<arrow::Schema> schema, int64_t count, int64_t batch_size,
             Constants constants)
        : schema_(std::move(schema)),
          remaining_(count),
          batch_size_(batch_size),
          constants_(std::move(constants)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (remaining_ <= 0) return nullptr;
        const int64_t size = std::min(remaining_, batch_size_);

        // Driven by the bound schema, not by a fixed column list: both
        // fixtures declare projection pushdown, so the engine may have
        // narrowed the scan to any subset in any order.
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(schema_->num_fields()));
        for (const auto& field : schema_->fields()) {
            if (field->name() == "n") {
                arrow::Int64Builder builder;
                (void)builder.Reserve(size);
                for (int64_t i = 0; i < size; ++i) (void)builder.Append(next_ + i);
                std::shared_ptr<arrow::Array> array;
                (void)builder.Finish(&array);
                columns.push_back(std::move(array));
            } else if (field->name() == "s") {
                arrow::StringBuilder builder;
                for (int64_t i = 0; i < size; ++i) {
                    (void)builder.Append("row_" + std::to_string(next_ + i));
                }
                std::shared_ptr<arrow::Array> array;
                (void)builder.Finish(&array);
                columns.push_back(std::move(array));
            } else {
                auto constant = constants_.find(field->name());
                if (constant == constants_.end()) {
                    throw std::runtime_error("hint echo: unexpected column " + field->name());
                }
                columns.push_back(constant_column(constant->second, size));
            }
        }

        next_ += size;
        remaining_ -= size;
        return arrow::RecordBatch::Make(schema_, size, columns);
    }

private:
    std::shared_ptr<arrow::Schema> schema_;
    int64_t remaining_;
    int64_t batch_size_;
    Constants constants_;
    int64_t next_ = 0;
};

// The sentinel a test reads as "no hint was pushed". Absence has to be
// visible, so it is spelled rather than left null.
std::shared_ptr<arrow::Scalar> hint_text(const std::optional<std::string>& value) {
    return std::make_shared<arrow::StringScalar>(value.value_or("(none)"));
}

// `order_echo(count, batch_size := 2048)` — echoes the ORDER BY + LIMIT hint.
class OrderEcho : public vgi::TableFunction {
public:
    std::string name() const override { return "order_echo"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = generator_metadata("Echoes ORDER BY + LIMIT pushdown hints in output");
        md.categories = {"generator", "diagnostic"};
        md.projection_pushdown = true;
        // Both, and for the same test: with filters applied by the framework a
        // WHERE clause becomes a table filter on the scan, which is the case
        // where DuckDB pushes the direction but withholds the row limit.
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
            vgi::ArgSpec::named("batch_size", "int64", "Batch size for output"),
        };
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({
            arrow::field("n", arrow::int64(), true),
            arrow::field("s", arrow::utf8(), true),
            arrow::field("order_column", arrow::utf8(), true),
            arrow::field("order_direction", arrow::utf8(), true),
            arrow::field("order_null_order", arrow::utf8(), true),
            arrow::field("order_limit", arrow::int64(), true),
        });
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto& hints = params.scan_hints;
        HintEcho::Constants constants{
            {"order_column", hint_text(hints.order_by_column)},
            {"order_direction", hint_text(hints.order_by_direction)},
            {"order_null_order", hint_text(hints.order_by_null_order)},
            {"order_limit",
             std::make_shared<arrow::Int64Scalar>(hints.order_by_limit.value_or(-1))},
        };
        return std::make_unique<HintEcho>(params.output_schema, echo_count(params),
                                          echo_batch_size(params), std::move(constants));
    }
};

// `sample_echo(count, batch_size := 2048)` — echoes the TABLESAMPLE hint.
class SampleEcho : public vgi::TableFunction {
public:
    std::string name() const override { return "sample_echo"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = generator_metadata("Echoes TABLESAMPLE pushdown hints in output");
        md.categories = {"generator", "diagnostic"};
        md.projection_pushdown = true;
        // Declaring this is what makes DuckDB drop its own sampling operator.
        // The fixture then returns every row on purpose, so a test can read
        // the hint off rows that sampling would otherwise have discarded.
        md.sampling_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
            vgi::ArgSpec::named("batch_size", "int64", "Batch size for output"),
        };
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({
            arrow::field("n", arrow::int64(), true),
            arrow::field("s", arrow::utf8(), true),
            arrow::field("sample_percentage", arrow::float64(), true),
            arrow::field("sample_seed", arrow::int64(), true),
        });
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto& hints = params.scan_hints;
        HintEcho::Constants constants{
            {"sample_percentage",
             std::make_shared<arrow::DoubleScalar>(hints.tablesample_percentage.value_or(-1.0))},
            {"sample_seed",
             std::make_shared<arrow::Int64Scalar>(hints.tablesample_seed.value_or(-1))},
        };
        return std::make_unique<HintEcho>(params.output_schema, echo_count(params),
                                          echo_batch_size(params), std::move(constants));
    }
};

// `profiling_demo(count, batch_size, increment)` — a sequence that publishes
// diagnostics under EXPLAIN ANALYZE.
//
// The counters are written to storage rather than kept on the producer because
// `dynamic_to_string` runs after the stream has ended, potentially in another
// process; by then the producer that counted the rows no longer exists.
class ProfilingDemo : public vgi::TableFunction {
public:
    std::string name() const override { return "profiling_demo"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Sequence generator publishing diagnostics under EXPLAIN ANALYZE");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
                vgi::ArgSpec::named("batch_size", "int64", "Batch size for output"),
                vgi::ArgSpec::named("increment", "int64", "Step between values")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema, echo_count(params),
                                          echo_batch_size(params),
                                          params.arguments.named_int64("increment").value_or(1),
                                          params.storage, params.execution_id);
    }

    std::vector<std::pair<std::string, std::string>> dynamic_to_string(
        const std::string& global_execution_id, vgi::FunctionStorage& storage) const override {
        const auto snapshot = storage.kv_get(global_execution_id, kCountersKey);
        Counters counters;
        if (snapshot && snapshot->size() >= sizeof(counters)) {
            std::memcpy(&counters, snapshot->data(), sizeof(counters));
        } else {
            counters.started_ns = now_ns();
        }
        return {{"rows_produced", std::to_string(counters.rows)},
                {"batches_emitted", std::to_string(counters.batches)},
                {"elapsed_ms", std::to_string((now_ns() - counters.started_ns) / 1000000)}};
    }

private:
    static constexpr const char* kCountersKey = "profiling";

    // Written as raw bytes: the value is read back only by this function, and
    // the fixture is about the callback rather than about an encoding.
    struct Counters {
        uint64_t rows = 0;
        uint64_t batches = 0;
        uint64_t started_ns = 0;
    };

    static uint64_t now_ns() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size,
                 int64_t increment, std::shared_ptr<vgi::FunctionStorage> storage,
                 std::string execution_id)
            : schema_(std::move(schema)),
              remaining_(rows),
              batch_size_(batch_size),
              increment_(increment),
              storage_(std::move(storage)),
              execution_id_(std::move(execution_id)) {
            counters_.started_ns = now_ns();
        }

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t size = std::min(remaining_, batch_size_);

            arrow::Int64Builder builder;
            (void)builder.Reserve(size);
            for (int64_t i = cursor_; i < cursor_ + size; ++i) (void)builder.Append(i * increment_);
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);

            cursor_ += size;
            remaining_ -= size;
            counters_.rows += static_cast<uint64_t>(size);
            counters_.batches += 1;
            // Published per batch, not at end of stream: the engine may stop
            // reading early, and a LIMIT that never drains the scan would
            // otherwise leave nothing for the profiler to find.
            if (storage_) {
                storage_->kv_put(execution_id_, kCountersKey,
                                 std::string(reinterpret_cast<const char*>(&counters_),
                                             sizeof(counters_)));
            }
            return arrow::RecordBatch::Make(schema_, size, {array});
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        int64_t increment_;
        std::shared_ptr<vgi::FunctionStorage> storage_;
        std::string execution_id_;
        Counters counters_;
        int64_t cursor_ = 0;
    };
};

}  // namespace

void register_more_tables(vgi::Worker& worker) {
    worker.register_table(std::make_shared<ConstantColumns>());
    worker.register_table(std::make_shared<RepeatValue>("int64", arrow::int64()));
    worker.register_table(std::make_shared<RepeatValue>("varchar", arrow::utf8()));
    worker.register_table(std::make_shared<ProjectedData>());
    worker.register_table(std::make_shared<TenThousand>());
    worker.register_table(std::make_shared<OrderEcho>());
    worker.register_table(std::make_shared<SampleEcho>());
    worker.register_table(std::make_shared<ProfilingDemo>());
}

}  // namespace example
