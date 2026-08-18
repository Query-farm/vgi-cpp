// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Table (producer) fixtures: functions that generate rows without consuming
// any.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

std::shared_ptr<arrow::Schema> sequence_schema() {
    static const auto schema =
        arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/false)});
    return schema;
}

// Validated in both bind and init: bind is where a bad argument should be
// reported, but a producer built without the check would go on to generate a
// nonsense scan if the engine ever skipped straight to init.
void validate_sequence_args(const vgi::Arguments& arguments) {
    if (auto count = arguments.const_int64(0); count && *count < 0) {
        throw std::invalid_argument("sequence: count must be >= 0");
    }
    if (auto batch_size = arguments.named_int64("batch_size"); batch_size && *batch_size < 1) {
        throw std::invalid_argument("sequence: batch_size must be >= 1");
    }
    if (auto increment = arguments.named_int64("increment"); increment && *increment < 1) {
        throw std::invalid_argument("sequence: increment must be >= 1");
    }
}

// Emits a precomputed run of values in fixed-size chunks.
class Chunks : public vgi::TableProducer {
public:
    Chunks(std::vector<int64_t> values, int64_t batch_size)
        : values_(std::move(values)), batch_size_(batch_size) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (offset_ >= static_cast<int64_t>(values_.size())) return nullptr;
        const int64_t end =
            std::min<int64_t>(offset_ + batch_size_, static_cast<int64_t>(values_.size()));

        arrow::Int64Builder builder;
        (void)builder.Reserve(end - offset_);
        for (int64_t i = offset_; i < end; ++i)
            (void)builder.Append(values_[static_cast<size_t>(i)]);
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);

        auto batch = arrow::RecordBatch::Make(sequence_schema(), end - offset_, {array});
        offset_ = end;
        return batch;
    }

private:
    std::vector<int64_t> values_;
    int64_t batch_size_;
    int64_t offset_ = 0;
};

// `sequence(count, batch_size := 1000, increment := 1)` -> {n: int64}
class Sequence : public vgi::TableFunction {
public:
    std::string name() const override { return "sequence"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Generates a sequence of integers from 0 to n-1";
        md.categories = {"generator", "utility"};
        md.tags = {{"category", "generator"}, {"type", "utility"}};
        md.projection_pushdown = true;
        // Applied by the framework rather than honoured here: the values are
        // an arange, so there is nothing to skip at the source, and the engine
        // trusts a function that declares filter_pushdown to have applied the
        // predicate.
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
            // Named-only: supplied as `batch_size := N`, never positionally.
            vgi::ArgSpec::named("batch_size", "int64", "Batch size for output"),
            vgi::ArgSpec::named("increment", "int64", "Step between values"),
        };
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        validate_sequence_args(params.arguments);
        return sequence_schema();
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        validate_sequence_args(params.arguments);
        const int64_t count = std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        const int64_t increment = params.arguments.named_int64("increment").value_or(1);
        const int64_t batch_size =
            std::max<int64_t>(1, params.arguments.named_int64("batch_size").value_or(1000));

        std::vector<int64_t> values;
        values.reserve(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) values.push_back(i * increment);
        return std::make_unique<Chunks>(std::move(values), batch_size);
    }

    // Exact, not approximate: the span follows from the arguments, so a
    // filter outside it can be folded away entirely — which is the whole
    // reason to answer this at all.
    std::optional<std::vector<vgi::ColumnStatistics>> statistics(
        const vgi::ProcessParams& params) const override {
        const auto count = params.arguments.const_int64(0);
        if (!count || *count <= 0) return std::vector<vgi::ColumnStatistics>{};
        const int64_t increment = params.arguments.named_int64("increment").value_or(1);

        vgi::ColumnStatistics stat;
        stat.column_name = "n";
        stat.min = vgi::StatValue::integer(0);
        stat.max = vgi::StatValue::integer((*count - 1) * increment);
        stat.has_null = false;
        stat.has_not_null = true;
        stat.distinct_count = *count;
        return std::vector<vgi::ColumnStatistics>{std::move(stat)};
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            // Exact, not an estimate: the row count is the argument.
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }
};

}  // namespace

void register_table_functions(vgi::Worker& worker) {
    worker.register_table(std::make_shared<Sequence>());
}

}  // namespace example
