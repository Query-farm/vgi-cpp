// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Fixtures that report what was pushed into them.
//
// These do not *use* the filters — they render them into a column, so a test
// can assert on exactly what the engine sent. That is why they declare
// `filter_pushdown` but not `auto_apply_filters`: applying them would change
// the row count the test is counting.

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

class FilterEcho : public vgi::TableFunction {
public:
    std::string name() const override { return "filter_echo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Echoes pushed-down filter predicates in output";
        md.categories = {"generator", "diagnostic"};
        md.filter_pushdown = true;
        // Deliberately *not* projection_pushdown. This fixture reports what it
        // was told, and the engine narrowing its output to the filtered column
        // alone would leave the report column unbuilt — the report is the
        // point, so the whole row is always produced.
        md.projection_pushdown = false;
        // Reports the filters *and* applies them. The engine trusts a function
        // that declares filter_pushdown to have honoured the predicate, so
        // advertising without applying returns rows the WHERE excluded.
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
                vgi::ArgSpec::named("batch_size", "int64", "Batch size for output")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), true),
                              arrow::field("s", arrow::utf8(), true),
                              arrow::field("pushed_filters", arrow::utf8(), true)});
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
            std::max<int64_t>(1, params.arguments.named_int64("batch_size").value_or(2048)),
            params.pushdown_filters.format());
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size,
                 std::string filters)
            : schema_(std::move(schema)),
              remaining_(rows),
              batch_size_(batch_size),
              filters_(std::move(filters)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t n = std::min(remaining_, batch_size_);

            arrow::Int64Builder ns;
            arrow::StringBuilder ss;
            arrow::StringBuilder pushed;
            (void)ns.Reserve(n);
            for (int64_t i = cursor_; i < cursor_ + n; ++i) {
                (void)ns.Append(i);
                (void)ss.Append("row_" + std::to_string(i));
                (void)pushed.Append(filters_);
            }

            // Built in the *bound* schema's column order, since projection
            // pushdown may have narrowed and reordered it.
            std::vector<std::shared_ptr<arrow::Array>> built(3);
            (void)ns.Finish(&built[0]);
            (void)ss.Finish(&built[1]);
            (void)pushed.Finish(&built[2]);

            static const char* kNames[] = {"n", "s", "pushed_filters"};
            std::vector<std::shared_ptr<arrow::Array>> columns;
            columns.reserve(static_cast<size_t>(schema_->num_fields()));
            for (int i = 0; i < schema_->num_fields(); ++i) {
                const auto& wanted = schema_->field(i)->name();
                for (size_t j = 0; j < 3; ++j) {
                    if (wanted == kNames[j]) columns.push_back(built[j]);
                }
            }
            cursor_ += n;
            remaining_ -= n;
            return arrow::RecordBatch::Make(schema_, n, columns);
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        std::string filters_;
        int64_t cursor_ = 0;
    };
};

}  // namespace

void register_filter_fixtures(vgi::Worker& worker) {
    worker.register_table(std::make_shared<FilterEcho>());
}

}  // namespace example
