// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Fixtures that read settings, including one whose *schema* depends on one.

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

// Verbose mode adds a column. The schema therefore has to be settled from the
// setting at bind, not from the arguments — which is the point of the fixture.
std::shared_ptr<arrow::Schema> settings_aware_schema(bool verbose) {
    std::vector<std::shared_ptr<arrow::Field>> fields{
        arrow::field("id", arrow::int64(), true),
        arrow::field("greeting", arrow::utf8(), true),
        arrow::field("value", arrow::float64(), true),
    };
    if (verbose) fields.push_back(arrow::field("details", arrow::utf8(), true));
    return arrow::schema(std::move(fields));
}

class SettingsAware : public vgi::TableFunction {
public:
    std::string name() const override { return "settings_aware"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Generates data demonstrating settings are passed";
        md.categories = {"generator", "utility"};
        md.required_settings = {"vgi_verbose_mode", "greeting", "multiplier"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        return settings_aware_schema(
            params.settings.get_bool("vgi_verbose_mode").value_or(false));
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const bool verbose = params.settings.get_bool("vgi_verbose_mode").value_or(false);
        return std::make_unique<Producer>(
            settings_aware_schema(verbose),
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)),
            params.settings.get_string("greeting").value_or("Hello"),
            params.settings.get_int64("multiplier").value_or(1), verbose);
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t count, std::string greeting,
                 int64_t multiplier, bool verbose)
            : schema_(std::move(schema)),
              count_(count),
              greeting_(std::move(greeting)),
              multiplier_(multiplier),
              verbose_(verbose) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (emitted_) return nullptr;
            emitted_ = true;

            arrow::Int64Builder ids;
            arrow::StringBuilder greetings;
            arrow::DoubleBuilder values;
            arrow::StringBuilder details;
            (void)ids.Reserve(count_);
            for (int64_t i = 0; i < count_; ++i) {
                (void)ids.Append(i);
                (void)greetings.Append(greeting_);
                (void)values.Append(static_cast<double>(i) * 2.5 *
                                    static_cast<double>(multiplier_));
                if (verbose_) (void)details.Append("row_" + std::to_string(i));
            }

            std::vector<std::shared_ptr<arrow::Array>> columns(verbose_ ? 4 : 3);
            (void)ids.Finish(&columns[0]);
            (void)greetings.Finish(&columns[1]);
            (void)values.Finish(&columns[2]);
            if (verbose_) (void)details.Finish(&columns[3]);
            return arrow::RecordBatch::Make(schema_, count_, columns);
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t count_;
        std::string greeting_;
        int64_t multiplier_;
        bool verbose_;
        bool emitted_ = false;
    };
};

// `filter_by_setting(data)` — keeps rows whose `value` is at least the
// `threshold` setting.
class FilterBySetting : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "filter_by_setting"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Filter rows where value column >= threshold setting";
        md.categories = {"transform", "settings"};
        md.required_settings = {"threshold"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t threshold = params.settings.get_int64("threshold").value_or(0);
        auto source = batch->GetColumnByName("value");
        if (!source) throw std::runtime_error("filter_by_setting: no 'value' column");

        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(source, arrow::int64()));
        arrow::BooleanBuilder keep;
        (void)keep.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            // A null value is dropped rather than kept: it is not >= anything.
            (void)keep.Append(!values->IsNull(i) && values->Value(i) >= threshold);
        }
        std::shared_ptr<arrow::Array> mask;
        (void)keep.Finish(&mask);

        auto filtered = arrow::compute::Filter(batch, mask);
        if (!filtered.ok()) {
            throw std::runtime_error("filter_by_setting: " + filtered.status().message());
        }
        return {vgi::project_batch(filtered.MoveValueUnsafe().record_batch(),
                                   params.output_schema)};
    }
};

}  // namespace

void register_settings_tables(vgi::Worker& worker) {
    worker.register_table(std::make_shared<SettingsAware>());
    worker.register_table_in_out(std::make_shared<FilterBySetting>());
}

}  // namespace example
