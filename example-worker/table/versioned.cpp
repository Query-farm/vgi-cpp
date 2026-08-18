// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The scan behind a time-travelling table.
//
// One function serves every version, selected by a constant argument the
// catalog supplies per version. That is why the table is declared with
// `inline_scan = false`: the engine re-resolves the scan per query so each AT
// clause reaches its own arguments, where an inlined scan would be bound once
// and reused.

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

struct Version {
    std::shared_ptr<arrow::Schema> schema;
    std::vector<std::shared_ptr<arrow::Array>> columns;
};

std::shared_ptr<arrow::Array> int64_column(std::vector<int64_t> values) {
    arrow::Int64Builder builder;
    for (int64_t value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> double_column(std::vector<double> values) {
    arrow::DoubleBuilder builder;
    for (double value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> string_column(std::vector<std::string> values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> bool_column(std::vector<bool> values) {
    arrow::BooleanBuilder builder;
    for (bool value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// Schema *and* rows change across versions — that is what the fixture is for.
// Version 3 is the fallback, so an unrecognised version reads as "current"
// rather than failing here; the catalog has already rejected an unknown AT.
Version build_version(int64_t version) {
    if (version == 1) {
        return {arrow::schema({arrow::field("id", arrow::int64(), true)}),
                {int64_column({1, 2, 3})}};
    }
    if (version == 2) {
        return {arrow::schema({arrow::field("id", arrow::int64(), true),
                               arrow::field("name", arrow::utf8(), true),
                               arrow::field("score", arrow::float64(), true),
                               arrow::field("active", arrow::boolean(), true)}),
                {int64_column({1, 2, 3, 4, 5}),
                 string_column({"alice", "bob", "carol", "dave", "eve"}),
                 double_column({10.0, 20.0, 30.0, 40.0, 50.0}),
                 bool_column({true, false, true, false, true})}};
    }
    return {arrow::schema({arrow::field("id", arrow::int64(), true),
                           arrow::field("score", arrow::float64(), true)}),
            {int64_column({1, 2, 3, 4}), double_column({15.0, 25.0, 35.0, 45.0})}};
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

class VersionedDataScan : public vgi::TableFunction {
public:
    std::string name() const override { return "versioned_data_scan"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Versioned data scan (time travel)";
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("version", 0, "int64", "Data version to return")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        return build_version(params.arguments.const_int64(0).value_or(3)).schema;
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        auto version = build_version(params.arguments.const_int64(0).value_or(3));
        vgi::TableCardinality estimate;
        const int64_t rows = version.columns.empty() ? 0 : version.columns.front()->length();
        estimate.estimate = rows;
        estimate.max = rows;
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        auto version = build_version(params.arguments.const_int64(0).value_or(3));
        const int64_t rows = version.columns.empty() ? 0 : version.columns.front()->length();
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(version.schema, rows, version.columns));
    }
};

}  // namespace

void register_versioned(vgi::Worker& worker) {
    worker.register_table(std::make_shared<VersionedDataScan>());
}

// Declares the time-travelling table itself; called from the catalog.
std::vector<vgi::TimeTravelVersion> versioned_data_versions() {
    std::vector<vgi::TimeTravelVersion> versions;
    // Year per version, so `AT (TIMESTAMP => '2021-…')` picks version 2.
    const int years[] = {2020, 2021, 2022};
    for (int64_t version = 1; version <= 3; ++version) {
        vgi::TimeTravelVersion entry;
        entry.version = version;
        entry.columns = build_version(version).schema;
        entry.scan_function = "versioned_data_scan";
        entry.scan_arguments = vgi::serialize_scan_arguments({int64_column({version})});
        entry.timestamp_year = years[version - 1];
        versions.push_back(std::move(entry));
    }
    return versions;
}

}  // namespace example
