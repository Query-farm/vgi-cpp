// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Fixtures that break, on purpose, one clause each of the two contracts a
// self-describing batch signs: the partition values it advertises, and the
// batch index it is ordered by.
//
// Every function here exists to be *rejected*. The tests assert the engine's
// complaint, so making one of these correct would delete the thing under test.
// They share a file because the two contracts are the same shape — a promise
// about a batch, made in its metadata — and because half of the checks fire on
// the worker side and half only reach the engine, which is the distinction
// worth seeing side by side.

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>

#include <vgi/partition.h>
#include <vgi/worker.h>

namespace example {
namespace {

// The per-batch tag ordered sinks reassemble parallel output by. Spelled out
// rather than taken from the SDK because it is *batch* metadata, and the SDK's
// constants cover schema metadata only.
constexpr const char* kBatchIndexKey = "vgi_batch_index";

std::shared_ptr<arrow::Array> counting_column(int64_t rows) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(rows);
    for (int64_t i = 0; i < rows; ++i) (void)builder.Append(i);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> strings(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    (void)builder.Reserve(static_cast<int64_t>(values.size()));
    for (const auto& value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

bool is_partition_field(const arrow::Field& field) {
    const auto& metadata = field.metadata();
    if (!metadata) return false;
    const auto index = metadata->FindKey(vgi::kPartitionColumnKey);
    return index >= 0 && metadata->value(index) == "true";
}

// `vgi::partition_metadata`, but refusing rather than skipping when a column
// the schema annotates is missing from the batch.
//
// The SDK skips it, which ships a batch whose advertisement quietly omits a
// partition the engine was promised. This fixture is about the worker catching
// that before the wire, so the check lives here.
std::map<std::string, std::string> require_partition_metadata(
    const std::shared_ptr<arrow::Schema>& full_schema,
    const std::shared_ptr<arrow::RecordBatch>& batch) {
    for (const auto& field : full_schema->fields()) {
        if (!is_partition_field(*field)) continue;
        if (!batch->GetColumnByName(field->name())) {
            throw std::runtime_error("partition column \"" + field->name() +
                                     "\" is partition-annotated but absent from emitted batch");
        }
    }
    return vgi::partition_metadata(full_schema, batch);
}

std::shared_ptr<arrow::Schema> country_schema(bool annotated) {
    auto country = annotated ? vgi::partition_field("country", arrow::utf8())
                             : arrow::field("country", arrow::utf8(), /*nullable=*/true);
    return arrow::schema({country, arrow::field("sales", arrow::int64(), /*nullable=*/true)});
}

enum class PartitionFault {
    // A partition-annotated column, and a batch carrying no advertisement at
    // all. Only the engine can catch this: the worker emitted a batch that is
    // well-formed in itself.
    MissingValues,
    // SINGLE_VALUE_PARTITIONS promises one distinct value per batch; this one
    // holds two, so the advertised range has min != max. DuckDB's own
    // assertion is debug-only, which is what makes the engine-side re-check
    // this fixture proves worth having.
    MinNeqMax,
    // Partition values offered when no field is annotated to carry them —
    // caught on the worker side, before the wire.
    NoAnnotation,
    // An annotated column missing from the batch that is supposed to describe
    // it. Also worker-side.
    AbsentColumn,
};

class BrokenPartitionProducer : public vgi::TableProducer {
public:
    BrokenPartitionProducer(PartitionFault fault, int64_t rows) : fault_(fault), rows_(rows) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (done_) return nullptr;
        done_ = true;

        if (fault_ == PartitionFault::NoAnnotation) {
            throw std::runtime_error(
                "EmitPartitioned requires partition-annotated fields, but none were declared");
        }
        if (fault_ == PartitionFault::AbsentColumn) {
            auto without_country = arrow::RecordBatch::Make(
                arrow::schema({arrow::field("sales", arrow::int64(), /*nullable=*/true)}), rows_,
                {counting_column(rows_)});
            metadata_ = require_partition_metadata(country_schema(true), without_country);
            return without_country;
        }

        // Two countries in one batch for the min != max fault, one for the
        // rest — the value column is what decides which contract breaks.
        std::vector<std::string> countries;
        for (int64_t i = 0; i < rows_; ++i) {
            countries.push_back(fault_ == PartitionFault::MinNeqMax && i % 2 == 1 ? "BR" : "AU");
        }
        auto schema = country_schema(true);
        auto batch =
            arrow::RecordBatch::Make(schema, rows_, {strings(countries), counting_column(rows_)});
        if (fault_ == PartitionFault::MinNeqMax) {
            metadata_ = vgi::partition_metadata(schema, batch);
        }
        return batch;
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    PartitionFault fault_;
    int64_t rows_;
    bool done_ = false;
    std::map<std::string, std::string> metadata_;
};

class BrokenPartitionFunction : public vgi::TableFunction {
public:
    BrokenPartitionFunction(std::string name, PartitionFault fault)
        : name_(std::move(name)), fault_(fault) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Deliberately-broken partition fixture";
        md.categories = {"testing", "broken"};
        // The no-annotation fault leaves the kind unset: declaring one against
        // a schema with no partition column is a contradiction the binder
        // rejects, and the rejection under test is the worker's.
        if (fault_ != PartitionFault::NoAnnotation) {
            md.partition_kind = vgi::partition_kinds::kSingleValuePartitions;
        }
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Rows to attempt to emit")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return country_schema(fault_ != PartitionFault::NoAnnotation);
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
        return std::make_unique<BrokenPartitionProducer>(
            fault_, std::max<int64_t>(1, params.arguments.const_int64(0).value_or(1)));
    }

private:
    std::string name_;
    PartitionFault fault_;
};

enum class BatchIndexFault {
    // No tag at all on a data batch, from a function that declared it tags
    // every one.
    MissingTag,
    // Tag 10 followed by tag 3 on the same stream. The engine's ordered sink
    // needs the tags non-decreasing to reassemble anything.
    NonMonotone,
    // A tag far above DuckDB's per-pipeline cap, which overflows the sink's
    // index arithmetic rather than merely misordering it.
    Overflow,
};

class BrokenBatchIndexProducer : public vgi::TableProducer {
public:
    BrokenBatchIndexProducer(BatchIndexFault fault, int64_t rows) : fault_(fault), rows_(rows) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        auto schema = arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
        if (emitted_ == 0) {
            ++emitted_;
            // 2^60 — well above the 10^13 per-pipeline cap, and still far
            // enough from int64's range that the fixture is testing the
            // engine's own bound rather than an overflow on the way there.
            metadata_ = tag(fault_ == BatchIndexFault::Overflow ? (int64_t{1} << 60) : 10);
            return arrow::RecordBatch::Make(schema, rows_, {counting_column(rows_)});
        }
        if (fault_ != BatchIndexFault::NonMonotone || emitted_ > 1) return nullptr;
        ++emitted_;
        metadata_ = tag(3);
        return arrow::RecordBatch::Make(schema, 1, {counting_column(1)});
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    std::map<std::string, std::string> tag(int64_t index) const {
        if (fault_ == BatchIndexFault::MissingTag) return {};
        return {{kBatchIndexKey, std::to_string(index)}};
    }

    BatchIndexFault fault_;
    int64_t rows_;
    int emitted_ = 0;
    std::map<std::string, std::string> metadata_;
};

class BrokenBatchIndexFunction : public vgi::TableFunction {
public:
    BrokenBatchIndexFunction(std::string name, BatchIndexFault fault)
        : name_(std::move(name)), fault_(fault) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Deliberately-broken batch_index fixture";
        md.categories = {"testing", "broken"};
        // Declaring the tag is what installs the engine's contract check; the
        // whole point of these fixtures is to then break it.
        md.supports_batch_index = true;
        md.order_preservation = vgi::order_preservations::kFixedOrder;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Rows to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
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
        return std::make_unique<BrokenBatchIndexProducer>(
            fault_, std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)));
    }

private:
    std::string name_;
    BatchIndexFault fault_;
};

}  // namespace

void register_partition_broken(vgi::Worker& worker) {
    worker.register_table(std::make_shared<BrokenPartitionFunction>(
        "broken_missing_partition_values", PartitionFault::MissingValues));
    worker.register_table(std::make_shared<BrokenPartitionFunction>("broken_partition_min_neq_max",
                                                                    PartitionFault::MinNeqMax));
    worker.register_table(std::make_shared<BrokenPartitionFunction>(
        "broken_partition_values_no_annotation", PartitionFault::NoAnnotation));
    worker.register_table(std::make_shared<BrokenPartitionFunction>(
        "broken_partition_column_absent_from_batch", PartitionFault::AbsentColumn));

    worker.register_table(std::make_shared<BrokenBatchIndexFunction>(
        "broken_missing_batch_index_tag", BatchIndexFault::MissingTag));
    worker.register_table(std::make_shared<BrokenBatchIndexFunction>(
        "broken_non_monotone_batch_index", BatchIndexFault::NonMonotone));
    worker.register_table(std::make_shared<BrokenBatchIndexFunction>("broken_batch_index_overflow",
                                                                     BatchIndexFault::Overflow));
}

}  // namespace example
