// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Fixtures whose work is divided once and claimed from a shared queue, plus
// the two families that describe each emitted batch — by partition id, for the
// ordered sinks that reassemble parallel output, and by partition value, for
// the planner that wants a partitioned aggregate.
//
// The queue is what makes them safe across workers: it holds the scan
// position, so no two workers can generate the same rows however the engine
// schedules them. `on_init` fills it exactly once, from the primary.

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/partition.h>
#include <vgi/worker.h>

namespace example {
namespace {

// Rows per emitted batch, independent of the chunk size: a chunk is how the
// work is split between workers, a batch is how much of it crosses the wire at
// once.
constexpr int64_t kBatchSize = 1000;

// Rows per chunk, for the fixtures that size their queue by the row count.
constexpr int64_t kChunkRows = 1000;

// Cap on the queue's length, for the ones that instead size it so the cost of
// fanning out follows the number of workers rather than the number of rows.
constexpr int64_t kMaxChunks = 24;

// The per-batch tag ordered sinks reassemble parallel output by. Spelled out
// here rather than taken from the SDK because it is *batch* metadata, and the
// SDK's constants cover schema metadata only.
constexpr const char* kBatchIndexKey = "vgi_batch_index";

// A short fixed list, so the partition count and the SQL tests' expected sums
// are both stable.
const char* const kCountries[] = {"AU", "BR", "CA", "FR", "US"};
constexpr int64_t kCountryCount = static_cast<int64_t>(std::size(kCountries));

const char* const kCategories[] = {"books", "music", "video"};
constexpr int64_t kCategoryCount = static_cast<int64_t>(std::size(kCategories));

// One partition per (region, year) pair. Listed year-major within a region so
// the partition index a test reasons about matches reading order.
struct RegionYear {
    const char* region;
    int64_t year;
};
constexpr RegionYear kRegionYears[] = {{"AMER", 2023}, {"AMER", 2024}, {"EMEA", 2023},
                                       {"EMEA", 2024}, {"APAC", 2023}, {"APAC", 2024}};
constexpr int64_t kRegionYearCount = static_cast<int64_t>(std::size(kRegionYears));

vgi::FunctionMetadata fixture_metadata(std::string description,
                                       std::vector<std::string> categories) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = std::move(categories);
    return md;
}

template <typename Fn>
std::shared_ptr<arrow::Array> int64_column(int64_t rows, Fn value) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(rows);
    for (int64_t i = 0; i < rows; ++i) (void)builder.Append(value(i));
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

template <typename Fn>
std::shared_ptr<arrow::Array> double_column(int64_t rows, Fn value) {
    arrow::DoubleBuilder builder;
    (void)builder.Reserve(rows);
    for (int64_t i = 0; i < rows; ++i) (void)builder.Append(value(i));
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> repeated_string(int64_t rows, const std::string& value) {
    arrow::StringBuilder builder;
    (void)builder.Reserve(rows);
    for (int64_t i = 0; i < rows; ++i) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// One claim from the shared queue: rows `[start, end)` of partition `id`.
struct Chunk {
    int64_t id = 0;
    int64_t start = 0;
    int64_t end = 0;
};

std::string encode(const Chunk& chunk) {
    return std::to_string(chunk.id) + ":" + std::to_string(chunk.start) + ":" +
           std::to_string(chunk.end);
}

std::optional<Chunk> decode(const std::string& item) {
    Chunk chunk;
    char* rest = nullptr;
    chunk.id = std::strtoll(item.c_str(), &rest, 10);
    if (*rest != ':') return std::nullopt;
    chunk.start = std::strtoll(rest + 1, &rest, 10);
    if (*rest != ':') return std::nullopt;
    chunk.end = std::strtoll(rest + 1, &rest, 10);
    return chunk;
}

// Divide `[0, count)` into `size`-row chunks numbered from zero.
void push_ranges(const vgi::ProcessParams& params, int64_t count, int64_t size) {
    std::vector<std::string> items;
    int64_t id = 0;
    for (int64_t start = 0; start < count; start += size, ++id) {
        items.push_back(encode({id, start, std::min(start + size, count)}));
    }
    params.storage->queue_push(params.execution_id, items);
}

// One chunk per partition, each covering the same `rows`.
//
// Distinct from `push_ranges` because here a chunk *is* a partition rather than
// a slice of one flat range: the partition id selects the values the rows carry.
void push_partitions(const vgi::ProcessParams& params, int64_t partitions, int64_t rows) {
    std::vector<std::string> items;
    for (int64_t id = 0; id < partitions; ++id) items.push_back(encode({id, 0, rows}));
    params.storage->queue_push(params.execution_id, items);
}

// Claims chunks from the execution's queue and emits each at most `kBatchSize`
// rows at a time.
//
// Everything about *which* rows are this worker's lives here, since it is the
// same for every fixture in this file; a subclass supplies one column at a
// time and, if it has anything to say about a batch, describes it.
class ChunkProducer : public vgi::TableProducer {
public:
    ChunkProducer(std::shared_ptr<arrow::Schema> schema,
                  std::shared_ptr<vgi::FunctionStorage> storage, std::string execution_id)
        : schema_(std::move(schema)),
          storage_(std::move(storage)),
          execution_id_(std::move(execution_id)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() final {
        while (cursor_ >= chunk_.end) {
            auto item = storage_->queue_pop(execution_id_);
            if (!item) return nullptr;  // queue drained: this worker is done
            auto claimed = decode(*item);
            if (!claimed) return nullptr;
            chunk_ = *claimed;
            cursor_ = chunk_.start;
        }
        const int64_t rows = std::min(kBatchSize, chunk_.end - cursor_);

        // Built against the *bound* schema rather than the declared one:
        // projection pushdown may have dropped columns or reordered them, and
        // a batch whose columns disagree with the schema it was made against
        // is accepted by Arrow and corrupts whoever reads it later.
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(schema_->num_fields()));
        for (int i = 0; i < schema_->num_fields(); ++i) {
            columns.push_back(column(schema_->field(i)->name(), chunk_, cursor_, rows));
        }
        auto batch = arrow::RecordBatch::Make(schema_, rows, columns);
        cursor_ += rows;
        // Computed now, not in `last_metadata`: the framework calls that once
        // per emitted batch, by which point the cursor has already moved past
        // the batch the metadata is supposed to describe.
        metadata_ = describe(chunk_, batch);
        return batch;
    }

    std::map<std::string, std::string> last_metadata() const final { return metadata_; }

protected:
    // Column `name` of `chunk`, holding `rows` rows starting at row `from` of
    // the flat range the chunk covers.
    virtual std::shared_ptr<arrow::Array> column(const std::string& name, const Chunk& chunk,
                                                 int64_t from, int64_t rows) const = 0;

    virtual std::map<std::string, std::string> describe(
        const Chunk&, const std::shared_ptr<arrow::RecordBatch>&) const {
        return {};
    }

private:
    std::shared_ptr<arrow::Schema> schema_;
    std::shared_ptr<vgi::FunctionStorage> storage_;
    std::string execution_id_;
    Chunk chunk_;
    int64_t cursor_ = 0;
    std::map<std::string, std::string> metadata_;
};

// `n = row * increment`, in queue order.
class SequenceChunks : public ChunkProducer {
public:
    SequenceChunks(std::shared_ptr<arrow::Schema> schema,
                   std::shared_ptr<vgi::FunctionStorage> storage, std::string execution_id,
                   int64_t increment)
        : ChunkProducer(std::move(schema), std::move(storage), std::move(execution_id)),
          increment_(increment) {}

protected:
    std::shared_ptr<arrow::Array> column(const std::string&, const Chunk&, int64_t from,
                                         int64_t rows) const override {
        return int64_column(rows, [&](int64_t i) { return (from + i) * increment_; });
    }

private:
    int64_t increment_;
};

// Tags every batch with the id of the partition it came from.
//
// That tag is the entire batch-index opt-in: ordered sinks reassemble a
// parallel scan by it, which is what lets the engine leave the source parallel
// instead of clamping it to one thread.
class TaggedChunks : public ChunkProducer {
public:
    using ChunkProducer::ChunkProducer;

protected:
    std::map<std::string, std::string> describe(
        const Chunk& chunk, const std::shared_ptr<arrow::RecordBatch>&) const override {
        return {{kBatchIndexKey, std::to_string(chunk.id)}};
    }
};

// `n` counts across the whole scan, so a reassembled result reads 0, 1, 2, …
class TaggedSequence : public TaggedChunks {
public:
    using TaggedChunks::TaggedChunks;

protected:
    std::shared_ptr<arrow::Array> column(const std::string&, const Chunk&, int64_t from,
                                         int64_t rows) const override {
        return int64_column(rows, [&](int64_t i) { return from + i; });
    }
};

// `(partition_id, seq)` — the partition id in the rows themselves, so a test
// can see reassembly directly rather than inferring it from a value order.
class TaggedMarked : public TaggedChunks {
public:
    using TaggedChunks::TaggedChunks;

protected:
    std::shared_ptr<arrow::Array> column(const std::string& name, const Chunk& chunk,
                                         int64_t from, int64_t rows) const override {
        if (name == "partition_id") {
            return int64_column(rows, [&](int64_t) { return chunk.id; });
        }
        const int64_t offset = from - chunk.start;
        return int64_column(rows, [&](int64_t i) { return offset + i; });
    }
};

// `(n, worker_pid, pushed_filters)` — every worker reports the predicate it
// was handed, so a test can confirm the filter reached all of them.
class FilterEchoChunks : public ChunkProducer {
public:
    FilterEchoChunks(std::shared_ptr<arrow::Schema> schema,
                     std::shared_ptr<vgi::FunctionStorage> storage, std::string execution_id,
                     std::string filters)
        : ChunkProducer(std::move(schema), std::move(storage), std::move(execution_id)),
          filters_(std::move(filters)),
          pid_(static_cast<int64_t>(::getpid())) {}

protected:
    std::shared_ptr<arrow::Array> column(const std::string& name, const Chunk&, int64_t from,
                                         int64_t rows) const override {
        if (name == "worker_pid") return int64_column(rows, [&](int64_t) { return pid_; });
        if (name == "pushed_filters") return repeated_string(rows, filters_);
        return int64_column(rows, [&](int64_t i) { return from + i; });
    }

private:
    std::string filters_;
    int64_t pid_;
};

// Describes each batch with the range its partition columns cover.
//
// The advertisement rides *every* batch, not just the first one this worker
// emits: the engine reads it per chunk, and a scan whose leading partition is
// filtered down to nothing would otherwise advertise nothing at all.
class PartitionedChunks : public ChunkProducer {
public:
    PartitionedChunks(std::shared_ptr<arrow::Schema> schema,
                      std::shared_ptr<vgi::FunctionStorage> storage, std::string execution_id,
                      std::shared_ptr<arrow::Schema> declared)
        : ChunkProducer(std::move(schema), std::move(storage), std::move(execution_id)),
          declared_(std::move(declared)) {}

protected:
    std::map<std::string, std::string> describe(
        const Chunk&, const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        // Against the declared schema rather than the bound one, since it is
        // the declaration that carries the partition-column markers.
        return vgi::partition_metadata(declared_, batch);
    }

private:
    std::shared_ptr<arrow::Schema> declared_;
};

// One country per chunk, so `country` holds a single value per batch — the
// SINGLE_VALUE_PARTITIONS contract, stated in data.
class CountryChunks : public PartitionedChunks {
public:
    using PartitionedChunks::PartitionedChunks;

protected:
    std::shared_ptr<arrow::Array> column(const std::string& name, const Chunk& chunk,
                                         int64_t from, int64_t rows) const override {
        if (name == "country") return repeated_string(rows, kCountries[chunk.id]);
        // Unique per (country, row), which is what makes the tests' SUM
        // expectations short to write and a misattributed row obvious.
        const int64_t base = chunk.id * 1000000 + (from - chunk.start);
        return int64_column(rows, [&](int64_t i) { return base + i; });
    }
};

// One (region, year) pair per chunk — two partition columns rather than one,
// so a batch advertises a tuple and the engine has to read every part of it.
class RegionYearChunks : public PartitionedChunks {
public:
    using PartitionedChunks::PartitionedChunks;

protected:
    std::shared_ptr<arrow::Array> column(const std::string& name, const Chunk& chunk,
                                         int64_t from, int64_t rows) const override {
        const auto& partition = kRegionYears[chunk.id];
        if (name == "region") return repeated_string(rows, partition.region);
        if (name == "year") return int64_column(rows, [&](int64_t) { return partition.year; });
        const auto base = static_cast<double>(chunk.id * 1000 + (from - chunk.start));
        return double_column(rows, [&](int64_t i) { return base + static_cast<double>(i); });
    }
};

// One category per chunk, `revenue` offset by the category — a batch credited
// to the wrong partition then shows up as a wrong sum rather than a plausible
// one.
class CategoryChunks : public PartitionedChunks {
public:
    using PartitionedChunks::PartitionedChunks;

protected:
    std::shared_ptr<arrow::Array> column(const std::string& name, const Chunk& chunk,
                                         int64_t from, int64_t rows) const override {
        if (name == "category") return repeated_string(rows, kCategories[chunk.id]);
        const int64_t base = (chunk.id + 1) * 100 + (from - chunk.start);
        return int64_column(rows, [&](int64_t i) { return base + i; });
    }
};

// Each chunk's `key` covers an interval, so the batch describes a range rather
// than a value — DISJOINT or OVERLAPPING rather than SINGLE_VALUE.
//
// `stride` is the whole difference between the two: 1000 keeps consecutive
// intervals apart whatever a caller asks for, 500 lets them meet once a
// partition holds more than 500 rows.
class RangeChunks : public PartitionedChunks {
public:
    RangeChunks(std::shared_ptr<arrow::Schema> schema,
                std::shared_ptr<vgi::FunctionStorage> storage, std::string execution_id,
                std::shared_ptr<arrow::Schema> declared, int64_t stride)
        : PartitionedChunks(std::move(schema), std::move(storage), std::move(execution_id),
                            std::move(declared)),
          stride_(stride) {}

protected:
    std::shared_ptr<arrow::Array> column(const std::string& name, const Chunk& chunk,
                                         int64_t from, int64_t rows) const override {
        const int64_t offset = from - chunk.start;
        const int64_t base = name == "key" ? chunk.id * stride_ : chunk.id * 10;
        return int64_column(rows, [&](int64_t i) { return base + offset + i; });
    }

private:
    int64_t stride_;
};

// Emits one batch and stops.
class SingleBatch : public vgi::TableProducer {
public:
    SingleBatch(std::shared_ptr<arrow::Schema> schema, int64_t rows)
        : schema_(std::move(schema)), rows_(rows) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() final {
        if (done_) return nullptr;
        done_ = true;
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(schema_->num_fields()));
        for (int i = 0; i < schema_->num_fields(); ++i) {
            columns.push_back(column(schema_->field(i)->name(), rows_));
        }
        return arrow::RecordBatch::Make(schema_, rows_, columns);
    }

protected:
    virtual std::shared_ptr<arrow::Array> column(const std::string& name,
                                                 int64_t rows) const = 0;

private:
    std::shared_ptr<arrow::Schema> schema_;
    int64_t rows_;
    bool done_ = false;
};

// `partitioned_sequence(count, increment := 1)` and the three
// `partitioned_*_order` variants — one generator behind four names, so the
// order-mode tests can compare identical rows under different planner
// expectations. The rows are the control; the declared mode is the variable.
class PartitionedSequence : public vgi::TableFunction {
public:
    PartitionedSequence(std::string name, std::string description, bool takes_increment,
                        std::string order)
        : name_(std::move(name)),
          description_(std::move(description)),
          takes_increment_(takes_increment),
          order_(std::move(order)) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        auto md = fixture_metadata(description_, {"generator", "utility"});
        md.order_preservation = order_;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        std::vector<vgi::ArgSpec> specs{
            vgi::ArgSpec::constant_arg("count", 0, "int64", "Total integers to generate")};
        if (takes_increment_) {
            specs.push_back(vgi::ArgSpec::named("increment", "int64", "Step between values"));
        }
        return specs;
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 4; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        const int64_t count = row_count(params);
        // The increment variant caps the queue's length so the fan-out cost
        // tracks the worker count; the order-mode one uses a fixed chunk so its
        // queue grows with the query, which is what puts several chunks in
        // flight at once for the ordering assertions to observe.
        const int64_t size = takes_increment_
                                 ? std::max<int64_t>(1, (count + kMaxChunks - 1) / kMaxChunks)
                                 : kChunkRows;
        push_ranges(params, count, size);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<SequenceChunks>(params.output_schema, params.storage,
                                                params.execution_id, increment(params));
    }

private:
    static int64_t row_count(const vgi::ProcessParams& params) {
        return std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
    }

    int64_t increment(const vgi::ProcessParams& params) const {
        if (!takes_increment_) return 1;
        return params.arguments.named_int64("increment").value_or(1);
    }

    std::string name_;
    std::string description_;
    bool takes_increment_;
    std::string order_;
};

// `partitioned_batch_index(count)` — 0..count across four workers, every batch
// tagged so the ordered sinks put them back in order.
class PartitionedBatchIndex : public vgi::TableFunction {
public:
    std::string name() const override { return "partitioned_batch_index"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = fixture_metadata(
            "Multi-worker partitioned sequence with per-batch batch_index tagging",
            {"generator", "utility"});
        md.projection_pushdown = true;
        md.supports_batch_index = true;
        md.order_preservation = vgi::order_preservations::kFixedOrder;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_arg("count", 0, "int64", "Total integers to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 4; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        push_ranges(params, std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)),
                    kChunkRows);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<TaggedSequence>(params.output_schema, params.storage,
                                                params.execution_id);
    }
};

// `partitioned_batch_index_marked(count, chunk_size := 1000)` — the same
// scheme with the partition id carried in the rows.
class PartitionedBatchIndexMarked : public vgi::TableFunction {
public:
    std::string name() const override { return "partitioned_batch_index_marked"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = fixture_metadata("Two-column batch_index demo: rows are (partition_id, seq)",
                                   {"generator", "utility"});
        // Projection pushdown stays off so `partition_id` survives whatever
        // the query selects — it is the column the ordering assertions read.
        md.projection_pushdown = false;
        md.supports_batch_index = true;
        md.order_preservation = vgi::order_preservations::kFixedOrder;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Total rows to generate"),
                vgi::ArgSpec::named("chunk_size", "int64", "Rows per partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("partition_id", arrow::int64(), /*nullable=*/true),
                              arrow::field("seq", arrow::int64(), /*nullable=*/true)});
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 4; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        push_ranges(params, std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)),
                    std::max<int64_t>(
                        1, params.arguments.named_int64("chunk_size").value_or(kChunkRows)));
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<TaggedMarked>(params.output_schema, params.storage,
                                              params.execution_id);
    }
};

// `filter_echo_partitioned(count)` — filter pushdown observed across several
// workers at once, which is what distinguishes it from `filter_echo`.
class FilterEchoPartitioned : public vgi::TableFunction {
public:
    std::string name() const override { return "filter_echo_partitioned"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = fixture_metadata(
            "Multi-worker partitioned sequence that echoes pushed-down filters",
            {"generator", "diagnostic", "testing"});
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        // Reports the filters *and* applies them: the engine trusts a function
        // declaring filter_pushdown to have honoured the predicate, so echoing
        // without applying returns rows the WHERE excluded.
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true),
                              arrow::field("worker_pid", arrow::int64(), /*nullable=*/true),
                              arrow::field("pushed_filters", arrow::utf8(), /*nullable=*/true)});
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 4; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        const int64_t count = std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        push_ranges(params, count,
                    std::max<int64_t>(1, (count + kMaxChunks - 1) / kMaxChunks));
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<FilterEchoChunks>(params.output_schema, params.storage,
                                                  params.execution_id,
                                                  params.pushdown_filters.format());
    }
};

std::shared_ptr<arrow::Schema> country_schema() {
    return arrow::schema({vgi::partition_field("country", arrow::utf8()),
                          arrow::field("sales", arrow::int64(), /*nullable=*/true)});
}

// `country_partitioned_sales(rows_per_country)` — one batch per country, the
// fixture the PARTITIONED_AGGREGATE planner check runs against.
class CountryPartitionedSales : public vgi::TableFunction {
public:
    std::string name() const override { return "country_partitioned_sales"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = fixture_metadata(
            "Per-country sales rows, one Arrow batch per country. Declares country as a "
            "SINGLE_VALUE partition column.",
            {"generator", "partitioning"});
        md.partition_kind = vgi::partition_kinds::kSingleValuePartitions;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows_per_country", 0, "int64",
                                           "Rows to emit per country partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return country_schema();
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 4; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto rows = params.arguments.const_int64(0)) {
            estimate.estimate = *rows * kCountryCount;
            estimate.max = *rows * kCountryCount;
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        push_partitions(params, kCountryCount, rows(params));
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<CountryChunks>(params.output_schema, params.storage,
                                               params.execution_id, country_schema());
    }

private:
    static int64_t rows(const vgi::ProcessParams& params) {
        return std::max<int64_t>(1, params.arguments.const_int64(0).value_or(1));
    }
};

std::shared_ptr<arrow::Schema> region_year_schema() {
    return arrow::schema({vgi::partition_field("region", arrow::utf8()),
                          vgi::partition_field("year", arrow::int64()),
                          arrow::field("value", arrow::float64(), /*nullable=*/true)});
}

// `region_year_partitioned(rows)` — the multi-column partitioning case, where
// a GROUP BY over a strict subset of the declared columns is a question the
// planner is free to answer either way.
class RegionYearPartitioned : public vgi::TableFunction {
public:
    std::string name() const override { return "region_year_partitioned"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = fixture_metadata(
            "Rows partitioned by (region, year), one Arrow batch per pair. Declares both as "
            "SINGLE_VALUE partition columns.",
            {"generator", "partitioning"});
        md.partition_kind = vgi::partition_kinds::kSingleValuePartitions;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows", 0, "int64",
                                           "Rows to emit per (region, year) partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return region_year_schema();
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 4; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto rows = params.arguments.const_int64(0)) {
            estimate.estimate = *rows * kRegionYearCount;
            estimate.max = *rows * kRegionYearCount;
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        push_partitions(params, kRegionYearCount, rows(params));
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<RegionYearChunks>(params.output_schema, params.storage,
                                                  params.execution_id, region_year_schema());
    }

private:
    static int64_t rows(const vgi::ProcessParams& params) {
        return std::max<int64_t>(1, params.arguments.const_int64(0).value_or(1));
    }
};

std::shared_ptr<arrow::Schema> category_schema() {
    return arrow::schema({vgi::partition_field("category", arrow::utf8()),
                          arrow::field("revenue", arrow::int64(), /*nullable=*/true)});
}

// `partitioned_with_explicit_override(rows)` — the same shape as the country
// fixture, but the batch's partition advertisement is stated outright instead
// of being read back off the emitted column.
//
// A worker that knows the partition it is serving does not have to scan the
// batch to say so, and the engine has to accept either answer.
class PartitionedWithExplicitOverride : public vgi::TableFunction {
public:
    std::string name() const override { return "partitioned_with_explicit_override"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = fixture_metadata(
            "Per-category revenue rows whose partition values are supplied explicitly rather "
            "than extracted from the batch",
            {"generator", "partitioning"});
        md.partition_kind = vgi::partition_kinds::kSingleValuePartitions;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows", 0, "int64",
                                           "Rows to emit per category partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return category_schema();
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 4; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto rows = params.arguments.const_int64(0)) {
            estimate.estimate = *rows * kCategoryCount;
            estimate.max = *rows * kCategoryCount;
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        push_partitions(params, kCategoryCount, rows(params));
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Override>(params.output_schema, params.storage,
                                          params.execution_id, category_schema());
    }

private:
    // Describes the batch from the chunk it came from rather than from the
    // rows: the category is known before a single row is built, so reading it
    // back out of the batch would only be a slower way to reach the same
    // answer — and the engine must accept an advertisement built that way.
    class Override : public CategoryChunks {
    public:
        using CategoryChunks::CategoryChunks;

    protected:
        std::map<std::string, std::string> describe(
            const Chunk& chunk, const std::shared_ptr<arrow::RecordBatch>&) const override {
            // One row carrying the value this chunk was assigned. Its min and
            // max are that value, which is the whole advertisement.
            auto stated = arrow::RecordBatch::Make(
                category_schema(), 1,
                {repeated_string(1, kCategories[chunk.id]),
                 int64_column(1, [](int64_t) { return int64_t{0}; })});
            return vgi::partition_metadata(category_schema(), stated);
        }
    };

    static int64_t rows(const vgi::ProcessParams& params) {
        return std::max<int64_t>(1, params.arguments.const_int64(0).value_or(1));
    }
};

std::shared_ptr<arrow::Schema> range_schema() {
    return arrow::schema({vgi::partition_field("key", arrow::int64()),
                          arrow::field("value", arrow::int64(), /*nullable=*/true)});
}

// `disjoint_range_partitioned(partitions, rows_per_partition := 10)` and
// `overlapping_range_partitioned(…)` — the wire paths for two kinds DuckDB
// declares but has no consumer for yet, so a GROUP BY over either still plans
// as a hash aggregate. Running them at all is what proves the enum values
// round-trip, since the engine rejects a kind it does not recognize.
class RangePartitioned : public vgi::TableFunction {
public:
    RangePartitioned(std::string name, std::string description, std::string kind,
                     int64_t stride)
        : name_(std::move(name)),
          description_(std::move(description)),
          kind_(std::move(kind)),
          stride_(stride) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        auto md = fixture_metadata(description_, {"generator", "partitioning"});
        md.partition_kind = kind_;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("partitions", 0, "int64", "Number of partitions"),
                vgi::ArgSpec::named("rows_per_partition", "int64", "Rows per partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return range_schema();
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 4; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto partitions = params.arguments.const_int64(0)) {
            estimate.estimate = *partitions * rows(params);
            estimate.max = *partitions * rows(params);
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        push_partitions(params,
                        std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)),
                        rows(params));
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<RangeChunks>(params.output_schema, params.storage,
                                             params.execution_id, range_schema(), stride_);
    }

private:
    static int64_t rows(const vgi::ProcessParams& params) {
        return std::max<int64_t>(1, params.arguments.named_int64("rows_per_partition")
                                        .value_or(10));
    }

    std::string name_;
    std::string description_;
    std::string kind_;
    int64_t stride_;
};

}  // namespace

void register_partitioned(vgi::Worker& worker) {
    // The `increment` form promises nothing about order — it exists to be
    // scanned in parallel, and a promise it cannot keep would let the engine
    // skip a sort the answer needs. The other three differ from it, and from
    // each other, only in the mode they declare, which is what makes the
    // planner's differing treatment attributable to the declaration alone.
    worker.register_table(std::make_shared<PartitionedSequence>(
        "partitioned_sequence", "Generates a partitioned sequence for multi-worker execution",
        /*takes_increment=*/true, /*order=*/""));
    worker.register_table(std::make_shared<PartitionedSequence>(
        "partitioned_preserves_order",
        "Generates a partitioned sequence whose emission order the engine may rely on",
        /*takes_increment=*/false, vgi::order_preservations::kPreservesOrder));
    worker.register_table(std::make_shared<PartitionedSequence>(
        "partitioned_no_order_guarantee",
        "Generates a partitioned sequence whose emission order means nothing",
        /*takes_increment=*/false, vgi::order_preservations::kNoOrderGuarantee));
    worker.register_table(std::make_shared<PartitionedSequence>(
        "partitioned_fixed_order",
        "Generates a partitioned sequence the engine must not reorder or parallelise",
        /*takes_increment=*/false, vgi::order_preservations::kFixedOrder));
    worker.register_table(std::make_shared<PartitionedBatchIndex>());
    worker.register_table(std::make_shared<PartitionedBatchIndexMarked>());
    worker.register_table(std::make_shared<FilterEchoPartitioned>());
    worker.register_table(std::make_shared<CountryPartitionedSales>());
    worker.register_table(std::make_shared<RegionYearPartitioned>());
    worker.register_table(std::make_shared<PartitionedWithExplicitOverride>());
    worker.register_table(std::make_shared<RangePartitioned>(
        "disjoint_range_partitioned",
        "Integer ranges that never overlap between chunks; declares DISJOINT_PARTITIONS",
        vgi::partition_kinds::kDisjointPartitions, /*stride=*/1000));
    worker.register_table(std::make_shared<RangePartitioned>(
        "overlapping_range_partitioned",
        "Integer ranges that may share keys between chunks; declares OVERLAPPING_PARTITIONS",
        vgi::partition_kinds::kOverlappingPartitions, /*stride=*/500));
}

}  // namespace example
