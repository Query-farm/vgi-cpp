// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Result-cache fixtures whose output is SINGLE_VALUE partitioned.
//
// Two distinct things are under test here. `cache_partitioned` only carries
// `vgi_partition_values#b64` through the spill blob, where a misframed length
// misaligns the streaming TOC seek. The rest additionally advertise
// `vgi.cache.partition_scope`, which has the engine store each partition as its
// own entry so a later `=`/`IN` scan is served without reaching the worker.
//
// Unlike the whole-result advertisement in `cache.cpp`, the partition-scope
// opt-in rides **every** batch. A fall-through scan whose leading partition is
// filtered to zero rows would otherwise never advertise, and the engine would
// silently stop caching per partition from then on.

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/partition.h>
#include <vgi/worker.h>

namespace example {
namespace {

constexpr int64_t kDefaultTtlSeconds = 300;

// Sorted, because `vgi_result_cache()` lists per-partition entries by label and
// the tests read them in that order.
const std::vector<std::string> kCountries = {"AU", "BR", "CA", "FR", "US"};

// The parallel fixture's partitions. The trailing absent value is a genuine
// NULL partition — SINGLE_VALUE permits one, and `IS NULL` is deliberately not
// enumerable, so it must fall through to the worker rather than be served.
const std::vector<std::optional<std::string>> kParallelCountries = {
    std::string("AU"), std::string("CA"), std::string("US"), std::nullopt};

// Two partitions for the projection fixture: enough to prove the split buckets
// correctly, few enough that its expected rows fit in the test inline.
const std::vector<std::string> kProjCountries = {"CA", "US"};

// Years are non-contiguous on purpose: DuckDB rewrites `year IN (2020, 2021)`
// into a BETWEEN range, which is not enumerable, and the cross-product path
// would then never be exercised.
constexpr std::array<int64_t, 2> kYears = {2020, 2022};
const std::vector<std::string> kRegions = {"EU", "US"};

// Each partition's values start at `index * kPartitionStride`, so a value alone
// names the partition it came from — a mis-bucketed split shows up as wrong
// numbers rather than a plausible total.
constexpr int64_t kPartitionStride = 1000000;

vgi::FunctionMetadata cache_metadata(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = {"generator", "cache", "testing"};
    md.tags = {{"category", "cache"}};
    md.partition_kind = vgi::partition_kinds::kSingleValuePartitions;
    return md;
}

// What every per-partition fixture declares on top of `cache_metadata`.
//
// `filter_pushdown` is what makes the client able to enumerate the requested
// partitions at all, and `auto_apply_filters` is what keeps a fall-through
// worker scan row-exact: DuckDB does not re-apply a predicate it pushed down.
vgi::FunctionMetadata partition_scope_metadata(std::string description) {
    auto md = cache_metadata(std::move(description));
    md.filter_pushdown = true;
    md.auto_apply_filters = true;
    return md;
}

vgi::CacheControl partition_scope_control() {
    vgi::CacheControl control;
    control.ttl_seconds = kDefaultTtlSeconds;
    control.partition_scope = true;
    return control;
}

std::shared_ptr<arrow::Array> finish(arrow::ArrayBuilder& builder) {
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> i64_range(int64_t from, int64_t count) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(count);
    for (int64_t i = 0; i < count; ++i) (void)builder.Append(from + i);
    return finish(builder);
}

std::shared_ptr<arrow::Array> i64_repeated(int64_t value, int64_t count) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(count);
    for (int64_t i = 0; i < count; ++i) (void)builder.Append(value);
    return finish(builder);
}

std::shared_ptr<arrow::Array> utf8_repeated(const std::optional<std::string>& value,
                                            int64_t count) {
    arrow::StringBuilder builder;
    for (int64_t i = 0; i < count; ++i) {
        if (value) {
            (void)builder.Append(*value);
        } else {
            (void)builder.AppendNull();
        }
    }
    return finish(builder);
}

std::shared_ptr<arrow::Schema> country_sales_schema() {
    return arrow::schema({vgi::partition_field("country", arrow::utf8()),
                          arrow::field("sales", arrow::int64(), /*nullable=*/true)});
}

// `{country, sales}` built from `schema` rather than from a fixed shape, so a
// pushed projection that dropped a column simply does not build it.
std::shared_ptr<arrow::RecordBatch> country_sales_batch(
    const std::shared_ptr<arrow::Schema>& schema, const std::optional<std::string>& country,
    int64_t index, int64_t rows) {
    const int64_t base = index * kPartitionStride;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (int i = 0; i < schema->num_fields(); ++i) {
        const auto& name = schema->field(i)->name();
        if (name == "country") {
            columns.push_back(utf8_repeated(country, rows));
        } else if (name == "sales") {
            columns.push_back(i64_range(base, rows));
        } else if (name == "extra") {
            columns.push_back(i64_range(base + 500, rows));
        } else {
            throw std::runtime_error("cache partition fixture: unknown column " + name);
        }
    }
    return arrow::RecordBatch::Make(schema, rows, columns);
}

// Partition values computed from a synthetic one-row batch instead of the
// emitted one.
//
// Two cases need it: a projected-away partition column is absent from the batch
// entirely, so auto-extraction has nothing to read; and an all-NULL column would
// take its scalar type from the data rather than from the declaration.
std::map<std::string, std::string> explicit_country_values(
    const std::optional<std::string>& country) {
    auto schema = arrow::schema({vgi::partition_field("country", arrow::utf8())});
    auto batch = arrow::RecordBatch::Make(schema, 1, {utf8_repeated(country, 1)});
    return vgi::partition_metadata(schema, batch);
}

void merge(std::map<std::string, std::string>& into,
           const std::map<std::string, std::string>& from) {
    for (const auto& [key, value] : from) into[key] = value;
}

// `cache_partitioned(rows_per_country)` — one country per batch, carrying
// partition values but *not* the per-partition opt-in.
//
// The only cacheable fixture with non-empty `pv_bytes`, which is what makes it
// the one that can catch a misframed length in the spilled blob.
class CachePartitioned : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_partitioned"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata(
            "Cacheable single-value-partitioned result (partition_values through the "
            "spill blob)");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows_per_country", 0, "int64",
                                           "Rows per country partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return country_sales_schema();
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        const int64_t rows =
            rows_per_country(params) * static_cast<int64_t>(kCountries.size());
        estimate.estimate = rows;
        estimate.max = rows;
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema, rows_per_country(params));
    }

private:
    static int64_t rows_per_country(const vgi::ProcessParams& params) {
        return std::max<int64_t>(0, params.arguments.const_int64(0).value_or(1));
    }

    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows)
            : schema_(std::move(schema)), rows_(rows) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (index_ >= static_cast<int64_t>(kCountries.size())) return nullptr;
            auto batch = country_sales_batch(schema_, kCountries[index_], index_, rows_);
            metadata_ = vgi::partition_metadata(schema_, batch);
            // Whole-result advertisement: first batch only.
            if (index_ == 0) {
                vgi::CacheControl control;
                control.ttl_seconds = kDefaultTtlSeconds;
                merge(metadata_, control.to_metadata());
            }
            ++index_;
            return batch;
        }

        std::map<std::string, std::string> last_metadata() const override {
            return metadata_;
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t rows_;
        int64_t index_ = 0;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_partition_scope(rows_per_country)` — the baseline per-partition
// fixture: one worker, five countries, one batch each.
class CachePartitionScope : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_partition_scope"; }

    vgi::FunctionMetadata metadata() const override {
        return partition_scope_metadata(
            "Per-partition cacheable single-value-partitioned result "
            "(vgi.cache.partition_scope)");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows_per_country", 0, "int64",
                                           "Rows per country partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return country_sales_schema();
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        const int64_t rows =
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(1)) *
            static_cast<int64_t>(kCountries.size());
        estimate.estimate = rows;
        estimate.max = rows;
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(1)));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows)
            : schema_(std::move(schema)), rows_(rows) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (index_ >= static_cast<int64_t>(kCountries.size())) return nullptr;
            auto batch = country_sales_batch(schema_, kCountries[index_], index_, rows_);
            metadata_ = vgi::partition_metadata(schema_, batch);
            merge(metadata_, partition_scope_control().to_metadata());
            ++index_;
            return batch;
        }

        std::map<std::string, std::string> last_metadata() const override {
            return metadata_;
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t rows_;
        int64_t index_ = 0;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_partition_parallel(rows_per_country)` — the same per-partition
// contract fanned across workers, plus a NULL partition.
//
// One queue item per partition, so a `threads > 1` scan draws partitions from
// several worker processes and the split at commit has to bucket batches taken
// from more than one capture substream.
class CachePartitionParallel : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_partition_parallel"; }

    vgi::FunctionMetadata metadata() const override {
        return partition_scope_metadata(
            "Per-partition cacheable; work-queue fan-out (parallel capture); one NULL "
            "partition");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows_per_country", 0, "int64",
                                           "Rows per country partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return country_sales_schema();
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 8; }

    void on_init(const vgi::ProcessParams& params) const override {
        std::vector<std::string> items;
        for (size_t i = 0; i < kParallelCountries.size(); ++i) {
            items.push_back(std::to_string(i));
        }
        params.storage->queue_push(params.execution_id, items);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(
            params.output_schema, params.storage, params.execution_id,
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(1)));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema,
                 std::shared_ptr<vgi::FunctionStorage> storage, std::string execution_id,
                 int64_t rows)
            : schema_(std::move(schema)),
              storage_(std::move(storage)),
              execution_id_(std::move(execution_id)),
              rows_(rows) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            auto item = storage_->queue_pop(execution_id_);
            if (!item) return nullptr;  // queue drained: this worker is done
            const int64_t index = std::stoll(*item);
            const auto& country = kParallelCountries[static_cast<size_t>(index)];
            auto batch = country_sales_batch(schema_, country, index, rows_);
            metadata_ = explicit_country_values(country);
            merge(metadata_, partition_scope_control().to_metadata());
            return batch;
        }

        std::map<std::string, std::string> last_metadata() const override {
            return metadata_;
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        std::shared_ptr<vgi::FunctionStorage> storage_;
        std::string execution_id_;
        int64_t rows_;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_partition_multicol(rows_per_partition)` — two partition columns, so
// the client has to enumerate the cross product `region IN (…) × year IN (…)`
// and canonicalize a two-column tuple rather than a scalar.
class CachePartitionMultiCol : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_partition_multicol"; }

    vgi::FunctionMetadata metadata() const override {
        return partition_scope_metadata(
            "Per-partition cacheable over (region, year) SINGLE_VALUE partition columns");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows_per_partition", 0, "int64",
                                           "Rows per (region, year) partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({vgi::partition_field("region", arrow::utf8()),
                              vgi::partition_field("year", arrow::int64()),
                              arrow::field("amount", arrow::int64(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(1)));
    }

private:
    // Region-major, matching the Python fixture the engine's expected labels
    // were written against.
    static constexpr int64_t kAmountStride = 1000;

    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows)
            : schema_(std::move(schema)), rows_(rows) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            const size_t index = static_cast<size_t>(index_);
            if (index >= kRegions.size() * kYears.size()) return nullptr;
            const auto& region = kRegions[index / kYears.size()];
            const int64_t year = kYears[index % kYears.size()];

            std::vector<std::shared_ptr<arrow::Array>> columns;
            for (int i = 0; i < schema_->num_fields(); ++i) {
                const auto& name = schema_->field(i)->name();
                if (name == "region") {
                    columns.push_back(utf8_repeated(region, rows_));
                } else if (name == "year") {
                    columns.push_back(i64_repeated(year, rows_));
                } else {
                    columns.push_back(i64_range(index_ * kAmountStride, rows_));
                }
            }
            auto batch = arrow::RecordBatch::Make(schema_, rows_, columns);
            metadata_ = vgi::partition_metadata(schema_, batch);
            merge(metadata_, partition_scope_control().to_metadata());
            ++index_;
            return batch;
        }

        std::map<std::string, std::string> last_metadata() const override {
            return metadata_;
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t rows_;
        int64_t index_ = 0;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_partition_proj(rows_per_country)` — per-partition caching under
// projection pushdown.
//
// `extra` exists only to be projected away while `country` stays pushable. The
// partition values are always explicit, because the interesting case is the one
// where `country` itself is projected out and the split has nothing in the batch
// to bucket on.
class CachePartitionProj : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_partition_proj"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = partition_scope_metadata(
            "Per-partition cacheable with projection pushdown + explicit partition_values");
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows_per_country", 0, "int64",
                                           "Rows per country partition")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({vgi::partition_field("country", arrow::utf8()),
                              arrow::field("sales", arrow::int64(), /*nullable=*/true),
                              arrow::field("extra", arrow::int64(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(1)));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows)
            : schema_(std::move(schema)), rows_(rows) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (index_ >= static_cast<int64_t>(kProjCountries.size())) return nullptr;
            const auto& country = kProjCountries[index_];
            auto batch = country_sales_batch(schema_, country, index_, rows_);
            metadata_ = explicit_country_values(country);
            merge(metadata_, partition_scope_control().to_metadata());
            ++index_;
            return batch;
        }

        std::map<std::string, std::string> last_metadata() const override {
            return metadata_;
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t rows_;
        int64_t index_ = 0;
        std::map<std::string, std::string> metadata_;
    };
};

}  // namespace

void register_cache_partitions(vgi::Worker& worker) {
    worker.register_table(std::make_shared<CachePartitioned>());
    worker.register_table(std::make_shared<CachePartitionScope>());
    worker.register_table(std::make_shared<CachePartitionParallel>());
    worker.register_table(std::make_shared<CachePartitionMultiCol>());
    worker.register_table(std::make_shared<CachePartitionProj>());
}

}  // namespace example
