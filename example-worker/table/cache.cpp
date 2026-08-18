// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Result-cache fixtures — table producers that advertise `vgi.cache.*`.
//
// Each returns a small deterministic result and folds its cache-control
// metadata onto the **first** emitted batch. First rather than every batch
// because the advertisement describes the whole result, and repeating it would
// let a later batch contradict an earlier one.

#include <atomic>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Long enough that freshness never lapses part-way through a test.
constexpr int64_t kDefaultTtlSeconds = 300;

// Bumped once per *real* invocation of the nonce fixtures.
//
// A pooled worker keeps this across calls, so a result served from the engine's
// cache never advances it. That is precisely the hit/miss signal the tests
// assert on — the counter is the observable, not an implementation detail.
std::atomic<int64_t> nonce_counter{0};

int64_t next_nonce() {
    return nonce_counter.fetch_add(1, std::memory_order_relaxed);
}

vgi::FunctionMetadata cache_metadata(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = {"generator", "cache", "testing"};
    md.tags = {{"category", "cache"}};
    return md;
}

std::shared_ptr<arrow::Schema> single_i64_schema(const std::string& name) {
    return arrow::schema({arrow::field(name, arrow::int64(), /*nullable=*/true)});
}

std::shared_ptr<arrow::RecordBatch> i64_batch(const std::shared_ptr<arrow::Schema>& schema,
                                              int64_t from, int64_t to) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(to - from);
    for (int64_t i = from; i < to; ++i) (void)builder.Append(i);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return arrow::RecordBatch::Make(schema, array->length(), {array});
}

// Emits `rows` values in chunks, advertising its cache control on the first.
class Countdown : public vgi::TableProducer {
public:
    Countdown(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size,
              vgi::CacheControl cache_control)
        : schema_(std::move(schema)),
          remaining_(rows < 0 ? 0 : rows),
          batch_size_(batch_size),
          cache_control_(std::move(cache_control)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (remaining_ <= 0) return nullptr;
        const bool first = index_ == 0;
        const int64_t size = std::min(remaining_, batch_size_);
        auto batch = i64_batch(schema_, index_, index_ + size);
        metadata_ = first ? cache_control_.to_metadata() : std::map<std::string, std::string>{};
        index_ += size;
        remaining_ -= size;
        return batch;
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    std::shared_ptr<arrow::Schema> schema_;
    int64_t remaining_;
    int64_t batch_size_;
    vgi::CacheControl cache_control_;
    int64_t index_ = 0;
    std::map<std::string, std::string> metadata_;
};

// Emits exactly one row holding `value`.
class OneRow : public vgi::TableProducer {
public:
    OneRow(std::shared_ptr<arrow::Schema> schema, int64_t value, vgi::CacheControl cache_control)
        : schema_(std::move(schema)), value_(value), cache_control_(std::move(cache_control)) {}

    void on_conditional_request(const std::optional<std::string>& if_none_match,
                                const std::optional<std::string>&) override {
        // The answer is unchanged whenever the caller quotes back the etag
        // this fixture always emits, so a match means the stored payload still
        // stands.
        confirmed_ = cache_control_.etag && if_none_match && *if_none_match == *cache_control_.etag;
    }

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (done_) return nullptr;
        done_ = true;

        auto control = cache_control_;
        // A zero-row batch, because the engine serves the bytes it already
        // holds: sending the rows again would be the answer it asked not to
        // get.
        const int64_t rows = confirmed_ ? 0 : 1;
        control.not_modified = confirmed_;

        arrow::Int64Builder builder;
        if (!confirmed_) (void)builder.Append(value_);
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        metadata_ = control.to_metadata();
        return arrow::RecordBatch::Make(schema_, rows, {array});
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    std::shared_ptr<arrow::Schema> schema_;
    int64_t value_;
    vgi::CacheControl cache_control_;
    bool confirmed_ = false;
    bool done_ = false;
    std::map<std::string, std::string> metadata_;
};

// The shared shape of most of these fixtures: n rows named by one column, with
// a cache advertisement the subclass chooses.
class CachedNumbers : public vgi::TableFunction {
public:
    // `positional_rows` decides how the row count is supplied. A table-backed
    // fixture is scanned with no arguments at all, so its count must be named
    // and defaulted; one called directly as `f(3)` needs it positional. The
    // engine resolves on arity, so the two cannot be the same declaration.
    CachedNumbers(std::string name, std::string description, std::string column, bool takes_ttl,
                  bool positional_rows = false)
        : name_(std::move(name)),
          description_(std::move(description)),
          column_(std::move(column)),
          takes_ttl_(takes_ttl),
          positional_rows_(positional_rows) {}

    std::string name() const override { return name_; }
    vgi::FunctionMetadata metadata() const override { return cache_metadata(description_); }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        std::vector<vgi::ArgSpec> specs{
            positional_rows_
                ? vgi::ArgSpec::constant_arg("n", 0, "int64", "Number of rows to generate")
                : vgi::ArgSpec::named("n", "int64", "Number of rows to generate")};
        if (takes_ttl_) {
            specs.push_back(vgi::ArgSpec::named("ttl", "int64", "Cache TTL in seconds"));
        }
        return specs;
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema(column_);
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        const int64_t rows = row_count(params);
        estimate.estimate = rows;
        estimate.max = rows;
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Countdown>(params.output_schema, row_count(params), 1000,
                                           cache_control(params));
    }

protected:
    virtual vgi::CacheControl cache_control(const vgi::ProcessParams& params) const {
        vgi::CacheControl control;
        control.ttl_seconds = params.arguments.named_int64("ttl").value_or(kDefaultTtlSeconds);
        return control;
    }

private:
    int64_t row_count(const vgi::ProcessParams& params) const {
        if (positional_rows_) return params.arguments.const_int64(0).value_or(10);
        return params.arguments.named_int64("n").value_or(10);
    }

    std::string name_;
    std::string description_;
    std::string column_;
    bool takes_ttl_;
    bool positional_rows_;
};

// `cache_no_store(n := 10)` — emits rows but forbids caching, so every scan
// re-invokes the worker.
class CacheNoStore : public CachedNumbers {
public:
    CacheNoStore()
        : CachedNumbers("cache_no_store", "Emits n rows but advertises no_store (never cached)",
                        "n",
                        /*takes_ttl=*/false) {}

protected:
    vgi::CacheControl cache_control(const vgi::ProcessParams&) const override {
        vgi::CacheControl control;
        control.no_store = true;
        return control;
    }
};

// `cache_scoped_txn(n := 10)` — cacheable only within one transaction.
//
// Every row carries the invocation's nonce alongside its index, so the
// transaction boundary is provable from the data: a second scan in the same
// transaction returns the same nonce (served) and one in a new transaction a
// fresh one (recomputed), without reading a log.
class CacheScopedTxn : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_scoped_txn"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("Emits n rows cacheable only within the transaction");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("n", "int64", "Number of rows to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true),
                              arrow::field("nonce", arrow::int64(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        vgi::CacheControl control;
        control.ttl_seconds = kDefaultTtlSeconds;
        control.scope = "transaction";
        // Drawn in init, which the engine only reaches on a miss.
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.named_int64("n").value_or(10)), next_nonce(),
            std::move(control));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t nonce,
                 vgi::CacheControl control)
            : schema_(std::move(schema)),
              rows_(rows),
              nonce_(nonce),
              control_(std::move(control)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (emitted_) return nullptr;
            emitted_ = true;
            arrow::Int64Builder indices;
            arrow::Int64Builder nonces;
            (void)indices.Reserve(rows_);
            (void)nonces.Reserve(rows_);
            for (int64_t row = 0; row < rows_; ++row) {
                (void)indices.Append(row);
                (void)nonces.Append(nonce_);
            }
            std::shared_ptr<arrow::Array> index_array;
            std::shared_ptr<arrow::Array> nonce_array;
            (void)indices.Finish(&index_array);
            (void)nonces.Finish(&nonce_array);
            metadata_ = control_.to_metadata();
            return arrow::RecordBatch::Make(schema_, rows_, {index_array, nonce_array});
        }

        std::map<std::string, std::string> last_metadata() const override { return metadata_; }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t rows_;
        int64_t nonce_;
        vgi::CacheControl control_;
        bool emitted_ = false;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_nonce()` — one row whose value changes on every real invocation.
class CacheNonce : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_nonce"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("Emits one row with a per-invocation nonce; cacheable");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("nonce");
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        vgi::CacheControl control;
        control.ttl_seconds = kDefaultTtlSeconds;
        // The nonce is drawn here, in init, which the engine only reaches on a
        // cache miss — so a served hit leaves it unchanged.
        return std::make_unique<OneRow>(params.output_schema, next_nonce(), std::move(control));
    }
};

// `cache_multicol(n := 4, ttl := 300)` — three columns, so the cache has to
// round-trip a wider result than a single int64.
class MultiCol : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_multicol"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("Emits n rows of (a, b, c); cacheable, multi-column");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("n", "int64", "Number of rows to generate"),
                vgi::ArgSpec::named("ttl", "int64", "Cache TTL in seconds")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("a", arrow::int64(), true),
                              arrow::field("b", arrow::int64(), true),
                              arrow::field("c", arrow::int64(), true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        vgi::CacheControl control;
        control.ttl_seconds = params.arguments.named_int64("ttl").value_or(kDefaultTtlSeconds);
        return std::make_unique<Rows>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.named_int64("n").value_or(4)),
            std::move(control));
    }

private:
    // a = i, b = i * 10, c = i * 100 — distinct per column so a transposition
    // in the cache path shows up as wrong values rather than plausible ones.
    class Rows : public vgi::TableProducer {
    public:
        Rows(std::shared_ptr<arrow::Schema> schema, int64_t rows, vgi::CacheControl control)
            : schema_(std::move(schema)), rows_(rows), control_(std::move(control)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (done_) return nullptr;
            done_ = true;
            std::vector<std::shared_ptr<arrow::Array>> columns;
            for (int64_t scale : {int64_t{1}, int64_t{10}, int64_t{100}}) {
                arrow::Int64Builder builder;
                (void)builder.Reserve(rows_);
                for (int64_t i = 0; i < rows_; ++i) (void)builder.Append(i * scale);
                std::shared_ptr<arrow::Array> array;
                (void)builder.Finish(&array);
                columns.push_back(array);
            }
            metadata_ = control_.to_metadata();
            return arrow::RecordBatch::Make(schema_, rows_, columns);
        }

        std::map<std::string, std::string> last_metadata() const override { return metadata_; }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t rows_;
        vgi::CacheControl control_;
        bool done_ = false;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_big(rows := 5000)` — many small batches, so multi-batch capture and
// the cache's size ceiling are both exercised.
class CacheBig : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_big"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("Emits many small batches totaling `rows` rows; cacheable");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("rows", "int64", "Number of rows to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("n");
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        const int64_t rows = params.arguments.named_int64("rows").value_or(5000);
        estimate.estimate = rows;
        estimate.max = rows;
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        vgi::CacheControl control;
        control.ttl_seconds = kDefaultTtlSeconds;
        return std::make_unique<Countdown>(params.output_schema,
                                           params.arguments.named_int64("rows").value_or(5000),
                                           1000, std::move(control));
    }
};

// `cache_revalidatable()` — advertises that the worker can answer a
// conditional request cheaply, which is what makes the engine send one.
class CacheRevalidatable : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_revalidatable"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("Emits one nonce row; revalidatable with an ETag");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("nonce");
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        // ttl 0 beside a validator is the "no-cache" semantic: the engine keeps
        // the bytes but marks them stale at once, so every repeat asks rather
        // than assumes. A non-zero TTL here would make the entry fresh and the
        // conditional path would never run.
        vgi::CacheControl control;
        control.ttl_seconds = 0;
        control.revalidatable = true;
        // A constant ETag: the fixture's result never actually changes, so
        // every revalidation is expected to confirm freshness.
        control.etag = kEtag;
        return std::make_unique<OneRow>(params.output_schema, next_nonce(), std::move(control));
    }

private:
    // Spelled as the canonical Python fixture spells it, so a conditional
    // request recorded against one implementation matches the other.
    static constexpr const char* kEtag = "\"rev-v1\"";
};

// `cache_parallel(rows, batch_size := 24000)` — the one cache fixture that
// runs across several workers.
//
// The primary init divides `[0, rows)` into chunks and pushes them onto the
// execution's shared queue; every worker then pops chunks until the queue is
// empty. Values are the plain sequence, so COUNT and SUM hold whatever order
// the chunks were claimed in — which is what makes the test independent of
// how the engine actually distributed them.
class CacheParallel : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_parallel"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata(
            "Multi-worker cacheable sequence (one substream per worker); parallel-capture "
            "fixture");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows", 0, "int64", "Total number of rows to generate"),
                vgi::ArgSpec::named("batch_size", "int64", "Rows per output batch")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("v");
    }

    int64_t max_workers(const vgi::ProcessParams&) const override { return 8; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto rows = params.arguments.const_int64(0)) {
            estimate.estimate = *rows;
            estimate.max = *rows;
        }
        return estimate;
    }

    void on_init(const vgi::ProcessParams& params) const override {
        const int64_t rows = std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        // A bounded chunk count rather than a fixed chunk size: the cost of
        // fanning out should scale with the number of workers, not the number
        // of rows.
        const int64_t chunk = std::max<int64_t>(1, (rows + kMaxChunks - 1) / kMaxChunks);

        std::vector<std::string> items;
        for (int64_t start = 0; start < rows; start += chunk) {
            const int64_t end = std::min(start + chunk, rows);
            items.push_back(std::to_string(start) + ":" + std::to_string(end));
        }
        params.storage->queue_push(params.execution_id, items);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<QueueProducer>(
            params.output_schema, params.storage, params.execution_id,
            std::max<int64_t>(1, params.arguments.named_int64("batch_size").value_or(24000)));
    }

private:
    // ~24 chunks whatever the row count.
    static constexpr int64_t kMaxChunks = 24;

    class QueueProducer : public vgi::TableProducer {
    public:
        QueueProducer(std::shared_ptr<arrow::Schema> schema,
                      std::shared_ptr<vgi::FunctionStorage> storage, std::string execution_id,
                      int64_t batch_size)
            : schema_(std::move(schema)),
              storage_(std::move(storage)),
              execution_id_(std::move(execution_id)),
              batch_size_(batch_size) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (position_ >= end_) {
                auto item = storage_->queue_pop(execution_id_);
                if (!item) return nullptr;  // queue drained: this worker is done
                const auto colon = item->find(':');
                if (colon == std::string::npos) return nullptr;
                position_ = std::strtoll(item->c_str(), nullptr, 10);
                end_ = std::strtoll(item->c_str() + colon + 1, nullptr, 10);
                if (position_ >= end_) return next_batch();
            }

            const int64_t n = std::min(batch_size_, end_ - position_);
            auto batch = i64_batch(schema_, position_, position_ + n);
            // The advertisement rides the first batch this worker emits, not
            // the first of each chunk.
            if (!advertised_) {
                advertised_ = true;
                vgi::CacheControl control;
                control.ttl_seconds = kDefaultTtlSeconds;
                metadata_ = control.to_metadata();
            } else {
                metadata_.clear();
            }
            position_ += n;
            return batch;
        }

        std::map<std::string, std::string> last_metadata() const override { return metadata_; }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        std::shared_ptr<vgi::FunctionStorage> storage_;
        std::string execution_id_;
        int64_t batch_size_;
        int64_t position_ = 0;
        int64_t end_ = 0;
        bool advertised_ = false;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_poison(n)` — a cacheable first batch, then an error.
//
// The point is that a *partial* result must never be cached: a scan that
// errors mid-stream has to leave nothing behind, or the next query serves half
// an answer with no indication it is half.
class CachePoison : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_poison"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata(
            "Cacheable first batch then a mid-stream error "
            "(never-partial check)");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("n", "int64", "Rows in the first batch")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("n");
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.named_int64("n").value_or(4)));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows)
            : schema_(std::move(schema)), rows_(rows) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (emitted_) {
                throw std::runtime_error("cache_poison: intentional mid-stream failure");
            }
            emitted_ = true;
            vgi::CacheControl control;
            control.ttl_seconds = kDefaultTtlSeconds;
            metadata_ = control.to_metadata();
            return i64_batch(schema_, 0, rows_);
        }

        std::map<std::string, std::string> last_metadata() const override { return metadata_; }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t rows_;
        bool emitted_ = false;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_projection()` — three columns, built only where the bound schema names
// them, so a test can see what projection pushdown asked for.
class CacheProjection : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_projection"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = cache_metadata("3-column projection-pushdown generator; cacheable");
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("a", arrow::int64(), true),
                              arrow::field("b", arrow::int64(), true),
                              arrow::field("c", arrow::int64(), true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema);
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        explicit Producer(std::shared_ptr<arrow::Schema> schema) : schema_(std::move(schema)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (emitted_) return nullptr;
            emitted_ = true;

            // Scales differ per column, and no value repeats across them, so a
            // cross-served projection shows up as wrong values rather than
            // plausible ones.
            std::vector<std::shared_ptr<arrow::Array>> columns;
            for (int i = 0; i < schema_->num_fields(); ++i) {
                const auto& name = schema_->field(i)->name();
                const int64_t scale = name == "b" ? 10 : (name == "c" ? 100 : 1);
                arrow::Int64Builder builder;
                (void)builder.Reserve(kRows);
                for (int64_t row = 1; row <= kRows; ++row) (void)builder.Append(row * scale);
                std::shared_ptr<arrow::Array> array;
                (void)builder.Finish(&array);
                columns.push_back(array);
            }
            vgi::CacheControl control;
            control.ttl_seconds = kDefaultTtlSeconds;
            metadata_ = control.to_metadata();
            return arrow::RecordBatch::Make(schema_, kRows, columns);
        }

        std::map<std::string, std::string> last_metadata() const override { return metadata_; }

    private:
        static constexpr int64_t kRows = 3;

        std::shared_ptr<arrow::Schema> schema_;
        bool emitted_ = false;
        std::map<std::string, std::string> metadata_;
    };
};

}  // namespace

void register_cache(vgi::Worker& worker) {
    worker.register_table(std::make_shared<CachePoison>());
    worker.register_table(std::make_shared<CacheProjection>());
    worker.register_table(std::make_shared<CacheParallel>());
    worker.register_table(std::make_shared<MultiCol>());
    // Backs `ex.data.cache_multicol` and nothing else, so it is not part of
    // the function surface the suite counts.
    worker.hide_function("cache_multicol");
    worker.register_table(std::make_shared<CacheBig>());
    worker.register_table(std::make_shared<CacheRevalidatable>());
    worker.register_table(std::make_shared<CachedNumbers>(
        "cacheable_numbers", "Emits n rows [0..n) and advertises a cache TTL", "n",
        /*takes_ttl=*/true));
    worker.register_table(std::make_shared<CacheNonce>());
    worker.register_table(std::make_shared<CacheNoStore>());
    worker.register_table(std::make_shared<CacheScopedTxn>());
    // Positional and untagged, unlike its neighbours: the scaling tests call
    // `cache_bench(200000)` directly and need the row count they asked for.
    worker.register_table(std::make_shared<CachedNumbers>(
        "cache_bench", "Emits `rows` int64 rows (positional arg); cacheable", "v",
        /*takes_ttl=*/false, /*positional_rows=*/true));
}

}  // namespace example
