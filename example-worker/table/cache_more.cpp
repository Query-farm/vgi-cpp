// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Result-cache fixtures beyond the flat int64 producers in `cache.cpp`.
//
// Every one reaches a part of the cache path a plain sequence cannot: nested
// and NULL-bearing columns through the spill blob, a pushed predicate in the
// key, source order across a replay, an unresolvable external pointer, an
// AT-pinned version, a schema-scoped key, and the exchange-mode entries.
//
// As in `cache.cpp`, a whole-result advertisement rides the **first** emitted
// batch only: it describes the result, and repeating it would let a later
// batch contradict an earlier one.

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/util.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/decimal.h>

#include <vgi_rpc/metadata.h>

#include <vgi/worker.h>

namespace example {
namespace {

// Long enough that freshness never lapses part-way through a test.
constexpr int64_t kDefaultTtlSeconds = 300;

vgi::FunctionMetadata cache_metadata(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = {"generator", "cache", "testing"};
    md.tags = {{"category", "cache"}};
    return md;
}

vgi::CacheControl default_ttl() {
    vgi::CacheControl control;
    control.ttl_seconds = kDefaultTtlSeconds;
    return control;
}

std::shared_ptr<arrow::Schema> single_i64_schema(const std::string& name) {
    return arrow::schema({arrow::field(name, arrow::int64(), /*nullable=*/true)});
}

std::shared_ptr<arrow::Array> finish(arrow::ArrayBuilder& builder) {
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> i64_column(int64_t from, int64_t to) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(to - from);
    for (int64_t i = from; i < to; ++i) (void)builder.Append(i);
    return finish(builder);
}

std::shared_ptr<arrow::Array> utf8_column(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) (void)builder.Append(value);
    return finish(builder);
}

// Emits `[0, rows)` in fixed-size batches, advertising on the first.
class Sequence : public vgi::TableProducer {
public:
    Sequence(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size)
        : schema_(std::move(schema)),
          remaining_(rows < 0 ? 0 : rows),
          batch_size_(batch_size < 1 ? 1 : batch_size) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (remaining_ <= 0) return nullptr;
        const int64_t size = std::min(remaining_, batch_size_);
        auto column = i64_column(index_, index_ + size);
        metadata_ =
            index_ == 0 ? default_ttl().to_metadata() : std::map<std::string, std::string>{};
        index_ += size;
        remaining_ -= size;
        return arrow::RecordBatch::Make(schema_, size, {column});
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    std::shared_ptr<arrow::Schema> schema_;
    int64_t remaining_;
    int64_t batch_size_;
    int64_t index_ = 0;
    std::map<std::string, std::string> metadata_;
};

// Emits one prepared batch, advertising on it.
class OneShot : public vgi::TableProducer {
public:
    explicit OneShot(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (!batch_) return nullptr;
        metadata_ = default_ttl().to_metadata();
        return std::move(batch_);
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    std::shared_ptr<arrow::RecordBatch> batch_;
    std::map<std::string, std::string> metadata_;
};

// `cache_types(rows)` — nested, wide and NULL-bearing columns.
//
// Every other cacheable fixture emits flat int64, so the spill blob's
// validity bitmaps and its variable-length and nested buffers are otherwise
// only exercised on fixed-width data. Here a serve has to reassemble all of
// them byte-identically.

// Rows where `j % 5 == 0` are NULL in every nullable column; `id` never is, so
// a test can still join live against served rows.
bool null_row(int64_t j) {
    return j % 5 == 0;
}

std::shared_ptr<arrow::Schema> cache_types_schema() {
    return arrow::schema(
        {arrow::field("id", arrow::int64(), true),
         arrow::field("tags", arrow::list(arrow::field("item", arrow::int64(), true)), true),
         arrow::field("attrs",
                      arrow::struct_({arrow::field("x", arrow::int64(), true),
                                      arrow::field("y", arrow::utf8(), true)}),
                      true),
         arrow::field("amt", arrow::decimal128(18, 2), true),
         arrow::field("ts", arrow::timestamp(arrow::TimeUnit::MICRO), true),
         arrow::field("label", arrow::utf8(), true)});
}

std::shared_ptr<arrow::Array> build_tags(const std::shared_ptr<arrow::DataType>& type,
                                         int64_t start, int64_t rows) {
    auto list_type = std::dynamic_pointer_cast<arrow::ListType>(type);
    if (!list_type || list_type->value_type()->id() != arrow::Type::INT64) {
        throw std::runtime_error("cache_types: `tags` bound to " + type->ToString());
    }
    // Built against the bound type rather than a fresh `list(int64)`, so the
    // array's own child field is the one the engine asked for.
    arrow::ListBuilder builder(arrow::default_memory_pool(),
                               std::make_shared<arrow::Int64Builder>(), list_type);
    auto* values = static_cast<arrow::Int64Builder*>(builder.value_builder());
    for (int64_t j = start; j < start + rows; ++j) {
        if (null_row(j)) {
            (void)builder.AppendNull();
            continue;
        }
        (void)builder.Append();
        (void)values->AppendValues({j, j + 1, j + 2});
    }
    return finish(builder);
}

std::shared_ptr<arrow::Array> build_attrs(const std::shared_ptr<arrow::DataType>& type,
                                          int64_t start, int64_t rows) {
    auto struct_type = std::dynamic_pointer_cast<arrow::StructType>(type);
    if (!struct_type) {
        throw std::runtime_error("cache_types: `attrs` bound to " + type->ToString());
    }
    // DuckDB pushes projection *into* a struct, so a declared `struct<x, y>`
    // can arrive as `struct<y>`. Building a fixed two-child array would
    // disagree with its own type — which Arrow's constructors accept and the
    // consumer later crashes on — so the children come from the bound type.
    std::vector<std::shared_ptr<arrow::ArrayBuilder>> children;
    // Paired with the builders as they are made, so a child is never reached
    // through an unchecked downcast of `field_builder(i)`.
    std::vector<std::function<void(int64_t)>> append_row;
    for (const auto& child : struct_type->fields()) {
        if (child->name() == "x") {
            auto values = std::make_shared<arrow::Int64Builder>();
            append_row.push_back([values](int64_t j) { (void)values->Append(j); });
            children.push_back(std::move(values));
        } else if (child->name() == "y") {
            auto values = std::make_shared<arrow::StringBuilder>();
            append_row.push_back(
                [values](int64_t j) { (void)values->Append("y" + std::to_string(j)); });
            children.push_back(std::move(values));
        } else {
            throw std::runtime_error("cache_types: unknown `attrs` field " + child->name());
        }
    }
    arrow::StructBuilder builder(struct_type, arrow::default_memory_pool(), children);
    for (int64_t j = start; j < start + rows; ++j) {
        if (null_row(j)) {
            (void)builder.AppendNull();
            continue;
        }
        (void)builder.Append();
        for (const auto& append : append_row) append(j);
    }
    return finish(builder);
}

std::shared_ptr<arrow::Array> build_amt(const std::shared_ptr<arrow::DataType>& type, int64_t start,
                                        int64_t rows) {
    auto decimal_type = std::dynamic_pointer_cast<arrow::Decimal128Type>(type);
    if (!decimal_type) {
        throw std::runtime_error("cache_types: `amt` bound to " + type->ToString());
    }
    arrow::Decimal128Builder builder(decimal_type);
    (void)builder.Reserve(rows);
    for (int64_t j = start; j < start + rows; ++j) {
        if (null_row(j)) {
            (void)builder.AppendNull();
        } else {
            // j.{j % 100} at scale 2, i.e. the scaled integer.
            (void)builder.Append(arrow::Decimal128(j * 100 + j % 100));
        }
    }
    return finish(builder);
}

std::shared_ptr<arrow::Array> build_ts(const std::shared_ptr<arrow::DataType>& type, int64_t start,
                                       int64_t rows) {
    if (type->id() != arrow::Type::TIMESTAMP) {
        throw std::runtime_error("cache_types: `ts` bound to " + type->ToString());
    }
    arrow::TimestampBuilder builder(type, arrow::default_memory_pool());
    (void)builder.Reserve(rows);
    for (int64_t j = start; j < start + rows; ++j) {
        if (null_row(j)) {
            (void)builder.AppendNull();
        } else {
            (void)builder.Append(j);
        }
    }
    return finish(builder);
}

std::shared_ptr<arrow::Array> build_label(int64_t start, int64_t rows) {
    arrow::StringBuilder builder;
    for (int64_t j = start; j < start + rows; ++j) {
        if (null_row(j)) {
            (void)builder.AppendNull();
        } else {
            (void)builder.Append("label-" + std::to_string(j));
        }
    }
    return finish(builder);
}

std::shared_ptr<arrow::Array> build_types_column(const arrow::Field& field, int64_t start,
                                                 int64_t rows) {
    const auto& name = field.name();
    if (name == "id") return i64_column(start, start + rows);
    if (name == "tags") return build_tags(field.type(), start, rows);
    if (name == "attrs") return build_attrs(field.type(), start, rows);
    if (name == "amt") return build_amt(field.type(), start, rows);
    if (name == "ts") return build_ts(field.type(), start, rows);
    if (name == "label") return build_label(start, rows);
    throw std::runtime_error("cache_types: unknown column " + name);
}

class TypesProducer : public vgi::TableProducer {
public:
    TypesProducer(std::shared_ptr<arrow::Schema> schema, int64_t rows)
        : schema_(std::move(schema)), remaining_(rows < 0 ? 0 : rows) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (remaining_ <= 0) return nullptr;
        const int64_t size = std::min(remaining_, kBatchRows);
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(schema_->num_fields()));
        for (const auto& field : schema_->fields()) {
            columns.push_back(build_types_column(*field, index_, size));
        }
        metadata_ =
            index_ == 0 ? default_ttl().to_metadata() : std::map<std::string, std::string>{};
        index_ += size;
        remaining_ -= size;
        return arrow::RecordBatch::Make(schema_, size, std::move(columns));
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    // Small enough that a ten-thousand-row result is genuinely multi-batch,
    // which is what makes the spilled serve reassemble rather than replay one
    // blob.
    static constexpr int64_t kBatchRows = 2048;

    std::shared_ptr<arrow::Schema> schema_;
    int64_t remaining_;
    int64_t index_ = 0;
    std::map<std::string, std::string> metadata_;
};

class CacheTypes : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_types"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata(
            "Nested/wide/NULL cacheable result (STRUCT/LIST/DECIMAL/TIMESTAMP + NULLs)");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("rows", 0, "int64", "Number of rows to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return cache_types_schema();
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto rows = params.arguments.const_int64(0)) {
            estimate.estimate = *rows;
            estimate.max = *rows;
        }
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<TypesProducer>(params.output_schema,
                                               params.arguments.const_int64(0).value_or(0));
    }
};

// `cache_filtered(rows := 100)` — the only cacheable fixture that takes a
// pushed predicate, so it is the one that shows `filter_bytes` keying: two
// different pushed `WHERE`s must land in two entries and never cross-serve.
class CacheFiltered : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_filtered"; }

    vgi::FunctionMetadata metadata() const override {
        auto md =
            cache_metadata("Cacheable sequence with static filter pushdown (filter_bytes keying)");
        md.filter_pushdown = true;
        // The fixture only advertises the capability — it has no partition to
        // skip — so the framework applies the predicate to what it emits.
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("rows", "int64", "Number of rows to generate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("n");
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Sequence>(params.output_schema,
                                          params.arguments.named_int64("rows").value_or(100), 2048);
    }
};

// `cache_ordered(rows := 200000, chunk_size := 1000)` — the correct output is
// strictly `0..rows-1`, so a served result is wrong unless the replay restores
// source *order*, not merely the row set.
class CacheOrdered : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_ordered"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("Order-sensitive cacheable sequence; ordered-serve cache fixture");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("rows", "int64", "Number of rows to generate"),
                // One worker, so a chunk is exactly a batch: there is no queue
                // to divide the range across.
                vgi::ArgSpec::named("chunk_size", "int64", "Rows per emitted batch")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("n");
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        const int64_t rows = params.arguments.named_int64("rows").value_or(200000);
        estimate.estimate = rows;
        estimate.max = rows;
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Sequence>(
            params.output_schema, params.arguments.named_int64("rows").value_or(200000),
            params.arguments.named_int64("chunk_size").value_or(1000));
    }
};

// `cache_external_fail()` — a cacheable first batch, then a pointer batch the
// client cannot resolve.
//
// The never-partial check `cache_poison` makes with a worker error, made again
// from the other side: the scan dies in the *client*, after cacheable rows have
// already streamed, and still nothing may be committed.
class CacheExternalFail : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_external_fail"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata(
            "Cacheable first batch then an unresolvable external-location pointer");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("n");
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema);
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        explicit Producer(std::shared_ptr<arrow::Schema> schema) : schema_(std::move(schema)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (poisoned_) return nullptr;
            if (!emitted_) {
                emitted_ = true;
                metadata_ = default_ttl().to_metadata();
                return arrow::RecordBatch::Make(schema_, 3, {i64_column(0, 3)});
            }
            poisoned_ = true;
            metadata_ = {{vgi_rpc::keys::LOCATION, kUnresolvable}};
            return arrow::RecordBatch::Make(schema_, 0, {i64_column(0, 0)});
        }

        std::map<std::string, std::string> last_metadata() const override { return metadata_; }

    private:
        // Loopback port 9 (discard) is closed, so the fetch fails with
        // connection-refused rather than hanging on a DNS or TLS timeout.
        static constexpr const char* kUnresolvable =
            "http://127.0.0.1:9/vgi-cache-poison-nonexistent";

        std::shared_ptr<arrow::Schema> schema_;
        bool emitted_ = false;
        bool poisoned_ = false;
        std::map<std::string, std::string> metadata_;
    };
};

// `cache_whoami()` — one row naming the caller, so two bearer identities must
// not share an entry.
class CacheWhoami : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_whoami"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("Emits the caller's auth principal; cacheable (identity-scoped)");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("who", arrow::utf8(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        // Always anonymous: the SDK does not surface the resolved auth
        // principal, and bearer identity only exists over HTTP anyway.
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, 1, {utf8_column({"anonymous"})}));
    }
};

// `cache_versioned_scan(version)` — the catalog maps an `AT` clause onto the
// argument, so each pinned version and the live scan must key differently.
class CacheVersionedScan : public vgi::TableFunction {
public:
    std::string name() const override { return "cache_versioned_scan"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("Version-specific rows; cacheable (AT-keyed)");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("version", 0, "int64",
                                           "Data version, resolved from the AT clause")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return single_i64_schema("v");
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        // Disjoint values per version: a cross-version serve then shows up as
        // the wrong rows rather than a plausible count.
        const auto rows = version_rows(params.arguments.const_int64(0).value_or(kCurrent));
        arrow::Int64Builder builder;
        for (int64_t value : rows) (void)builder.Append(value);
        return std::make_unique<OneShot>(arrow::RecordBatch::Make(
            params.output_schema, static_cast<int64_t>(rows.size()), {finish(builder)}));
    }

private:
    static constexpr int64_t kCurrent = 3;

    static std::vector<int64_t> version_rows(int64_t version) {
        if (version == 1) return {101, 102, 103};
        if (version == 2) return {201, 202};
        return {301, 302, 303, 304};
    }
};

// `test_same_name_cached()` — registered under one name in two schemas, each
// row naming its own.
//
// The cache key was once catalog + auth + function name with no schema
// dimension, so the two produced identical keys and cross-served. A serve of
// the wrong schema's row is what that regression looks like.
class SameNameCached : public vgi::TableFunction {
public:
    explicit SameNameCached(std::string schema) : schema_(std::move(schema)) {}

    std::string name() const override { return "test_same_name_cached"; }

    vgi::FunctionMetadata metadata() const override {
        return cache_metadata("One cacheable row tagged with its owning schema");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("tag", arrow::utf8(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, 1, {utf8_column({schema_})}));
    }

private:
    std::string schema_;
};

// `cached_echo(data)` — a cacheable classic (TABLE-input) passthrough.
//
// Its output is memoized per input batch, so a repeat scan is served without
// the worker exchange running at all.
class CachedEcho : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "cached_echo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Cacheable classic (TABLE-input) passthrough (advertises vgi.cache.ttl)";
        md.categories = {"cache", "test"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input table")};
    }

    std::optional<vgi::CacheControl> cache_control() const override { return default_ttl(); }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return {vgi::project_batch(batch, params.output_schema)};
    }
};

// `cached_reval_echo(data)` — the same passthrough under the always-revalidate
// contract: stored, but immediately stale, so every repeat asks rather than
// assumes.
class CachedRevalEcho : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "cached_reval_echo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Classic passthrough with always-revalidate (304 not_modified) contract";
        md.categories = {"cache", "test"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input table")};
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        vgi::CacheControl control;
        // ttl 0 with a validator is the "no-cache" semantic: keep the bytes,
        // but confirm before reusing them.
        control.ttl_seconds = 0;
        control.revalidatable = true;
        // A constant validator: this fixture's answer never actually changes,
        // so every revalidation is expected to confirm it.
        control.etag = kEtag;

        // Per emission rather than per function, because the answer to a
        // conditional request is about *this* input: a validator that matches
        // means the stored bytes still stand, and the rows must not be sent
        // again.
        if (params.if_none_match && *params.if_none_match == kEtag) {
            control.not_modified = true;
            vgi::EmittedBatch confirmed{empty_like(params.output_schema)};
            confirmed.cache_control = control;
            return {std::move(confirmed)};
        }

        vgi::EmittedBatch emitted{vgi::project_batch(batch, params.output_schema)};
        emitted.cache_control = control;
        return {std::move(emitted)};
    }

private:
    static constexpr const char* kEtag = "\"vgi-cpp-reval-echo\"";

    static std::shared_ptr<arrow::RecordBatch> empty_like(
        const std::shared_ptr<arrow::Schema>& schema) {
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(schema->num_fields()));
        for (const auto& field : schema->fields()) {
            columns.push_back(arrow::MakeArrayOfNull(field->type(), 0).ValueOrDie());
        }
        return arrow::RecordBatch::Make(schema, 0, columns);
    }
};

}  // namespace

void register_more_cache(vgi::Worker& worker) {
    // The primary catalog, not a hardcoded name: one binary stands in for
    // several fixtures, and these belong to whichever it is serving.
    const auto& primary = worker.catalog().name;
    worker.register_table(std::make_shared<CacheTypes>());
    worker.register_table(std::make_shared<CacheFiltered>());
    worker.register_table(std::make_shared<CacheOrdered>());
    worker.register_table(std::make_shared<CacheExternalFail>());
    worker.register_table(std::make_shared<CacheWhoami>());
    worker.register_table(std::make_shared<CacheVersionedScan>());
    worker.register_table_in(primary, "main", std::make_shared<SameNameCached>("main"));
    worker.register_table_in(primary, "data", std::make_shared<SameNameCached>("data"));
    worker.register_table_in_out(std::make_shared<CachedEcho>());
    worker.register_table_in_out(std::make_shared<CachedRevalEcho>());
}

}  // namespace example
