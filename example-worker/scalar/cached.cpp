// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Cacheable scalar fixtures whose *output* shape is what the per-value memo
// has to survive, rather than the arithmetic the cached_* fixtures in
// table_in_out/cached.cpp cover: a heap string and a NULL, which the memo
// stores and replays out of a cache entry rather than recomputing.

#include <memory>
#include <string>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

constexpr int64_t kCacheTtlSeconds = 300;

// `cached_label(value)` — 'lbl-<value>' for non-negative values, NULL below.
//
// Deterministic, so advertising the cache is sound; the negatives are there to
// force a NULL through the per-value store, which is a different path from a
// present value.
class CachedLabel : public vgi::ScalarFunction {
public:
    std::string name() const override { return "cached_label"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "value -> 'lbl-<value>' or NULL for negatives (advertises vgi.cache.ttl)";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Value")};
    }

    std::optional<vgi::CacheControl> cache_control() const override {
        vgi::CacheControl control;
        control.ttl_seconds = kCacheTtlSeconds;
        control.per_value = true;
        return control;
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto values = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(batch->column(0), arrow::int64()));
        arrow::StringBuilder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i) || values->Value(i) < 0) {
                (void)out.AppendNull();
            } else {
                (void)out.Append("lbl-" + std::to_string(values->Value(i)));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

}  // namespace

void register_cached_scalars(vgi::Worker& worker) {
    worker.register_scalar(std::make_shared<CachedLabel>());
}

}  // namespace example
