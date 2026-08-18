// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Cacheable maps: exchange-mode functions that advertise `vgi.cache.*`.
//
// Advertising is only sound because these are deterministic: the engine may
// serve a later call an earlier answer, so a function whose result depends on
// anything outside its arguments must not opt in.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

constexpr int64_t kCacheTtlSeconds = 300;

vgi::CacheControl per_value_cache() {
    vgi::CacheControl control;
    control.ttl_seconds = kCacheTtlSeconds;
    // Per-value memoization on top of the whole-result cache. Worth it only
    // when one call is expensive; these fixtures exist to exercise the tier,
    // not because doubling an integer is expensive.
    control.per_value = true;
    return control;
}

// `cached_double(x)` — a 1:1 map, x -> x*2.
class CachedDouble : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "cached_double"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Cacheable blended map x -> x*2 (advertises vgi.cache.ttl)";
        md.categories = {"blended", "cache", "test"};
        // Blended: its input rows come from its arguments, which is what lets
        // the same registration serve a plain call and a correlated one.
        md.input_from_args = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("x", 0, "int64", "Input column")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("doubled", arrow::int64(), /*nullable=*/true)});
    }

    std::optional<vgi::CacheControl> cache_control() const override {
        return per_value_cache();
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto values = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(batch->column(0), arrow::int64()));
        arrow::Int64Builder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(values->Value(i) * 2);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return {arrow::RecordBatch::Make(params.output_schema, array->length(), {array})};
    }
};

// `cached_explode(n)` — a 1:N fan-out, emitting `0..n-1` per input row.
//
// It exists beside `cached_double` because the memo tier has to handle
// cardinalities a 1:1 map never reaches: `n = 0` is a *negative* memo — a
// zero-length slot that must not be confused with a miss — and `n > 1` is a
// multi-row slot whose rows have to survive being gathered back out.
class CachedExplode : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "cached_explode"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Cacheable blended 1:N fan-out emitting range(n) per input row";
        md.categories = {"blended", "cache", "test"};
        // Blended: its input rows come from its arguments, which is what lets
        // the same registration serve a plain call and a correlated one.
        md.input_from_args = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("n", 0, "int64", "Rows to emit per input row")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("i", arrow::int64(), /*nullable=*/true)});
    }

    std::optional<vgi::CacheControl> cache_control() const override {
        return per_value_cache();
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto counts = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(batch->column(0), arrow::int64()));
        arrow::Int64Builder out;
        for (int64_t row = 0; row < counts->length(); ++row) {
            if (counts->IsNull(row)) continue;
            for (int64_t i = 0; i < counts->Value(row); ++i) (void)out.Append(i);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return {arrow::RecordBatch::Make(params.output_schema, array->length(), {array})};
    }
};

// The scalar counterparts, for the per-value memo on the scalar path.
class CachedDoubleScalar : public vgi::ScalarFunction {
public:
    std::string name() const override { return "cached_double_scalar"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Doubles a BIGINT value (advertises vgi.cache.ttl for per-value memo)";
        md.return_type = arrow::int64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Value to double")};
    }

    std::optional<vgi::CacheControl> cache_control() const override {
        return per_value_cache();
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto values = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(batch->column(0), arrow::int64()));
        arrow::Int64Builder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(values->Value(i) * 2);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `cached_add_const(value, addend)` — the const argument must be folded into
// the cache key, or two calls with the same `value` and different `addend`
// cross-serve each other's answers.
class CachedAddConst : public vgi::ScalarFunction {
public:
    std::string name() const override { return "cached_add_const"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Adds a const addend to a value (cacheable; const arg keys the memo)";
        md.return_type = arrow::int64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Value to add to"),
                vgi::ArgSpec::constant_arg("addend", 1, "int64", "Constant addend")};
    }

    std::optional<vgi::CacheControl> cache_control() const override {
        return per_value_cache();
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t addend = params.arguments.const_int64(1).value_or(0);
        auto values = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(batch->column(0), arrow::int64()));
        arrow::Int64Builder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(values->Value(i) + addend);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

}  // namespace

void register_cached(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<CachedDouble>());
    worker.register_table_in_out(std::make_shared<CachedExplode>());
    worker.register_scalar(std::make_shared<CachedDoubleScalar>());
    worker.register_scalar(std::make_shared<CachedAddConst>());
}

}  // namespace example
