// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "util.h"

#include <atomic>
#include <chrono>
#include <stdexcept>

namespace example {

std::shared_ptr<arrow::RecordBatch> result(const vgi::ProcessParams& params,
                                           std::shared_ptr<arrow::Array> array) {
    return arrow::RecordBatch::Make(params.output_schema, array->length(),
                                    {std::move(array)});
}

std::shared_ptr<arrow::Array> cast_to(const std::shared_ptr<arrow::Array>& array,
                                      const std::shared_ptr<arrow::DataType>& type) {
    if (array->type()->Equals(*type)) return array;
    auto casted = arrow::compute::Cast(*array, type);
    if (!casted.ok()) {
        throw std::runtime_error("cannot cast " + array->type()->ToString() + " to " +
                                 type->ToString() + ": " + casted.status().ToString());
    }
    return casted.MoveValueUnsafe();
}

std::shared_ptr<arrow::DataType> output_type(const vgi::ProcessParams& params) {
    if (!params.output_schema || params.output_schema->num_fields() == 0) {
        throw std::runtime_error("output schema has no fields");
    }
    return params.output_schema->field(0)->type();
}

uint64_t Rng::next() {
    state_ += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int64_t Rng::range(int64_t low, int64_t high) {
    if (high <= low) return low;
    // Width in unsigned arithmetic: high - low can overflow int64 for a range
    // spanning most of the type, and the fixtures do exercise wide ranges.
    const auto span = static_cast<unsigned __int128>(
        static_cast<__int128>(high) - static_cast<__int128>(low) + 1);
    return low + static_cast<int64_t>(static_cast<unsigned __int128>(next()) % span);
}

uint64_t volatile_seed() {
    static std::atomic<uint64_t> counter{0x123456789ABCDEF0ULL};
    const uint64_t n = counter.fetch_add(0x9E3779B97F4A7C15ULL, std::memory_order_relaxed);
    const auto now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    return n ^ now;
}

}  // namespace example
