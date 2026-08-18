// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Shared helpers for the scalar fixtures.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <arrow/array.h>
#include <arrow/compute/cast.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <vgi/function.h>

namespace example {

// Build the single-column `result` batch every scalar function answers with.
std::shared_ptr<arrow::RecordBatch> result(const vgi::ProcessParams& params,
                                           std::shared_ptr<arrow::Array> array);

// Cast `array` to `type`, throwing with the reason rather than returning a
// status nobody checks.
std::shared_ptr<arrow::Array> cast_to(const std::shared_ptr<arrow::Array>& array,
                                      const std::shared_ptr<arrow::DataType>& type);

// The single output type this call was bound to.
std::shared_ptr<arrow::DataType> output_type(const vgi::ProcessParams& params);

// SplitMix64 — a small, fast, deterministic generator.
//
// Deterministic matters: the seeded fixtures assert exact values, so the
// sequence has to match the Rust and Python workers bit for bit. That rules
// out the standard library's engines, whose output is not specified.
class Rng {
public:
    explicit Rng(uint64_t seed) : state_(seed) {}

    uint64_t next();
    // Uniform in [low, high], inclusive.
    int64_t range(int64_t low, int64_t high);
    uint8_t byte() { return static_cast<uint8_t>(next() & 0xFF); }

private:
    uint64_t state_;
};

// Entropy for the VOLATILE fixtures, which must differ between calls.
uint64_t volatile_seed();

}  // namespace example
