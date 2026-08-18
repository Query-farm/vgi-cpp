// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The per-substream finalize fixture.
//
// A streaming table-in-out with a finalize is still a *per-substream*
// operation: the engine fans the input across several substreams, each with its
// own worker, and finalizes each one separately. What this probes is that a
// substream's finalize sees what its own ticks accumulated and nothing else —
// a global combine across substreams is a different shape entirely, and is
// what `TableBufferingFunction` is for.

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// The log this substream's partials go into. Namespaced so it cannot collide
// with anything else keyed on the same scope.
constexpr const char* kNamespace = "substream.partial";

// The scope a substream's state lives under.
//
// The client-minted substream id where there is one, because it is stable
// across the substream's init, ticks and finalize even when a load balancer
// sends them to different backends. The worker-minted execution id is the
// fallback, which is enough over a pipe where all three land in one process.
std::string state_scope(const vgi::ProcessParams& params) {
    return params.substream_id.empty() ? params.execution_id : params.substream_id;
}

std::string encode_i64(int64_t value) {
    std::string bytes(sizeof(value), '\0');
    for (size_t i = 0; i < sizeof(value); ++i) {
        bytes[i] = static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 0xFF);
    }
    return bytes;
}

int64_t decode_i64(const std::string& bytes) {
    if (bytes.size() < sizeof(int64_t)) return 0;
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return static_cast<int64_t>(value);
}

class SubstreamPartialSum : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "substream_partial_sum"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Per-substream partial sum emitted at finalize (parallel streaming finalize)";
        md.categories = {"aggregation", "streaming"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema || params.input_schema->num_fields() == 0) {
            throw std::invalid_argument("substream_partial_sum requires an input column");
        }
        // The input's column name, so an outer `sum(n)` still names `n`.
        return arrow::schema({arrow::field(params.input_schema->field(0)->name(),
                                           arrow::int64(), /*nullable=*/true)});
    }

    bool has_finish() const override { return true; }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        if (!batch || batch->num_columns() == 0) return {};
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(0), arrow::int64()));

        int64_t total = 0;
        for (int64_t i = 0; i < values->length(); ++i) {
            if (!values->IsNull(i)) total += values->Value(i);
        }
        // Appended rather than accumulated in memory: a substream's ticks may
        // be spread across worker processes, and the finalize may run in yet
        // another.
        params.storage->append(state_scope(params), kNamespace, "", encode_i64(total));
        // Accumulate only — the whole answer is one row, and it is not known
        // until the substream ends.
        return {};
    }

    std::vector<vgi::EmittedBatch> finish(const vgi::ProcessParams& params) const override {
        int64_t total = 0;
        for (const auto& [id, blob] :
             params.storage->scan(state_scope(params), kNamespace, "", 0, SIZE_MAX)) {
            (void)id;
            total += decode_i64(blob);
        }

        arrow::Int64Builder builder;
        (void)builder.Append(total);
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        return {arrow::RecordBatch::Make(params.output_schema, 1, {array})};
    }
};

}  // namespace

void register_substream_finalize(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<SubstreamPartialSum>());
}

}  // namespace example
