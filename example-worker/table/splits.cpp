// © Copyright 2025, 2026 Query Farm LLC - https://query.farm

// Split-capable producers: the plan()/on_split() path.
//
// A split names work rather than describing it, so each payload here is a
// half-open row range `[begin, end)` over the same 0..n-1 the plain `sequence`
// emits. That is what makes redemption replayable — the same payload redeemed
// twice, or on another host, names the same rows — and it is what lets these
// fixtures assert against `sequence(n)` rather than merely against themselves.
//
// All five are deliberately split-*only*: they refuse an init that carries no
// payloads. A worker whose scan is only correct when planned should say so
// rather than quietly serving a different row set when the engine turns
// planning off, which is what `splits/rollback.test` pins.

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// A range as 16 bytes, two little-endian int64s. Fixed-width on purpose: a
// payload is compared and stored by the framework, and a text encoding would
// invite a locale or a leading zero into an identity.
std::string encode_range(int64_t begin, int64_t end) {
    std::string out(16, '\0');
    for (int i = 0; i < 8; ++i) {
        out[static_cast<size_t>(i)] = static_cast<char>((static_cast<uint64_t>(begin) >> (8 * i)) & 0xFF);
        out[static_cast<size_t>(i + 8)] = static_cast<char>((static_cast<uint64_t>(end) >> (8 * i)) & 0xFF);
    }
    return out;
}

std::optional<std::pair<int64_t, int64_t>> decode_range(const std::string& payload) {
    if (payload.size() != 16) return std::nullopt;
    const auto read = [&](size_t at) {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<uint64_t>(static_cast<unsigned char>(payload[at + static_cast<size_t>(i)]))
                 << (8 * i);
        }
        return static_cast<int64_t>(v);
    };
    return std::pair{read(0), read(8)};
}

std::shared_ptr<arrow::Schema> n_schema() {
    return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/false)});
}

// Emits the rows of one or more half-open ranges, in order.
class RangeProducer : public vgi::TableProducer {
public:
    RangeProducer(std::shared_ptr<arrow::Schema> schema,
                  std::vector<std::pair<int64_t, int64_t>> ranges)
        : schema_(std::move(schema)), ranges_(std::move(ranges)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        // An empty range is not the end of the scan: `split_empty_ranges`
        // exists because a producer that stopped at the first empty one would
        // silently truncate every split after it.
        while (at_ < ranges_.size() && cursor_ >= ranges_[at_].second) {
            ++at_;
            if (at_ < ranges_.size()) cursor_ = ranges_[at_].first;
        }
        if (at_ >= ranges_.size()) return nullptr;
        if (cursor_ < ranges_[at_].first) cursor_ = ranges_[at_].first;

        const int64_t end = std::min(cursor_ + kBatchRows, ranges_[at_].second);
        arrow::Int64Builder builder;
        (void)builder.Reserve(end - cursor_);
        for (int64_t v = cursor_; v < end; ++v) (void)builder.Append(v);
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        auto batch = arrow::RecordBatch::Make(schema_, end - cursor_, {array});
        cursor_ = end;
        return batch;
    }

private:
    static constexpr int64_t kBatchRows = 2048;
    std::shared_ptr<arrow::Schema> schema_;
    std::vector<std::pair<int64_t, int64_t>> ranges_;
    size_t at_ = 0;
    int64_t cursor_ = 0;
};

// The shared shape of all five: `n` rows divided into `splits` ranges, with the
// division itself the only thing that differs.
class SplitFunction : public vgi::TableFunction {
public:
    enum class Shape { Even, EmptyRanges, Zero, Skewed, Many };

    SplitFunction(std::string name, Shape shape, std::string description)
        : name_(std::move(name)), shape_(shape), description_(std::move(description)) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = description_;
        md.categories = {"generator"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("n", "int64", "How many rows to generate"),
                vgi::ArgSpec::named("splits", "int64", "How many splits to divide them into")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return n_schema();
    }

    bool supports_splits() const override { return true; }

    vgi::PlanResult plan(const vgi::BindParams& params, const vgi::PlanParams&) const override {
        const int64_t rows = std::max<int64_t>(0, params.arguments.named_int64("n").value_or(0));
        const int64_t want = std::max<int64_t>(1, params.arguments.named_int64("splits").value_or(1));

        vgi::PlanResult result;
        result.estimated_total_rows = shape_ == Shape::Zero ? 0 : rows;
        result.estimated_total_splits = want;
        for (const auto& range : divide(rows, want)) {
            vgi::ScanSplit split;
            split.payload = encode_range(range.first, range.second);
            split.estimated_rows = range.second - range.first;
            // Exact, because a range over a generated sequence knows its own
            // size — which is what lets the engine answer COUNT(*) from the
            // plan rather than by reading.
            split.rows_exact = true;
            result.splits.push_back(std::move(split));
        }
        return result;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        if (!params.split_payloads) {
            // The message is asserted on: this function is split-only, and
            // saying so beats serving a silently different row set.
            throw std::runtime_error("table function '" + name_ +
                                     "' is split-only: it has no unplanned scan path, so it "
                                     "cannot run with split scans turned off");
        }
        std::vector<std::pair<int64_t, int64_t>> ranges;
        ranges.reserve(params.split_payloads->size());
        for (const auto& payload : *params.split_payloads) {
            auto range = decode_range(payload);
            if (!range) {
                throw std::runtime_error("table function '" + name_ +
                                         "' was handed a split payload it did not mint");
            }
            ranges.push_back(*range);
        }
        return std::make_unique<RangeProducer>(params.output_schema ? params.output_schema
                                                                    : n_schema(),
                                               std::move(ranges));
    }

private:
    // Every shape covers 0..rows-1 exactly once. That invariant is the point:
    // the tests assert each one against `sequence(n)`, so a division that
    // dropped or repeated a boundary row would show up as a row-set
    // difference rather than as a count that happens to match.
    std::vector<std::pair<int64_t, int64_t>> divide(int64_t rows, int64_t want) const {
        std::vector<std::pair<int64_t, int64_t>> out;
        switch (shape_) {
            case Shape::Zero:
                // Splits that name no rows at all. The engine must still claim
                // and redeem them without deciding the scan ended early.
                for (int64_t i = 0; i < want; ++i) out.emplace_back(0, 0);
                return out;

            case Shape::EmptyRanges: {
                // Every other split is empty, so an empty one is never last and
                // never first — a producer that stopped at one would truncate.
                const int64_t real = (want + 1) / 2;
                int64_t begin = 0;
                for (int64_t i = 0; i < want; ++i) {
                    if (i % 2 == 1) {
                        out.emplace_back(begin, begin);
                        continue;
                    }
                    const int64_t index = i / 2;
                    const int64_t end = rows * (index + 1) / real;
                    out.emplace_back(begin, end);
                    begin = end;
                }
                return out;
            }

            case Shape::Skewed: {
                // Deliberately uneven: the first split takes half the rows and
                // the rest divide what is left, so a scheduler that assumed
                // equal cost is visibly wrong rather than merely slower.
                if (want == 1 || rows == 0) {
                    out.emplace_back(0, rows);
                    return out;
                }
                const int64_t head = rows / 2;
                out.emplace_back(0, head);
                for (int64_t i = 1; i < want; ++i) {
                    const int64_t begin = head + (rows - head) * (i - 1) / (want - 1);
                    const int64_t end = head + (rows - head) * i / (want - 1);
                    out.emplace_back(begin, end);
                }
                return out;
            }

            case Shape::Even:
            case Shape::Many:
                break;
        }
        // Proportional rather than `rows / want` with a remainder bolted onto
        // the last split: this keeps every split within one row of every other,
        // and covers the range exactly whether or not the division is even.
        for (int64_t i = 0; i < want; ++i) {
            out.emplace_back(rows * i / want, rows * (i + 1) / want);
        }
        return out;
    }

    std::string name_;
    Shape shape_;
    std::string description_;
};

}  // namespace

void register_splits(vgi::Worker& worker) {
    using Shape = SplitFunction::Shape;
    worker.register_table(std::make_shared<SplitFunction>(
        "split_sequence", Shape::Even, "Integers 0..n-1, divided into n contiguous splits"));
    worker.register_table(std::make_shared<SplitFunction>(
        "split_empty_ranges", Shape::EmptyRanges,
        "Integers 0..n-1, where every other split names no rows"));
    worker.register_table(std::make_shared<SplitFunction>(
        "split_zero", Shape::Zero, "Splits that name no rows at all"));
    worker.register_table(std::make_shared<SplitFunction>(
        "split_skewed", Shape::Skewed, "Integers 0..n-1, divided very unevenly"));
    worker.register_table(std::make_shared<SplitFunction>(
        "split_many", Shape::Many, "Integers 0..n-1, divided into many more splits than threads"));
}

}  // namespace example
