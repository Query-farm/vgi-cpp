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
#include <arrow/array/builder_binary.h>
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
        out[static_cast<size_t>(i)] =
            static_cast<char>((static_cast<uint64_t>(begin) >> (8 * i)) & 0xFF);
        out[static_cast<size_t>(i + 8)] =
            static_cast<char>((static_cast<uint64_t>(end) >> (8 * i)) & 0xFF);
    }
    return out;
}

std::optional<std::pair<int64_t, int64_t>> decode_range(const std::string& payload) {
    if (payload.size() != 16) return std::nullopt;
    const auto read = [&](size_t at) {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<uint64_t>(
                     static_cast<unsigned char>(payload[at + static_cast<size_t>(i)]))
                 << (8 * i);
        }
        return static_cast<int64_t>(v);
    };
    return std::pair{read(0), read(8)};
}

std::shared_ptr<arrow::Schema> n_schema() {
    return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/false)});
}

std::shared_ptr<arrow::Schema> dynamic_filter_schema() {
    return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/false),
                          arrow::field("pushed_filters", arrow::utf8(), /*nullable=*/false)});
}

std::string render_filter_bounds(const vgi::PushdownFilters& filters) {
    std::string result;
    for (const auto& column : filters.filtered_columns()) {
        const auto bounds = filters.column_bounds(column);
        if (bounds.min) {
            if (!result.empty()) result += ',';
            result += column + ">=" + std::to_string(*bounds.min);
        }
        if (bounds.max) {
            if (!result.empty()) result += ',';
            result += column + "<=" + std::to_string(*bounds.max);
        }
    }
    return result.empty() ? "(none)" : result;
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
        const int64_t want =
            std::max<int64_t>(1, params.arguments.named_int64("splits").value_or(1));

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
        return std::make_unique<RangeProducer>(
            params.output_schema ? params.output_schema : n_schema(), std::move(ranges));
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
            case Shape::Many: break;
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

// `split_fail_at(n, splits, fail_at, fail_in_init)` — a scan that dies where it
// is told to.
//
// Two distinct failures, because a client has to survive both: one during
// `init`, before any row is produced, and one mid-stream after several batches
// have already been handed over. The second is the one that poisons a
// connection if the transport mishandles it, which is why `poisoned_conn.test`
// runs an ordinary scan straight afterwards.
class SplitFailAt : public vgi::TableFunction {
public:
    std::string name() const override { return "split_fail_at"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Integers 0..n-1 in splits, failing where told to";
        md.categories = {"generator"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("n", "int64", "How many rows to generate"),
                vgi::ArgSpec::named("splits", "int64", "How many splits"),
                vgi::ArgSpec::named("fail_at", "int64", "Row to fail at; negative never fails"),
                vgi::ArgSpec::named("fail_in_init", "boolean",
                                    "Fail while initializing rather than mid-stream")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return n_schema();
    }

    bool supports_splits() const override { return true; }

    vgi::PlanResult plan(const vgi::BindParams& params, const vgi::PlanParams&) const override {
        const int64_t rows = std::max<int64_t>(0, params.arguments.named_int64("n").value_or(0));
        const int64_t want =
            std::max<int64_t>(1, params.arguments.named_int64("splits").value_or(1));
        vgi::PlanResult result;
        for (int64_t i = 0; i < want; ++i) {
            vgi::ScanSplit split;
            split.payload = encode_range(rows * i / want, rows * (i + 1) / want);
            result.splits.push_back(std::move(split));
        }
        return result;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        if (!params.split_payloads) {
            throw std::runtime_error("table function 'split_fail_at' is split-only");
        }
        const int64_t fail_at = params.arguments.named_int64("fail_at").value_or(-1);
        const bool in_init = params.arguments.named_bool("fail_in_init").value_or(false);

        std::vector<std::pair<int64_t, int64_t>> ranges;
        for (const auto& payload : *params.split_payloads) {
            auto range = decode_range(payload);
            if (!range) throw std::runtime_error("split_fail_at: unrecognized split payload");
            if (in_init && fail_at >= range->first && fail_at < range->second) {
                throw std::runtime_error(
                    "split_fail_at refuses to initialize the split covering row " +
                    std::to_string(fail_at));
            }
            ranges.push_back(*range);
        }
        return std::make_unique<FailingProducer>(
            params.output_schema ? params.output_schema : n_schema(), std::move(ranges),
            in_init ? -1 : fail_at);
    }

private:
    // Emits rows until it reaches `fail_at`, then throws. The rows already
    // handed over stay handed over, which is the situation a client has to
    // recover from.
    class FailingProducer : public vgi::TableProducer {
    public:
        FailingProducer(std::shared_ptr<arrow::Schema> schema,
                        std::vector<std::pair<int64_t, int64_t>> ranges, int64_t fail_at)
            : inner_(std::move(schema), std::move(ranges)), fail_at_(fail_at) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            auto batch = inner_.next_batch();
            if (!batch || fail_at_ < 0) return batch;
            const auto& values = static_cast<const arrow::Int64Array&>(*batch->column(0));
            for (int64_t i = 0; i < batch->num_rows(); ++i) {
                if (values.Value(i) == fail_at_) {
                    throw std::runtime_error("split_fail_at failed mid-stream at row " +
                                             std::to_string(fail_at_));
                }
            }
            return batch;
        }

    private:
        RangeProducer inner_;
        int64_t fail_at_;
    };
};

// `split_endless_cursor(n, splits)` — a plan that never finishes enumerating.
//
// Every page names a next cursor, so a client that followed it forever would
// hang. The engine caps the pages instead and refuses to scan a partial
// enumeration, which is the behaviour `plan_bounds.test` pins: a worker can
// waste a bounded amount of a client's time, not an unbounded amount.
class SplitEndlessCursor : public vgi::TableFunction {
public:
    std::string name() const override { return "split_endless_cursor"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "A plan whose split enumeration never terminates";
        md.categories = {"generator"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("n", "int64", "How many rows"),
                vgi::ArgSpec::named("splits", "int64", "How many splits per page")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return n_schema();
    }

    bool supports_splits() const override { return true; }

    vgi::PlanResult plan(const vgi::BindParams& params,
                         const vgi::PlanParams& request) const override {
        const int64_t rows = std::max<int64_t>(0, params.arguments.named_int64("n").value_or(0));
        vgi::PlanResult result;
        vgi::ScanSplit split;
        split.payload = encode_range(0, rows);
        result.splits.push_back(std::move(split));
        // Always another page, whatever the client sent back. The cursor even
        // advances, so this is not a client failing to forward it.
        const int64_t page = request.cursor ? static_cast<int64_t>(request.cursor->size()) : 0;
        result.next_cursor = std::string(static_cast<size_t>(page) + 1, 'x');
        return result;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        if (!params.split_payloads) {
            throw std::runtime_error("table function 'split_endless_cursor' is split-only");
        }
        std::vector<std::pair<int64_t, int64_t>> ranges;
        for (const auto& payload : *params.split_payloads) {
            if (auto range = decode_range(payload)) ranges.push_back(*range);
        }
        return std::make_unique<RangeProducer>(
            params.output_schema ? params.output_schema : n_schema(), std::move(ranges));
    }
};

// `split_dynamic_filter(n, splits)` — a split scan that reports and applies
// the filter in force for every batch. A reader re-initializes between claimed
// splits, so reporting the filter as data makes lost state observable even
// though DuckDB also checks the predicate above the scan.
class SplitDynamicFilter : public vgi::TableFunction {
public:
    std::string name() const override { return "split_dynamic_filter"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Echoes the dynamic filter each tick carried, per split";
        md.categories = {"generator", "diagnostic"};
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("n", "int64", "How many rows to generate"),
                vgi::ArgSpec::named("splits", "int64", "How many splits")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return dynamic_filter_schema();
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        const auto rows = std::max<int64_t>(0, params.arguments.named_int64("n").value_or(0));
        return {rows, rows};
    }

    bool supports_splits() const override { return true; }

    vgi::PlanResult plan(const vgi::BindParams& params, const vgi::PlanParams&) const override {
        const int64_t rows = std::max<int64_t>(0, params.arguments.named_int64("n").value_or(0));
        const int64_t want =
            std::max<int64_t>(1, params.arguments.named_int64("splits").value_or(1));

        vgi::PlanResult result;
        result.estimated_total_rows = rows;
        result.estimated_total_splits = want;
        for (int64_t i = 0; i < want; ++i) {
            vgi::ScanSplit split;
            const int64_t begin = rows * i / want;
            const int64_t end = rows * (i + 1) / want;
            split.payload = encode_range(begin, end);
            split.estimated_rows = end - begin;
            split.rows_exact = true;
            result.splits.push_back(std::move(split));
        }
        return result;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        if (!params.split_payloads) {
            throw std::runtime_error(
                "split_dynamic_filter is split-only but was initialized with no split tokens");
        }
        std::vector<std::pair<int64_t, int64_t>> ranges;
        ranges.reserve(params.split_payloads->size());
        for (const auto& payload : *params.split_payloads) {
            auto range = decode_range(payload);
            if (!range) {
                throw std::runtime_error("split_dynamic_filter: unrecognized split payload");
            }
            ranges.push_back(*range);
        }
        return std::make_unique<Producer>(
            params.output_schema ? params.output_schema : dynamic_filter_schema(),
            std::move(ranges), render_filter_bounds(params.pushdown_filters));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema,
                 std::vector<std::pair<int64_t, int64_t>> ranges, std::string rendered)
            : schema_(std::move(schema)),
              ranges_(std::move(ranges)),
              rendered_(std::move(rendered)) {
            if (!ranges_.empty()) cursor_ = ranges_.front().first;
        }

        void on_dynamic_filters(const vgi::PushdownFilters& filters) override {
            rendered_ = render_filter_bounds(filters);
        }

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            while (at_ < ranges_.size() && cursor_ >= ranges_[at_].second) {
                ++at_;
                if (at_ < ranges_.size()) cursor_ = ranges_[at_].first;
            }
            if (at_ >= ranges_.size()) return nullptr;

            constexpr int64_t kBatchRows = 4;
            const int64_t begin = cursor_;
            const int64_t end = std::min(begin + kBatchRows, ranges_[at_].second);
            cursor_ = end;

            arrow::Int64Builder ns;
            arrow::StringBuilder reports;
            (void)ns.Reserve(end - begin);
            (void)reports.Reserve(end - begin);
            for (int64_t value = begin; value < end; ++value) {
                (void)ns.Append(value);
                (void)reports.Append(rendered_);
            }
            std::vector<std::shared_ptr<arrow::Array>> built(2);
            (void)ns.Finish(&built[0]);
            (void)reports.Finish(&built[1]);

            const std::vector<std::string> names{"n", "pushed_filters"};
            std::vector<std::shared_ptr<arrow::Array>> projected;
            projected.reserve(static_cast<size_t>(schema_->num_fields()));
            for (const auto& field : schema_->fields()) {
                const auto found = std::find(names.begin(), names.end(), field->name());
                if (found == names.end()) {
                    throw std::runtime_error("split_dynamic_filter: unexpected column '" +
                                             field->name() + "'");
                }
                projected.push_back(built[static_cast<size_t>(found - names.begin())]);
            }
            return arrow::RecordBatch::Make(schema_, end - begin, std::move(projected));
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        std::vector<std::pair<int64_t, int64_t>> ranges_;
        size_t at_ = 0;
        int64_t cursor_ = 0;
        std::string rendered_;
    };
};

// `split_echo_filters(splits)` — reports what `plan()` was told.
//
// One row per split, carrying the split's ordinal and whether planning saw any
// pushdown. It is the only way to observe from SQL that filters and projections
// reach planning at all, rather than only reaching `init`.
class SplitEchoFilters : public vgi::TableFunction {
public:
    std::string name() const override { return "split_echo_filters"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "One row per split, reporting the pushdown plan() saw";
        md.categories = {"generator"};
        md.filter_pushdown = true;
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("splits", "int64", "How many splits")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("split_ordinal", arrow::int64(), /*nullable=*/false),
                              arrow::field("saw_filters", arrow::boolean(), /*nullable=*/false),
                              arrow::field("n_projection", arrow::int64(), /*nullable=*/false)});
    }

    bool supports_splits() const override { return true; }

    vgi::PlanResult plan(const vgi::BindParams& params,
                         const vgi::PlanParams& request) const override {
        const int64_t want =
            std::max<int64_t>(1, params.arguments.named_int64("splits").value_or(1));
        vgi::PlanResult result;
        // Pruning at plan time is what pushing a filter into planning is *for*:
        // a split the predicate excludes should never be handed out, rather
        // than handed out, read, and filtered away afterwards.
        const auto bounds = request.pushdown_filters.column_bounds("split_ordinal");
        for (int64_t i = 0; i < want; ++i) {
            if (bounds.min && i < *bounds.min) continue;
            if (bounds.max && i > *bounds.max) continue;
            vgi::ScanSplit split;
            // The observation travels in the payload, because planning and
            // reading are different processes: a member here would be empty by
            // the time the split is redeemed.
            // Reported as a *narrowing*, not as a raw count: the engine names
            // every column when it wants them all, and "3 of 3" is not a
            // projection pushed down — it is the absence of one.
            // This fixture's bound schema is always its own three columns.
            constexpr size_t kColumns = 3;
            const int64_t narrowed =
                (request.projection_ids.empty() || request.projection_ids.size() >= kColumns)
                    ? 0
                    : static_cast<int64_t>(request.projection_ids.size());
            split.payload =
                encode_range(i, (request.pushdown_filters.empty() ? 0 : 1) * 1000000 + narrowed);
            result.splits.push_back(std::move(split));
        }
        return result;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        if (!params.split_payloads) {
            throw std::runtime_error("table function 'split_echo_filters' is split-only");
        }
        arrow::Int64Builder ordinal;
        arrow::BooleanBuilder saw;
        arrow::Int64Builder narrowing;
        for (const auto& payload : *params.split_payloads) {
            auto observed = decode_range(payload);
            if (!observed) throw std::runtime_error("split_echo_filters: unrecognized payload");
            (void)ordinal.Append(observed->first);
            (void)saw.Append(observed->second >= 1000000);
            (void)narrowing.Append(observed->second % 1000000);
        }
        std::vector<std::shared_ptr<arrow::Array>> columns(3);
        (void)ordinal.Finish(&columns[0]);
        (void)saw.Finish(&columns[1]);
        (void)narrowing.Finish(&columns[2]);
        // Built against the *bound* schema, which projection pushdown may have
        // narrowed: a fixed three-column batch shipped under a two-column
        // schema is exactly the disagreement Arrow does not check.
        const auto& schema = params.output_schema ? params.output_schema : bind({});
        const std::vector<std::string> names{"split_ordinal", "saw_filters", "n_projection"};
        std::vector<std::shared_ptr<arrow::Array>> projected;
        for (const auto& field : schema->fields()) {
            const auto at = std::find(names.begin(), names.end(), field->name());
            if (at == names.end()) {
                throw std::runtime_error("split_echo_filters: bound an unknown column '" +
                                         field->name() + "'");
            }
            projected.push_back(columns[static_cast<size_t>(at - names.begin())]);
        }
        return std::make_unique<OneShotBatch>(
            arrow::RecordBatch::Make(schema, columns[0]->length(), projected));
    }

private:
    class OneShotBatch : public vgi::TableProducer {
    public:
        explicit OneShotBatch(std::shared_ptr<arrow::RecordBatch> batch)
            : batch_(std::move(batch)) {}
        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            auto out = batch_;
            batch_ = nullptr;
            return out;
        }

    private:
        std::shared_ptr<arrow::RecordBatch> batch_;
    };
};

}  // namespace

void register_splits(vgi::Worker& worker) {
    using Shape = SplitFunction::Shape;
    worker.register_table(std::make_shared<SplitFunction>(
        "split_sequence", Shape::Even, "Integers 0..n-1, divided into n contiguous splits"));
    worker.register_table(
        std::make_shared<SplitFunction>("split_empty_ranges", Shape::EmptyRanges,
                                        "Integers 0..n-1, where every other split names no rows"));
    worker.register_table(std::make_shared<SplitFunction>("split_zero", Shape::Zero,
                                                          "Splits that name no rows at all"));
    worker.register_table(std::make_shared<SplitFunction>(
        "split_skewed", Shape::Skewed, "Integers 0..n-1, divided very unevenly"));
    worker.register_table(std::make_shared<SplitFunction>(
        "split_many", Shape::Many, "Integers 0..n-1, divided into many more splits than threads"));
    worker.register_table(std::make_shared<SplitFailAt>());
    worker.register_table(std::make_shared<SplitEndlessCursor>());
    worker.register_table(std::make_shared<SplitDynamicFilter>());
    worker.register_table(std::make_shared<SplitEchoFilters>());
}

}  // namespace example
