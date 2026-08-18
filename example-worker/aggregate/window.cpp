// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Windowed aggregates.
//
// A different shape from the grouped ones: the engine ships a whole partition
// once and then asks for one value per output row over sub-frames within it.
// Nothing accumulates across calls, so these implement `window` and leave the
// five-call surface as the minimum the interface requires.

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// An empty mask means no FILTER clause, so every row counts. Treating empty
// as "nothing passes" would make an unfiltered window return all NULLs.
bool keeps(const std::vector<bool>& mask, int64_t row) {
    if (mask.empty()) return true;
    return static_cast<size_t>(row) < mask.size() && mask[static_cast<size_t>(row)];
}

// A grouped state holding a running int64 total, as text. Text rather than
// packed bytes because these states are small and readable beats compact when
// something goes wrong.
int64_t decode_total(const std::string& state) {
    return state.empty() ? 0 : std::strtoll(state.c_str(), nullptr, 10);
}

std::vector<std::string> split_lines(const std::string& state) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= state.size()) {
        const size_t end = state.find('\n', start);
        if (end == std::string::npos) {
            parts.push_back(state.substr(start));
            break;
        }
        parts.push_back(state.substr(start, end - start));
        start = end + 1;
    }
    return parts;
}

enum class ReduceKind { Median, Listagg };

std::shared_ptr<arrow::RecordBatch> reduce_states(
    const std::shared_ptr<arrow::Schema>& output_schema,
    const std::vector<std::optional<std::string>>& states, ReduceKind kind) {
    if (kind == ReduceKind::Listagg) {
        arrow::StringBuilder out;
        for (const auto& state : states) {
            if (!state || state->empty()) {
                (void)out.AppendNull();
                continue;
            }
            std::string joined;
            for (const auto& part : split_lines(*state)) {
                if (!joined.empty()) joined += ",";
                joined += part;
            }
            (void)out.Append(joined);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }

    arrow::DoubleBuilder out;
    for (const auto& state : states) {
        if (!state || state->empty()) {
            (void)out.AppendNull();
            continue;
        }
        std::vector<double> values;
        for (const auto& part : split_lines(*state)) {
            if (!part.empty()) values.push_back(std::strtod(part.c_str(), nullptr));
        }
        if (values.empty()) {
            (void)out.AppendNull();
            continue;
        }
        std::sort(values.begin(), values.end());
        const size_t middle = values.size() / 2;
        (void)out.Append(values.size() % 2 == 1 ? values[middle]
                                                : (values[middle - 1] + values[middle]) / 2.0);
    }
    std::shared_ptr<arrow::Array> array;
    (void)out.Finish(&array);
    return arrow::RecordBatch::Make(output_schema, array->length(), {array});
}

std::shared_ptr<arrow::Schema> result_schema(std::shared_ptr<arrow::DataType> type) {
    return arrow::schema({arrow::field("result", std::move(type), /*nullable=*/true)});
}

// `vgi_window_sum(value) OVER (…)` — the sum over each output row's frames.
class WindowSum : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_window_sum"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Windowed sum over the frame";
        md.return_type = arrow::int64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Column to sum")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::int64());
    }

    bool supports_window() const override { return true; }

    std::shared_ptr<arrow::Array> window(
        const std::shared_ptr<arrow::RecordBatch>& partition, const std::shared_ptr<arrow::Schema>&,
        const std::vector<std::vector<std::pair<int64_t, int64_t>>>& frames,
        const std::vector<bool>& filter_mask) const override {
        auto values = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(partition->column(0), arrow::int64()));

        arrow::Int64Builder out;
        (void)out.Reserve(static_cast<int64_t>(frames.size()));
        for (const auto& row : frames) {
            // A row with no frames is NULL, not zero: an empty window has no
            // sum, the same as SUM over no rows in SQL.
            if (row.empty()) {
                (void)out.AppendNull();
                continue;
            }
            int64_t total = 0;
            bool any = false;
            for (const auto& [begin, end] : row) {
                // The engine's frame bounds can run past the partition when a
                // window is wider than the data; clamping is cheaper than
                // trusting them.
                const int64_t from = std::max<int64_t>(0, begin);
                const int64_t to = std::min<int64_t>(values->length(), end);
                for (int64_t i = from; i < to; ++i) {
                    if (!keeps(filter_mask, i) || values->IsNull(i)) continue;
                    total += values->Value(i);
                    any = true;
                }
            }
            if (any) {
                (void)out.Append(total);
            } else {
                (void)out.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return array;
    }

    // Also callable grouped.
    //
    // Declaring `supports_window` does not stop the engine using the ordinary
    // aggregate path — a query may do both, and DuckDB picks. Refusing here
    // failed queries that were perfectly well formed.
    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(columns[0], arrow::int64()));
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            state = std::to_string(decode_total(state) + values->Value(i));
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        return std::to_string(decode_total(target) + decode_total(source));
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        arrow::Int64Builder out;
        (void)out.Reserve(static_cast<int64_t>(states.size()));
        for (const auto& state : states) {
            if (state) {
                (void)out.Append(decode_total(*state));
            } else {
                (void)out.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }
};

// `vgi_window_sum_batch(value) OVER (…)` — vgi_window_sum reached through the
// engine's *batched* window call.
//
// Answers identically, which is the point: the engine may cover a run of
// output rows with one `aggregate_window_batch` request instead of one
// `aggregate_window` per row, and the two paths must not disagree. The
// framework hands both shapes to the same `window`, so the only thing this
// fixture adds is a second name for the engine to route down the batched path.
class WindowSumBatch : public WindowSum {
public:
    std::string name() const override { return "vgi_window_sum_batch"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Windowed sum answered through the batched window call";
        md.return_type = arrow::int64();
        return md;
    }
};

// `vgi_window_median(value) OVER (…)` — the median of each output row's
// frames, over doubles.
//
// Median rather than another sum because it is *not* decomposable: it needs
// every value in the frame at once, which is exactly what the windowed shape
// gives and the grouped one cannot.
class WindowMedian : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_window_median"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Windowed median over the frame";
        md.return_type = arrow::float64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "float64", "Column to take the median of")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::float64());
    }

    bool supports_window() const override { return true; }

    std::shared_ptr<arrow::Array> window(
        const std::shared_ptr<arrow::RecordBatch>& partition, const std::shared_ptr<arrow::Schema>&,
        const std::vector<std::vector<std::pair<int64_t, int64_t>>>& frames,
        const std::vector<bool>& filter_mask) const override {
        auto values = std::static_pointer_cast<arrow::DoubleArray>(
            cast_to(partition->column(0), arrow::float64()));

        arrow::DoubleBuilder out;
        (void)out.Reserve(static_cast<int64_t>(frames.size()));
        for (const auto& row : frames) {
            std::vector<double> window;
            for (const auto& [begin, end] : row) {
                const int64_t from = std::max<int64_t>(0, begin);
                const int64_t to = std::min<int64_t>(values->length(), end);
                for (int64_t i = from; i < to; ++i) {
                    if (keeps(filter_mask, i) && !values->IsNull(i)) {
                        window.push_back(values->Value(i));
                    }
                }
            }
            if (window.empty()) {
                (void)out.AppendNull();
                continue;
            }
            std::sort(window.begin(), window.end());
            const size_t middle = window.size() / 2;
            // An even count averages the two middle values, which is what SQL
            // and the reference both do.
            (void)out.Append(window.size() % 2 == 1 ? window[middle]
                                                    : (window[middle - 1] + window[middle]) / 2.0);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return array;
    }

    // Neither of these is decomposable, so the grouped state is the values
    // themselves — newline-joined, since the reduction needs them all at
    // finalize and a running summary cannot stand in.
    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values = cast_to(columns[0], arrow::utf8());
        const auto& text = static_cast<const arrow::StringArray&>(*values);
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (text.IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            if (!state.empty()) state += "\n";
            state += text.GetString(i);
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        if (target.empty()) return source;
        if (source.empty()) return target;
        return target + "\n" + source;
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        return reduce_states(output_schema, states, ReduceKind::Median);
    }
};

// `vgi_window_listagg(value) OVER (…)` — the frame's strings, joined.
//
// Order-sensitive where the numeric ones are not: the result depends on the
// order the frame walks its rows, which is what makes it worth having beside
// them.
class WindowListagg : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_window_listagg"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Windowed string concatenation over the frame";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "varchar", "Column to concatenate")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::utf8());
    }

    bool supports_window() const override { return true; }

    std::shared_ptr<arrow::Array> window(
        const std::shared_ptr<arrow::RecordBatch>& partition, const std::shared_ptr<arrow::Schema>&,
        const std::vector<std::vector<std::pair<int64_t, int64_t>>>& frames,
        const std::vector<bool>& filter_mask) const override {
        auto values = cast_to(partition->column(0), arrow::utf8());
        const auto& strings = static_cast<const arrow::StringArray&>(*values);

        arrow::StringBuilder out;
        (void)out.Reserve(static_cast<int64_t>(frames.size()));
        for (const auto& row : frames) {
            std::string joined;
            bool any = false;
            for (const auto& [begin, end] : row) {
                const int64_t from = std::max<int64_t>(0, begin);
                const int64_t to = std::min<int64_t>(strings.length(), end);
                for (int64_t i = from; i < to; ++i) {
                    if (!keeps(filter_mask, i) || strings.IsNull(i)) continue;
                    if (any) joined += ",";
                    joined += strings.GetString(i);
                    any = true;
                }
            }
            if (any) {
                (void)out.Append(joined);
            } else {
                (void)out.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return array;
    }

    // Neither of these is decomposable, so the grouped state is the values
    // themselves — newline-joined, since the reduction needs them all at
    // finalize and a running summary cannot stand in.
    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values = cast_to(columns[0], arrow::utf8());
        const auto& text = static_cast<const arrow::StringArray&>(*values);
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (text.IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            if (!state.empty()) state += "\n";
            state += text.GetString(i);
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        if (target.empty()) return source;
        if (source.empty()) return target;
        return target + "\n" + source;
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        return reduce_states(output_schema, states, ReduceKind::Listagg);
    }
};

// `vgi_streaming_sum(value) OVER (PARTITION BY …)` — a running total that
// continues across chunk boundaries.
//
// The streaming shape exists for exactly this: the engine feeds chunks and
// expects one value per *input row*, with state carried per partition key. A
// grouped aggregate cannot produce a running total, and a windowed one would
// re-scan the frame for every row.
class StreamingSum : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_streaming_sum"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Running sum across PARTITION BY keys via the streaming protocol";
        md.return_type = arrow::int64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Column to sum")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::int64());
    }

    bool streaming_partitioned() const override { return true; }

    std::shared_ptr<arrow::Array> streaming_chunk(
        const std::shared_ptr<arrow::RecordBatch>& chunk, size_t partition_key_count,
        size_t order_key_count, std::map<std::string, std::string>& states) const override {
        // Columns arrive as [partition keys…, order keys…, values…], so the
        // value's index is the sum of the two counts rather than 0.
        const int value_index = static_cast<int>(partition_key_count + order_key_count);
        auto values = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(chunk->column(value_index), arrow::int64()));

        arrow::Int64Builder out;
        (void)out.Reserve(chunk->num_rows());
        for (int64_t row = 0; row < chunk->num_rows(); ++row) {
            const auto key = partition_key(chunk, partition_key_count, row);
            auto& state = states[key];
            int64_t running = state.empty() ? 0 : std::strtoll(state.c_str(), nullptr, 10);
            // A NULL contributes nothing but does not reset: the running total
            // continues, which is what SQL's running SUM does.
            if (!values->IsNull(row)) {
                running += values->Value(row);
                state = std::to_string(running);
            }
            (void)out.Append(running);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return array;
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(columns[0], arrow::int64()));
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            state = std::to_string(decode_total(state) + values->Value(i));
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        return std::to_string(decode_total(target) + decode_total(source));
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        arrow::Int64Builder out;
        for (const auto& state : states) {
            if (state) {
                (void)out.Append(decode_total(*state));
            } else {
                (void)out.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }

private:
    // The partition key's rendered values, joined. Rendered rather than
    // hashed: a hash collision would silently merge two partitions' totals,
    // and these keys are short.
    static std::string partition_key(const std::shared_ptr<arrow::RecordBatch>& chunk,
                                     size_t key_count, int64_t row) {
        std::string key;
        for (size_t i = 0; i < key_count; ++i) {
            auto scalar = chunk->column(static_cast<int>(i))->GetScalar(row);
            key += scalar.ok() ? scalar.ValueUnsafe()->ToString() : std::string{};
            key += "\x1f";
        }
        return key;
    }
};

}  // namespace

void register_window_aggregates(vgi::Worker& worker) {
    worker.register_aggregate(std::make_shared<StreamingSum>());
    worker.register_aggregate(std::make_shared<WindowSum>());
    worker.register_aggregate(std::make_shared<WindowSumBatch>());
    worker.register_aggregate(std::make_shared<WindowMedian>());
    worker.register_aggregate(std::make_shared<WindowListagg>());
}

}  // namespace example
