// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Aggregate fixtures.
//
// Every one of these encodes its state as little-endian bytes. That is not
// gratuitous: aggregate state crosses the wire between calls and, on the HTTP
// transport, may be rebuilt in another process, so a fixed byte layout is the
// contract rather than an implementation choice.

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

std::string encode_i64(int64_t value) {
    std::string bytes(sizeof(value), '\0');
    // Little-endian by construction rather than by memcpy of the native
    // representation, so the encoding does not change on a big-endian host.
    for (size_t i = 0; i < sizeof(value); ++i) {
        bytes[i] = static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 0xFF);
    }
    return bytes;
}

int64_t decode_i64(const std::string& bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value) && i < bytes.size(); ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return static_cast<int64_t>(value);
}

std::string encode_f64(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return encode_i64(static_cast<int64_t>(bits));
}

double decode_f64(const std::string& bytes) {
    const auto bits = static_cast<uint64_t>(decode_i64(bytes));
    double value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::shared_ptr<arrow::Schema> result_schema(std::shared_ptr<arrow::DataType> type) {
    return arrow::schema({arrow::field("result", std::move(type), /*nullable=*/true)});
}

vgi::FunctionMetadata aggregate_metadata(std::string description, vgi::NullHandling null_handling) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.null_handling = null_handling;
    return md;
}

// `vgi_sum(value)` — DEFAULT null handling, so a group of only NULLs never
// acquires state and finalizes to NULL rather than 0.
class Sum : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_sum"; }

    vgi::FunctionMetadata metadata() const override {
        return aggregate_metadata("Sum integer values", vgi::NullHandling::Default);
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Column to sum")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::int64());
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(columns[0], arrow::int64()));
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            // Skipping nulls *before* touching `states` is what keeps an
            // all-NULL group stateless, and so NULL rather than 0.
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            state = encode_i64(decode_i64(state) + values->Value(i));
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        return encode_i64(decode_i64(target) + decode_i64(source));
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array& group_ids,
        const std::vector<std::optional<std::string>>& states) const override {
        arrow::Int64Builder out;
        (void)out.Reserve(group_ids.length());
        for (size_t i = 0; i < states.size(); ++i) {
            if (states[i]) {
                (void)out.Append(decode_i64(*states[i]));
            } else {
                (void)out.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }
};

// `vgi_count()` — SPECIAL null handling, so every row is counted including
// those with NULL arguments, and an empty group finalizes to 0 rather than
// NULL.
class Count : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_count"; }

    vgi::FunctionMetadata metadata() const override {
        return aggregate_metadata("Count rows", vgi::NullHandling::Special);
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::int64());
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>&) const override {
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            auto& state = states[group_ids.Value(i)];
            state = encode_i64(decode_i64(state) + 1);
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        return encode_i64(decode_i64(target) + decode_i64(source));
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array& group_ids,
        const std::vector<std::optional<std::string>>& states) const override {
        arrow::Int64Builder out;
        (void)out.Reserve(group_ids.length());
        for (const auto& state : states) {
            // A count with no state is zero, not NULL — the difference from
            // vgi_sum, and the reason both fixtures exist.
            (void)out.Append(state ? decode_i64(*state) : 0);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }
};

// `vgi_avg(value)` — state is a running total and a count, side by side.
class Avg : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_avg"; }

    vgi::FunctionMetadata metadata() const override {
        return aggregate_metadata("Average of integer values", vgi::NullHandling::Default);
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Column to average")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::float64());
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(columns[0], arrow::int64()));
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            auto [total, count] = decode(state);
            state = encode(total + static_cast<double>(values->Value(i)), count + 1);
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        auto [target_total, target_count] = decode(target);
        auto [source_total, source_count] = decode(source);
        return encode(target_total + source_total, target_count + source_count);
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array& group_ids,
        const std::vector<std::optional<std::string>>& states) const override {
        arrow::DoubleBuilder out;
        (void)out.Reserve(group_ids.length());
        for (const auto& state : states) {
            if (!state) {
                (void)out.AppendNull();
                continue;
            }
            auto [total, count] = decode(*state);
            if (count == 0) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(total / static_cast<double>(count));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }

private:
    static std::string encode(double total, int64_t count) {
        return encode_f64(total) + encode_i64(count);
    }
    static std::pair<double, int64_t> decode(const std::string& state) {
        if (state.size() < 16) return {0.0, 0};
        return {decode_f64(state.substr(0, 8)), decode_i64(state.substr(8, 8))};
    }
};

}  // namespace

void register_aggregates(vgi::Worker& worker) {
    worker.register_aggregate(std::make_shared<Sum>());
    worker.register_aggregate(std::make_shared<Count>());
    worker.register_aggregate(std::make_shared<Avg>());
}

}  // namespace example
