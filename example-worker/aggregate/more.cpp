// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Aggregate fixtures beyond the three in basic.cpp, each probing one thing the
// simple cases do not: a second value column, a result type that follows the
// input, a result type chosen from a secret, varargs, a string-valued
// accumulation, and publication into the engine's global function namespace.
//
// The numeric states are little-endian bytes for the same reason as basic.cpp:
// state crosses the wire between calls and may be rebuilt in another process,
// so its layout is the contract rather than an implementation choice.

#include <cstdlib>
#include <cstring>
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

std::string encode_f64(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    std::string bytes(sizeof(bits), '\0');
    // Byte-explicit rather than a memcpy of the native representation, so the
    // encoding does not change on a big-endian host.
    for (size_t i = 0; i < sizeof(bits); ++i) {
        bytes[i] = static_cast<char>((bits >> (8 * i)) & 0xFF);
    }
    return bytes;
}

double decode_f64(const std::string& bytes) {
    uint64_t bits = 0;
    for (size_t i = 0; i < sizeof(bits) && i < bytes.size(); ++i) {
        bits |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    double value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// A running integer total, as decimal text. Only global_agg uses it, and its
// state is one small number the byte encodings above would only obscure.
int64_t decode_total(const std::string& state) {
    return state.empty() ? 0 : std::strtoll(state.c_str(), nullptr, 10);
}

std::shared_ptr<arrow::Schema> result_schema(std::shared_ptr<arrow::DataType> type) {
    return arrow::schema({arrow::field("result", std::move(type), /*nullable=*/true)});
}

vgi::FunctionMetadata aggregate_metadata(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.null_handling = vgi::NullHandling::Default;
    return md;
}

// The numeric fixtures below all accumulate into one double, so the merge and
// the finalize are shared rather than written four times.
class DoubleAccumulator : public vgi::AggregateFunction {
public:
    std::string combine(const std::string& target, const std::string& source) const override {
        return encode_f64(decode_f64(target) + decode_f64(source));
    }

protected:
    // One row per group, NULL where the group never accumulated anything. The
    // value type is read from the bound schema rather than assumed to be
    // double: vgi_generic_sum and secret_typed_sum settle it at bind.
    static std::shared_ptr<arrow::RecordBatch> finalize_doubles(
        const std::shared_ptr<arrow::Schema>& output_schema,
        const std::vector<std::optional<std::string>>& states) {
        arrow::DoubleBuilder out;
        (void)out.Reserve(static_cast<int64_t>(states.size()));
        for (const auto& state : states) {
            if (state) {
                (void)out.Append(decode_f64(*state));
            } else {
                (void)out.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        auto column = cast_to(array, output_schema->field(0)->type());
        return arrow::RecordBatch::Make(output_schema, column->length(), {column});
    }

    // Accumulation happens in double whatever the call site shipped, so the
    // same state layout serves an INTEGER column and a DOUBLE one.
    static std::shared_ptr<arrow::DoubleArray> as_doubles(
        const std::shared_ptr<arrow::Array>& column) {
        return std::static_pointer_cast<arrow::DoubleArray>(cast_to(column, arrow::float64()));
    }
};

// `vgi_weighted_sum(value, weight)` — two value columns, which is what it
// exists to prove: the engine ships them side by side in one update batch.
class WeightedSum : public DoubleAccumulator {
public:
    std::string name() const override { return "vgi_weighted_sum"; }

    vgi::FunctionMetadata metadata() const override {
        return aggregate_metadata("Weighted sum of values");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "float64", "Values"),
                vgi::ArgSpec::column("weight", 1, "float64", "Weights")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::float64());
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.size() < 2) return;
        auto values = as_doubles(columns[0]);
        auto weights = as_doubles(columns[1]);
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            // Either NULL and there is no product to fold, so the group must
            // not acquire state on this row.
            if (values->IsNull(i) || weights->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            state = encode_f64(decode_f64(state) + values->Value(i) * weights->Value(i));
        }
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        return finalize_doubles(output_schema, states);
    }
};

// `vgi_generic_sum(value)` — the result type follows the argument, so an
// INTEGER column sums to INTEGER and a DOUBLE column to DOUBLE.
class GenericSum : public DoubleAccumulator {
public:
    std::string name() const override { return "vgi_generic_sum"; }

    vgi::FunctionMetadata metadata() const override {
        return aggregate_metadata("Sum any numeric type");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::any_column("value", 0, "Numeric value")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        // Registration binds with no input schema, and there is no honest
        // answer then. Throwing is how the catalog is told to advertise ANY;
        // the real bind, which has the schema, settles the type.
        if (!params.input_schema || params.input_schema->num_fields() == 0) {
            throw std::runtime_error("vgi_generic_sum: input type deferred to bind");
        }
        return result_schema(params.input_schema->field(0)->type());
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values = as_doubles(columns[0]);
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            state = encode_f64(decode_f64(state) + values->Value(i));
        }
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        return finalize_doubles(output_schema, states);
    }
};

// `secret_typed_sum(value)` — the aggregate × secret intersection: DOUBLE when
// the `vgi_example` secret's `use_ssl` is set, BIGINT otherwise.
class SecretTypedSum : public DoubleAccumulator {
public:
    std::string name() const override { return "secret_typed_sum"; }

    vgi::FunctionMetadata metadata() const override {
        auto md =
            aggregate_metadata("Sum an integer column; the result type is chosen from a secret");
        md.categories = {"aggregate", "secret"};
        // Advertising the lookup is what makes the engine pre-resolve the
        // secret and deliver it on the bind. An aggregate gets secrets at bind
        // and nowhere else, so any decision they drive has to be made here.
        md.required_secrets = {{"vgi_example", std::nullopt, std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::any_column("value", 0, "Integer column to sum")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        // Same registration-time throw as vgi_generic_sum, for the same
        // reason: without the input schema this function reports ANY.
        if (!params.input_schema) {
            throw std::runtime_error("secret_typed_sum: result type deferred to bind");
        }
        const auto use_ssl = params.secrets.typed_field("vgi_example", "use_ssl");
        const bool as_double = use_ssl && (*use_ssl == "true" || *use_ssl == "1");
        return result_schema(as_double ? arrow::float64() : arrow::int64());
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values = as_doubles(columns[0]);
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            state = encode_f64(decode_f64(state) + values->Value(i));
        }
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        return finalize_doubles(output_schema, states);
    }
};

// `vgi_sum_all(cols...)` — one grand total across a variadic column list.
class SumAll : public DoubleAccumulator {
public:
    std::string name() const override { return "vgi_sum_all"; }

    vgi::FunctionMetadata metadata() const override {
        return aggregate_metadata("Sum all numeric columns");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto columns = vgi::ArgSpec::any_column("columns", 0, "Numeric columns");
        columns.with_varargs();
        return {columns};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        // A zero-argument call binds with an empty schema, which has to fail
        // here: nothing downstream would name the function in its complaint.
        // Registration, which binds with no schema at all, is not that case.
        if (params.input_schema && params.input_schema->num_fields() == 0) {
            throw std::invalid_argument("vgi_sum_all requires at least 1 value");
        }
        return result_schema(arrow::float64());
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        std::vector<std::shared_ptr<arrow::DoubleArray>> values;
        values.reserve(columns.size());
        for (const auto& column : columns) values.push_back(as_doubles(column));

        for (int64_t i = 0; i < group_ids.length(); ++i) {
            double row = 0;
            for (const auto& column : values) {
                if (!column->IsNull(i)) row += column->Value(i);
            }
            // The row itself is the value being folded, so the group acquires
            // state for any row it is handed — a NULL column narrows the row
            // total rather than voiding the row.
            auto& state = states[group_ids.Value(i)];
            state = encode_f64(decode_f64(state) + row);
        }
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        return finalize_doubles(output_schema, states);
    }
};

// `vgi_listagg(value)` — the accumulation is a string, so the state is the
// joined text itself rather than a fixed-width encoding of a number.
class ListAgg : public vgi::AggregateFunction {
public:
    std::string name() const override { return "vgi_listagg"; }

    vgi::FunctionMetadata metadata() const override {
        return aggregate_metadata("Concatenate strings with comma separator");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "varchar", "String column")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return result_schema(arrow::utf8());
    }

    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.empty()) return;
        auto values =
            std::static_pointer_cast<arrow::StringArray>(cast_to(columns[0], arrow::utf8()));
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (values->IsNull(i)) continue;
            auto& state = states[group_ids.Value(i)];
            if (!state.empty()) state += ",";
            state += values->GetView(i);
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        if (target.empty()) return source;
        if (source.empty()) return target;
        return target + "," + source;
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array&,
        const std::vector<std::optional<std::string>>& states) const override {
        arrow::StringBuilder out;
        (void)out.Reserve(static_cast<int64_t>(states.size()));
        for (const auto& state : states) {
            // An empty join is indistinguishable from no state, and both mean
            // the group contributed nothing.
            if (state && !state->empty()) {
                (void)out.Append(*state);
            } else {
                (void)out.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return arrow::RecordBatch::Make(output_schema, array->length(), {array});
    }
};

// `global_agg(value)` — the aggregate half of the global-registration probes,
// beside `global_scalar` in scalar/secrets.cpp.
//
// Deliberately not a reuse of vgi_sum: the example catalog is a cross-language
// contract, and a probe that shares an implementation would force every other
// worker to change a function it already ships.
class GlobalAgg : public vgi::AggregateFunction {
public:
    std::string name() const override { return "global_agg"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = aggregate_metadata("Global-registration probe (aggregate)");
        md.return_type = arrow::int64();
        md.categories = {"test", "global"};
        md.examples = {{"SELECT vgi_example_global_agg(v) FROM t",
                        "Aggregate probe published into system.main", std::nullopt}};
        return md;
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

}  // namespace

void register_more_aggregates(vgi::Worker& worker) {
    worker.register_aggregate(std::make_shared<WeightedSum>());
    worker.register_aggregate(std::make_shared<GenericSum>());
    worker.register_aggregate(std::make_shared<SecretTypedSum>());
    worker.register_aggregate(std::make_shared<SumAll>());
    worker.register_aggregate(std::make_shared<ListAgg>());
    worker.register_aggregate(std::make_shared<GlobalAgg>());
}

}  // namespace example
