// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Fixtures for NULL handling, stability, and seeded generation.
//
// The point of this group is the *metadata*, not the arithmetic: whether the
// engine short-circuits NULLs, whether it may fold a call, and whether a
// per-row generator is re-evaluated. Each is only observable through the
// declared flags, which is what the tests check.

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

// `null_handling(value)` — declares SPECIAL so it is called for NULL rows too,
// which is the only way it can map NULL to a value.
class NullHandling : public vgi::ScalarFunction {
public:
    std::string name() const override { return "null_handling"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Returns value or -5000 if null";
        md.return_type = arrow::int64();
        // Without this the engine never calls the function for a NULL row and
        // the -5000 branch is unreachable.
        md.null_handling = vgi::NullHandling::Special;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Integer value to process")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(0), arrow::int64()));
        arrow::Int64Builder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            (void)out.Append(values->IsNull(i) ? -5000 : values->Value(i));
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `conditional_message(repeat_count, message, condition)` — two constants and
// one column. Column 0 of the input batch is `condition`, not `repeat_count`:
// constants never ship per row.
class ConditionalMessage : public vgi::ScalarFunction {
public:
    std::string name() const override { return "conditional_message"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Returns repeated message when condition is true";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_arg("repeat_count", 0, "int64", "Number of times to repeat"),
            vgi::ArgSpec::constant_arg("message", 1, "varchar", "Message to repeat"),
            vgi::ArgSpec::column("condition", 2, "bool", "Apply message condition"),
        };
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t count = std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        const auto message = params.arguments.const_string(1).value_or("");
        std::string repeated;
        repeated.reserve(message.size() * static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) repeated += message;

        auto condition = std::static_pointer_cast<arrow::BooleanArray>(
            cast_to(batch->column(0), arrow::boolean()));
        arrow::StringBuilder out;
        (void)out.Reserve(condition->length());
        for (int64_t i = 0; i < condition->length(); ++i) {
            const bool on = !condition->IsNull(i) && condition->Value(i);
            (void)out.Append(on ? repeated : std::string{});
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `hash_seed(seed)` — seed + row index. Deterministic, so the tests can assert
// exact values.
class HashSeed : public vgi::ScalarFunction {
public:
    std::string name() const override { return "hash_seed"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Generate deterministic integers from a constant seed";
        md.return_type = arrow::int64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("seed", 0, "int64", "Seed value")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t seed = params.arguments.const_int64(0).value_or(0);
        arrow::Int64Builder out;
        (void)out.Reserve(batch->num_rows());
        for (int64_t i = 0; i < batch->num_rows(); ++i) (void)out.Append(seed + i);
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `query_seed(value)` — the only fixture declaring CONSISTENT_WITHIN_QUERY.
// The offset is a fixed constant so the SQL tests have a stable expectation;
// the stability flag is what is under test, not the arithmetic.
class QuerySeed : public vgi::ScalarFunction {
public:
    std::string name() const override { return "query_seed"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Add a per-query-stable seed to each value (demonstrates "
            "CONSISTENT_WITHIN_QUERY stability)";
        md.return_type = arrow::int64();
        md.stability = vgi::Stability::ConsistentWithinQuery;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Value to offset")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(0), arrow::int64()));
        arrow::Int64Builder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(values->Value(i) + 1000);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `bernoulli()` — VOLATILE. Without that flag the engine evaluates once and
// repeats the answer for every row.
class Bernoulli : public vgi::ScalarFunction {
public:
    std::string name() const override { return "bernoulli"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Generate random booleans (demonstrates VOLATILE stability)";
        md.return_type = arrow::boolean();
        md.stability = vgi::Stability::Volatile;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        Rng rng(volatile_seed());
        arrow::BooleanBuilder out;
        (void)out.Reserve(batch->num_rows());
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            (void)out.Append((rng.next() & 1) == 1);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `random_int(min_val, max_val)` — per-row bounds, so both are columns.
class RandomInt : public vgi::ScalarFunction {
public:
    std::string name() const override { return "random_int"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Generate random integers (demonstrates VOLATILE stability)";
        md.return_type = arrow::int64();
        md.stability = vgi::Stability::Volatile;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("min_val", 0, "int64", "Minimum value (inclusive)"),
                vgi::ArgSpec::column("max_val", 1, "int64", "Maximum value (inclusive)")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto low =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(0), arrow::int64()));
        auto high =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(1), arrow::int64()));
        Rng rng(volatile_seed());
        arrow::Int64Builder out;
        (void)out.Reserve(low->length());
        for (int64_t i = 0; i < low->length(); ++i) {
            if (low->IsNull(i) || high->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(rng.range(low->Value(i), high->Value(i)));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `random_bytes(seed, byte_length)` — both constant, so the output is
// deterministic and the tests can assert exact blobs.
class RandomBytes : public vgi::ScalarFunction {
public:
    std::string name() const override { return "random_bytes"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Generate pseudo-random binary blobs from seed and length";
        md.return_type = arrow::binary();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_arg("seed", 0, "int64",
                                       "Seed for pseudo-random byte generation"),
            vgi::ArgSpec::constant_arg("byte_length", 1, "int64", "Output blob length in bytes"),
        };
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t seed = params.arguments.const_int64(0).value_or(0);
        const int64_t length = params.arguments.const_int64(1).value_or(0);
        if (length < 0) throw std::invalid_argument("byte_length must be >= 0");

        Rng rng(static_cast<uint64_t>(seed));
        arrow::BinaryBuilder out;
        (void)out.Reserve(batch->num_rows());
        std::string blob(static_cast<size_t>(length), '\0');
        for (int64_t row = 0; row < batch->num_rows(); ++row) {
            for (int64_t i = 0; i < length; ++i) {
                blob[static_cast<size_t>(i)] = static_cast<char>(rng.byte());
            }
            (void)out.Append(blob);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

}  // namespace

void register_seeded(vgi::Worker& worker) {
    worker.register_scalar(std::make_shared<NullHandling>());
    worker.register_scalar(std::make_shared<ConditionalMessage>());
    worker.register_scalar(std::make_shared<HashSeed>());
    worker.register_scalar(std::make_shared<QuerySeed>());
    worker.register_scalar(std::make_shared<Bernoulli>());
    worker.register_scalar(std::make_shared<RandomInt>());
    worker.register_scalar(std::make_shared<RandomBytes>());
}

}  // namespace example
