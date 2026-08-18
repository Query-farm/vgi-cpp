// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// String and binary scalar fixtures.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>

#include <vgi_rpc/crypto.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Reuses the RPC framework's SHA-256 rather than a second copy: it is pinned
// against the FIPS 180-4 vectors there, and a hash fixture whose digest only
// agrees with itself proves nothing.
std::array<uint8_t, 32> sha256_bytes(const std::string& input) {
    return vgi_rpc::crypto::sha256(reinterpret_cast<const uint8_t*>(input.data()), input.size());
}

std::string sha256_hex(const std::string& input) {
    const auto digest = sha256_bytes(input);
    return vgi_rpc::crypto::hex_encode(digest.data(), digest.size());
}

// Checked, and casting where it can.
//
// A `static_cast` here is a memory-safety bug, not a shortcut: DuckDB ships an
// ENUM column as `dictionary<int16, utf8>` and a long string column as
// `large_utf8`, and reading either through a StringArray& dereferences an
// offsets pointer that does not exist on the object. Casting first turns both
// into the string column the caller expects.
std::shared_ptr<arrow::Array> as_strings(const std::shared_ptr<arrow::Array>& array) {
    if (array->type()->id() == arrow::Type::STRING) return array;
    return cast_to(array, arrow::utf8());
}

// Apply `fn` to every non-null string, producing a string column.
template <typename Fn>
std::shared_ptr<arrow::Array> map_strings(const std::shared_ptr<arrow::Array>& column, Fn fn) {
    const auto& values = static_cast<const arrow::StringArray&>(*as_strings(column));
    arrow::StringBuilder out;
    (void)out.Reserve(values.length());
    for (int64_t i = 0; i < values.length(); ++i) {
        if (values.IsNull(i)) {
            (void)out.AppendNull();
        } else {
            (void)out.Append(fn(values.GetString(i)));
        }
    }
    std::shared_ptr<arrow::Array> array;
    (void)out.Finish(&array);
    return array;
}

class UpperCase : public vgi::ScalarFunction {
public:
    std::string name() const override { return "upper_case"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Converts string values to uppercase";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "varchar", "String value to uppercase")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return result(params, map_strings(batch->column(0), [](std::string value) {
                          for (auto& c : value) {
                              c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                          }
                          return value;
                      }));
    }
};

// Identity. Zero compute, so a payload sweep over it measures pure wire cost.
class Passthru : public vgi::ScalarFunction {
public:
    std::string name() const override { return "passthru"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Returns the input string unchanged (zero-compute wire probe)";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "varchar", "String value")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return result(params, batch->column(0));
    }
};

class Sha256Hex : public vgi::ScalarFunction {
public:
    std::string name() const override { return "sha256_hex"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Lowercase hex SHA-256 of the UTF-8 string";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "varchar", "String to hash")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return result(params, map_strings(batch->column(0), sha256_hex));
    }
};

// `hash_rounds(value, rounds)` — SHA-256 applied `rounds` times. The rounds
// count is constant, which makes it a compute knob at fixed payload size.
class HashRounds : public vgi::ScalarFunction {
public:
    std::string name() const override { return "hash_rounds"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Apply SHA-256 `rounds` times (key-stretching); rounds is a const compute knob";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "varchar", "String to stretch"),
                vgi::ArgSpec::constant_arg("rounds", 1, "int64", "Number of SHA-256 rounds")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const int64_t rounds = std::max<int64_t>(0, params.arguments.const_int64(1).value_or(1));
        return result(params, map_strings(batch->column(0), [rounds](const std::string& value) {
                          // Each round hashes the previous digest's *bytes*,
                          // not its hex rendering, which is what makes this
                          // key-stretching rather than a hash of a hex string.
                          std::string digest = value;
                          for (int64_t i = 0; i < rounds; ++i) {
                              const auto raw = sha256_bytes(digest);
                              digest.assign(reinterpret_cast<const char*>(raw.data()),
                                            raw.size());
                          }
                          return vgi_rpc::crypto::hex_encode(
                              reinterpret_cast<const uint8_t*>(digest.data()), digest.size());
                      }));
    }
};

// `collatz_steps(n)` — steps to reach 1. Data-dependent per-row work.
class CollatzSteps : public vgi::ScalarFunction {
public:
    std::string name() const override { return "collatz_steps"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Number of Collatz (3n+1) steps to reach 1";
        md.return_type = arrow::int64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Positive integer")};
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
                continue;
            }
            // 128-bit accumulator: 3n+1 spikes above int64 for inputs well
            // within range, and wrapping there would turn a long trajectory
            // into a wrong answer rather than an error.
            __int128 n = values->Value(i);
            if (n <= 0) {
                (void)out.Append(0);
                continue;
            }
            int64_t steps = 0;
            while (n != 1) {
                n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
                ++steps;
            }
            (void)out.Append(steps);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `concat_values(...)` over strings — concatenates varargs, NULL if any is.
class ConcatValuesStr : public vgi::ScalarFunction {
public:
    std::string name() const override { return "concat_values"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Concatenate string varargs";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto values =
            vgi::ArgSpec::column("values", 0, "varchar", "String values to concatenate");
        values.with_varargs();
        return {values};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        // The cast results are held, not just borrowed: a converted column is
        // a fresh array, and taking a pointer into a temporary would dangle.
        std::vector<std::shared_ptr<arrow::Array>> converted;
        std::vector<const arrow::StringArray*> columns;
        converted.reserve(static_cast<size_t>(batch->num_columns()));
        columns.reserve(static_cast<size_t>(batch->num_columns()));
        for (int i = 0; i < batch->num_columns(); ++i) {
            converted.push_back(as_strings(batch->column(i)));
            columns.push_back(static_cast<const arrow::StringArray*>(converted.back().get()));
        }

        arrow::StringBuilder out;
        (void)out.Reserve(batch->num_rows());
        for (int64_t row = 0; row < batch->num_rows(); ++row) {
            bool any_null = false;
            std::string joined;
            for (const auto* column : columns) {
                if (column->IsNull(row)) {
                    any_null = true;
                    break;
                }
                joined += column->GetString(row);
            }
            if (any_null) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(joined);
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `concat_values(...)` over integers — the int overload of the same name.
//
// Two overloads of a varargs function, distinguished only by element type.
// Registering just the string one makes `concat_values(1, 10)` fail to bind
// rather than fall back, because DuckDB resolves before calling.
class ConcatValuesInt : public vgi::ScalarFunction {
public:
    std::string name() const override { return "concat_values"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Sum integer varargs and return as string";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto values = vgi::ArgSpec::column("values", 0, "int64", "Integer values to sum");
        values.with_varargs();
        return {values};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        std::vector<std::shared_ptr<arrow::Int64Array>> columns;
        columns.reserve(static_cast<size_t>(batch->num_columns()));
        for (int i = 0; i < batch->num_columns(); ++i) {
            columns.push_back(std::static_pointer_cast<arrow::Int64Array>(
                cast_to(batch->column(i), arrow::int64())));
        }

        arrow::StringBuilder out;
        (void)out.Reserve(batch->num_rows());
        for (int64_t row = 0; row < batch->num_rows(); ++row) {
            bool any_null = false;
            int64_t total = 0;
            for (const auto& column : columns) {
                if (column->IsNull(row)) {
                    any_null = true;
                    break;
                }
                total += column->Value(row);
            }
            if (any_null) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(std::to_string(total));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

}  // namespace

void register_strings(vgi::Worker& worker) {
    worker.register_scalar(std::make_shared<ConcatValuesInt>());
    worker.register_scalar(std::make_shared<UpperCase>());
    worker.register_scalar(std::make_shared<Passthru>());
    worker.register_scalar(std::make_shared<Sha256Hex>());
    worker.register_scalar(std::make_shared<HashRounds>());
    worker.register_scalar(std::make_shared<CollatzSteps>());
    worker.register_scalar(std::make_shared<ConcatValuesStr>());
}

}  // namespace example
