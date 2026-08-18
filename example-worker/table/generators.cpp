// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Generators that each probe one thing the engine has to get right: a scan
// that fails mid-stream, nested output types, named-parameter binding,
// UNION-typed arguments, and the typed constant-argument getters.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/cast.h>

#include <vgi/worker.h>

namespace example {
namespace {

vgi::FunctionMetadata generator_metadata(std::string description,
                                         std::vector<std::string> categories) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = std::move(categories);
    return md;
}

// `Arguments` has typed readers for int64, double and string but not bool, so
// the raw one-row array is cast here. A failed cast is a missing value, not an
// error, which is what the other readers do too.
std::optional<bool> named_bool(const vgi::Arguments& arguments, const std::string& name) {
    auto array = arguments.named(name);
    if (!array || array->length() < 1 || array->IsNull(0)) return std::nullopt;
    auto casted = arrow::compute::Cast(*array, arrow::boolean());
    if (!casted.ok()) return std::nullopt;
    return std::static_pointer_cast<arrow::BooleanArray>(casted.MoveValueUnsafe())->Value(0);
}

// A producer that emits one prepared batch and stops.
class OneShot : public vgi::TableProducer {
public:
    explicit OneShot(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        auto batch = batch_;
        batch_ = nullptr;
        return batch;
    }

private:
    std::shared_ptr<arrow::RecordBatch> batch_;
};

// `generator_exception(fail_after)` — one row per batch, then a failure.
//
// Mid-stream is the point: the engine has already taken rows and settled the
// schema by the time the throw arrives, which is a different path through the
// protocol than a bind that rejects its arguments.
class GeneratorException : public vgi::TableFunction {
public:
    std::string name() const override { return "generator_exception"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Raises an exception after N batches for testing",
                                  {"testing"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_arg("fail_after", 0, "int64", "Batches before failure")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema,
                                          params.arguments.const_int64(0).value_or(0));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t fail_after)
            : schema_(std::move(schema)), fail_after_(fail_after) {}

        // Never returns null: the scan ends by throwing, never by exhausting.
        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (count_ >= fail_after_) {
                throw std::invalid_argument("Intentional failure after " +
                                            std::to_string(fail_after_) + " batches");
            }
            arrow::Int64Builder builder;
            (void)builder.Append(count_);
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            ++count_;
            return arrow::RecordBatch::Make(schema_, 1, {array});
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t fail_after_;
        int64_t count_ = 0;
    };
};

// `nested_sequence(count, batch_size := …, history_size := 20)` — a struct
// column and a list column, so the wire's nested-type encoding is exercised in
// both directions.
class NestedSequence : public vgi::TableFunction {
public:
    std::string name() const override { return "nested_sequence"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = generator_metadata("Generates a sequence with nested struct and list columns",
                                     {"generator", "utility", "testing"});
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
                vgi::ArgSpec::named("batch_size", "int64", "Batch size for output"),
                vgi::ArgSpec::named("history_size", "int64", "Max items in history list")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema(
            {arrow::field("n", arrow::int64(), /*nullable=*/false),
             arrow::field("metadata",
                          arrow::struct_({arrow::field("index", arrow::int64(), true),
                                          arrow::field("label", arrow::utf8(), true)}),
                          true),
             arrow::field("history", arrow::list(arrow::field("item", arrow::int64(), true)),
                          true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)),
            std::max<int64_t>(1, params.arguments.named_int64("history_size").value_or(20)));
    }

private:
    // The whole result rides in one batch. `batch_size` is declared so the
    // signature matches the other generators, but chunking a nested result
    // would only obscure what this fixture is for.
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t count, int64_t history_size)
            : schema_(std::move(schema)), count_(count), history_size_(history_size) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (done_) return nullptr;
            done_ = true;
            std::vector<std::shared_ptr<arrow::Array>> columns;
            columns.reserve(static_cast<size_t>(schema_->num_fields()));
            for (int i = 0; i < schema_->num_fields(); ++i) {
                columns.push_back(build(schema_->field(i)));
            }
            return arrow::RecordBatch::Make(schema_, count_, columns);
        }

    private:
        // Built per field of the *bound* schema, which projection pushdown may
        // have narrowed — the struct type comes from the field for the same
        // reason, rather than being reconstructed here.
        std::shared_ptr<arrow::Array> build(const std::shared_ptr<arrow::Field>& field) const {
            if (field->name() == "metadata") return build_metadata(field->type());
            if (field->name() == "history") return build_history();
            return build_index();
        }

        std::shared_ptr<arrow::Array> build_index() const {
            arrow::Int64Builder builder;
            (void)builder.Reserve(count_);
            for (int64_t i = 0; i < count_; ++i) (void)builder.Append(i);
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            return array;
        }

        std::shared_ptr<arrow::Array> build_metadata(
            const std::shared_ptr<arrow::DataType>& type) const {
            arrow::StringBuilder labels;
            (void)labels.Reserve(count_);
            for (int64_t i = 0; i < count_; ++i) {
                (void)labels.Append("row_" + std::to_string(i));
            }
            std::shared_ptr<arrow::Array> label_array;
            (void)labels.Finish(&label_array);
            return std::make_shared<arrow::StructArray>(
                type, count_,
                std::vector<std::shared_ptr<arrow::Array>>{build_index(), label_array});
        }

        // Row i lists the last `history_size` values ending at i, so the list
        // lengths ramp up and then stay fixed — a uniform length would hide an
        // offsets bug behind a regular stride.
        std::shared_ptr<arrow::Array> build_history() const {
            arrow::ListBuilder builder(arrow::default_memory_pool(),
                                       std::make_shared<arrow::Int64Builder>());
            auto& values = *static_cast<arrow::Int64Builder*>(builder.value_builder());
            for (int64_t i = 0; i < count_; ++i) {
                (void)builder.Append();
                for (int64_t v = std::max<int64_t>(0, i - history_size_ + 1); v <= i; ++v) {
                    (void)values.Append(v);
                }
            }
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            return array;
        }

        std::shared_ptr<arrow::Schema> schema_;
        int64_t count_;
        int64_t history_size_;
        bool done_ = false;
    };
};

// `named_params_echo(count, greeting := , multiplier := , scale := , enabled := )`
// — every named argument comes back out in its own column, so a test can tell
// which defaults the engine applied and which values it actually forwarded.
class NamedParamsEcho : public vgi::TableFunction {
public:
    std::string name() const override { return "named_params_echo"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Echoes named parameter values in output columns",
                                  {"generator", "testing"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
                vgi::ArgSpec::named("greeting", "varchar", "Greeting text echoed in output"),
                vgi::ArgSpec::named("multiplier", "int64", "Multiplier for value column"),
                vgi::ArgSpec::named("scale", "double", "Scale factor for float_value column"),
                vgi::ArgSpec::named("enabled", "boolean", "Boolean echoed in output")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("id", arrow::int64(), true),
                              arrow::field("greeting", arrow::utf8(), true),
                              arrow::field("value", arrow::int64(), true),
                              arrow::field("float_value", arrow::float64(), true),
                              arrow::field("enabled", arrow::boolean(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto& args = params.arguments;
        return std::make_unique<Producer>(
            params.output_schema, std::max<int64_t>(0, args.const_int64(0).value_or(0)),
            args.named_string("greeting").value_or("hello"),
            args.named_int64("multiplier").value_or(1), args.named_double("scale").value_or(1.0),
            named_bool(args, "enabled").value_or(true));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, std::string greeting,
                 int64_t multiplier, double scale, bool enabled)
            : schema_(std::move(schema)),
              remaining_(rows),
              greeting_(std::move(greeting)),
              multiplier_(multiplier),
              scale_(scale),
              enabled_(enabled) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t n = std::min<int64_t>(remaining_, kBatchSize);

            arrow::Int64Builder ids;
            arrow::StringBuilder greetings;
            arrow::Int64Builder values;
            arrow::DoubleBuilder floats;
            arrow::BooleanBuilder enabled;
            (void)ids.Reserve(n);
            (void)values.Reserve(n);
            (void)floats.Reserve(n);
            (void)enabled.Reserve(n);
            for (int64_t i = cursor_; i < cursor_ + n; ++i) {
                (void)ids.Append(i);
                (void)greetings.Append(greeting_);
                (void)values.Append(i * multiplier_);
                (void)floats.Append(static_cast<double>(i) * scale_);
                (void)enabled.Append(enabled_);
            }

            std::vector<std::shared_ptr<arrow::Array>> columns(5);
            (void)ids.Finish(&columns[0]);
            (void)greetings.Finish(&columns[1]);
            (void)values.Finish(&columns[2]);
            (void)floats.Finish(&columns[3]);
            (void)enabled.Finish(&columns[4]);

            cursor_ += n;
            remaining_ -= n;
            return arrow::RecordBatch::Make(schema_, n, columns);
        }

    private:
        static constexpr int64_t kBatchSize = 2048;

        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        std::string greeting_;
        int64_t multiplier_;
        double scale_;
        bool enabled_;
        int64_t cursor_ = 0;
    };
};

// The argument type of `union_varargs`, spelled as an exact Arrow type so
// DuckDB renders the parameter as `UNION(i BIGINT, s VARCHAR)`. The VGI type
// names have nothing that says "union", and sparse is the only union mode
// DuckDB ever emits over Arrow.
std::shared_ptr<arrow::DataType> union_arg_type() {
    return arrow::sparse_union({arrow::field("i", arrow::int64(), true),
                                arrow::field("s", arrow::utf8(), true)},
                               {0, 1});
}

// The single element of a one-row array, rendered the way the shared test
// expects it.
std::string stringify_member(const std::shared_ptr<arrow::Array>& array) {
    if (array->IsNull(0)) return "NULL";
    if (array->type_id() == arrow::Type::INT64) {
        return std::to_string(std::static_pointer_cast<arrow::Int64Array>(array)->Value(0));
    }
    if (array->type_id() == arrow::Type::STRING) {
        return std::static_pointer_cast<arrow::StringArray>(array)->GetString(0);
    }
    throw std::invalid_argument("union_varargs: unsupported member type " +
                                array->type()->ToString());
}

// `union_varargs(configs...)` — one row per argument: which member of the
// union is live and what it holds.
//
// A union argument's payload is only half the information; the discriminator
// is the other half, and it is the half a naive encoder drops.
class UnionVarargs : public vgi::TableFunction {
public:
    std::string name() const override { return "union_varargs"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata("Echo the active member tag and value of each union vararg",
                                  {"generator", "utility"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto configs = vgi::ArgSpec::constant_typed(
            "configs", 0, union_arg_type(),
            "Union values whose active member tag is echoed back");
        configs.with_varargs();
        return {configs};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("idx", arrow::int64(), true),
                              arrow::field("tag", arrow::utf8(), true),
                              arrow::field("value", arrow::utf8(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        const auto rows = static_cast<int64_t>(params.arguments.positional_count());
        return {rows, rows};
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        arrow::Int64Builder idx;
        arrow::StringBuilder tags;
        arrow::StringBuilder values;
        int64_t rows = 0;
        for (size_t i = 0; i < params.arguments.positional_count(); ++i) {
            auto value = params.arguments.positional(i);
            if (!value) continue;
            auto members = std::dynamic_pointer_cast<arrow::SparseUnionArray>(value);
            if (!members) {
                throw std::invalid_argument("union_varargs: expected a union argument, got " +
                                            value->type()->ToString());
            }
            // The row-0 discriminator names the live child; everything else in
            // a sparse union is padding.
            const int child = members->child_id(0);
            (void)idx.Append(static_cast<int64_t>(i));
            (void)tags.Append(value->type()->field(child)->name());
            (void)values.Append(stringify_member(members->field(child)));
            ++rows;
        }

        std::vector<std::shared_ptr<arrow::Array>> columns(3);
        (void)idx.Finish(&columns[0]);
        (void)tags.Finish(&columns[1]);
        (void)values.Finish(&columns[2]);
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, rows, columns));
    }
};

// The `typed_probe` defaults, shared verbatim with the Rust and Go fixtures so
// that `typed_probe(n)` produces byte-identical rows in all three.
constexpr int64_t kDefaultTsUs = 1767323045000000;  // 2026-01-02T03:04:05Z
constexpr int64_t kDefaultIvMs = 1500;
constexpr uint64_t kDefaultUb = 9;
constexpr double kDefaultF = 2.5;

// A named constant carrying an exact Arrow type. `ArgSpec::named` only takes a
// VGI type name, and the names cannot spell TIMESTAMPTZ, INTERVAL or UBIGINT.
vgi::ArgSpec named_typed(std::string name, std::shared_ptr<arrow::DataType> type,
                         std::string description) {
    auto spec = vgi::ArgSpec::named(std::move(name), "", std::move(description));
    spec.arrow_type = std::move(type);
    return spec;
}

bool readable(const std::shared_ptr<arrow::Array>& array) {
    return array && array->length() >= 1 && !array->IsNull(0);
}

// Unix microseconds from the named TIMESTAMPTZ, which already stores micros.
// A different unit means the engine sent something other than what was
// declared, so fall back to the default rather than silently rescaling.
std::optional<int64_t> named_ts_us(const vgi::Arguments& arguments) {
    auto array = arguments.named("ts");
    if (!readable(array) || array->type_id() != arrow::Type::TIMESTAMP) return std::nullopt;
    const auto& type = static_cast<const arrow::TimestampType&>(*array->type());
    if (type.unit() != arrow::TimeUnit::MICRO) return std::nullopt;
    return std::static_pointer_cast<arrow::TimestampArray>(array)->Value(0);
}

// Whole milliseconds from the named INTERVAL, counting a month as 30 days.
// The normalization is arbitrary but shared across the language fixtures,
// which is what lets one test assert a single number.
std::optional<int64_t> named_iv_ms(const vgi::Arguments& arguments) {
    auto array = arguments.named("iv");
    if (!readable(array) || array->type_id() != arrow::Type::INTERVAL_MONTH_DAY_NANO) {
        return std::nullopt;
    }
    const auto value =
        std::static_pointer_cast<arrow::MonthDayNanoIntervalArray>(array)->Value(0);
    return (static_cast<int64_t>(value.months) * 30 + value.days) * 86400000 +
           value.nanoseconds / 1000000;
}

std::optional<std::string> named_blob(const vgi::Arguments& arguments) {
    auto array = arguments.named("blob");
    if (!readable(array)) return std::nullopt;
    if (array->type_id() == arrow::Type::BINARY) {
        return std::static_pointer_cast<arrow::BinaryArray>(array)->GetString(0);
    }
    if (array->type_id() == arrow::Type::LARGE_BINARY) {
        return std::static_pointer_cast<arrow::LargeBinaryArray>(array)->GetString(0);
    }
    return std::nullopt;
}

// Read as uint64 rather than through `named_int64`, which would fold a UBIGINT
// past 2^63 into a negative — the range this fixture exists to check.
std::optional<uint64_t> named_ub(const vgi::Arguments& arguments) {
    auto array = arguments.named("ub");
    if (!readable(array) || array->type_id() != arrow::Type::UINT64) return std::nullopt;
    return std::static_pointer_cast<arrow::UInt64Array>(array)->Value(0);
}

// `typed_probe(n, ts := , iv := , blob := , ub := , f := )` — echoes the
// less-common constant-argument types back as plain integers and bytes.
//
// Every named argument has a default, so `typed_probe(n)` alone drives the
// default-binding path while passing them drives extraction; the echoed values
// are normalized (micros, milliseconds, raw bytes) so the Rust, Go and Python
// fixtures all answer identically.
class TypedProbe : public vgi::TableFunction {
public:
    std::string name() const override { return "typed_probe"; }

    vgi::FunctionMetadata metadata() const override {
        return generator_metadata(
            "Echoes typed const args (timestamp/interval/blob/ubigint) into typed columns",
            {"generator", "testing"});
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("n", 0, "int64", "Number of rows to emit"),
                named_typed("ts", arrow::timestamp(arrow::TimeUnit::MICRO, "UTC"),
                            "Timestamp const (TIMESTAMPTZ)"),
                named_typed("iv", arrow::month_day_nano_interval(),
                            "Interval const (INTERVAL)"),
                named_typed("blob", arrow::binary(), "Blob const (BLOB)"),
                named_typed("ub", arrow::uint64(), "Unsigned const (UBIGINT)"),
                vgi::ArgSpec::named("f", "double", "Float const (DOUBLE)")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("idx", arrow::uint64(), true),
                              arrow::field("ts_us", arrow::int64(), true),
                              arrow::field("iv_ms", arrow::int64(), true),
                              arrow::field("payload", arrow::binary(), true),
                              arrow::field("ub", arrow::uint64(), true),
                              arrow::field("f", arrow::float64(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto& args = params.arguments;
        return std::make_unique<Producer>(
            params.output_schema, std::max<int64_t>(0, args.const_int64(0).value_or(0)),
            named_ts_us(args).value_or(kDefaultTsUs), named_iv_ms(args).value_or(kDefaultIvMs),
            named_blob(args).value_or("vgi"), named_ub(args).value_or(kDefaultUb),
            args.named_double("f").value_or(kDefaultF));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t ts_us,
                 int64_t iv_ms, std::string payload, uint64_t ub, double f)
            : schema_(std::move(schema)),
              remaining_(rows),
              ts_us_(ts_us),
              iv_ms_(iv_ms),
              payload_(std::move(payload)),
              ub_(ub),
              f_(f) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t n = std::min<int64_t>(remaining_, kBatchSize);

            arrow::UInt64Builder idx;
            arrow::Int64Builder ts;
            arrow::Int64Builder iv;
            arrow::BinaryBuilder payload;
            arrow::UInt64Builder ub;
            arrow::DoubleBuilder f;
            (void)idx.Reserve(n);
            (void)ts.Reserve(n);
            (void)iv.Reserve(n);
            (void)ub.Reserve(n);
            (void)f.Reserve(n);
            for (int64_t i = cursor_; i < cursor_ + n; ++i) {
                (void)idx.Append(static_cast<uint64_t>(i));
                (void)ts.Append(ts_us_);
                (void)iv.Append(iv_ms_);
                (void)payload.Append(payload_);
                (void)ub.Append(ub_);
                // The only column that varies down the batch, so a producer
                // that repeated row 0 forever would still be caught.
                (void)f.Append(f_ + static_cast<double>(i));
            }

            std::vector<std::shared_ptr<arrow::Array>> columns(6);
            (void)idx.Finish(&columns[0]);
            (void)ts.Finish(&columns[1]);
            (void)iv.Finish(&columns[2]);
            (void)payload.Finish(&columns[3]);
            (void)ub.Finish(&columns[4]);
            (void)f.Finish(&columns[5]);

            cursor_ += n;
            remaining_ -= n;
            return arrow::RecordBatch::Make(schema_, n, columns);
        }

    private:
        static constexpr int64_t kBatchSize = 2048;

        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t ts_us_;
        int64_t iv_ms_;
        std::string payload_;
        uint64_t ub_;
        double f_;
        int64_t cursor_ = 0;
    };
};

}  // namespace

void register_generators(vgi::Worker& worker) {
    worker.register_table(std::make_shared<GeneratorException>());
    worker.register_table(std::make_shared<NestedSequence>());
    worker.register_table(std::make_shared<NamedParamsEcho>());
    worker.register_table(std::make_shared<UnionVarargs>());
    worker.register_table(std::make_shared<TypedProbe>());
}

}  // namespace example
