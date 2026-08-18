// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The `attach_options` catalog: one option per Arrow type the wire carries.
//
// What it probes is the whole ATTACH-option path — the declared types and
// defaults reaching discovery, the engine validating and casting a user's
// value against them, and the merged result reaching a function *of that
// attachment*, so two attachments of the same catalog see different values.
// `echo_attach_options()` reports them, which is the only way that is
// observable from SQL.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/scalar.h>
#include <arrow/util/decimal.h>

#include <vgi/worker.h>

#include "registry.h"

namespace example {
namespace {

constexpr const char* kCatalog = "attach_options";

// 2026-04-24 as days since the epoch, and 12:34:56 as microseconds of day.
constexpr int32_t kDefaultDate = 20567;
constexpr int64_t kDefaultTimeOfDay = 45296000000;
// The same instant, as microseconds since the epoch.
constexpr int64_t kDefaultTimestamp = 1777034096000000;

template <typename Builder, typename Value>
std::shared_ptr<arrow::Array> one(Value value) {
    Builder builder;
    (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> one_typed(const std::shared_ptr<arrow::DataType>& type,
                                        int64_t value) {
    std::unique_ptr<arrow::ArrayBuilder> raw;
    (void)arrow::MakeBuilder(arrow::default_memory_pool(), type, &raw);
    auto scalar = arrow::MakeScalar(type, value).ValueOrDie();
    (void)raw->AppendScalar(*scalar);
    std::shared_ptr<arrow::Array> array;
    (void)raw->Finish(&array);
    return array;
}

std::shared_ptr<arrow::DataType> list_type() {
    return arrow::list(arrow::field("item", arrow::int64(), /*nullable=*/true));
}

std::shared_ptr<arrow::DataType> struct_type() {
    return arrow::struct_({arrow::field("a", arrow::int64(), /*nullable=*/true),
                           arrow::field("b", arrow::utf8(), /*nullable=*/true)});
}

std::shared_ptr<arrow::Array> list_default() {
    auto items = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder builder(arrow::default_memory_pool(), items, list_type());
    (void)builder.Append();
    for (int64_t value : {1, 2, 3}) (void)items->Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> struct_default() {
    auto a = one<arrow::Int64Builder>(int64_t{1});
    auto b = one<arrow::StringBuilder>("x");
    return std::make_shared<arrow::StructArray>(struct_type(), 1,
                                                std::vector<std::shared_ptr<arrow::Array>>{a, b});
}

std::shared_ptr<arrow::Array> decimal_default() {
    const auto type = arrow::decimal128(18, 4);
    arrow::Decimal128Builder builder(type);
    (void)builder.Append(arrow::Decimal128(1234500));
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// One entry per option, in the order the echo reports them.
std::vector<vgi::AttachOptionSpec> option_specs() {
    const auto timestamp = arrow::timestamp(arrow::TimeUnit::MICRO);
    const auto timestamp_tz = arrow::timestamp(arrow::TimeUnit::MICRO, "UTC");
    return {
        {"opt_bool", "Boolean option", arrow::boolean(), one<arrow::BooleanBuilder>(true), false},
        {"opt_int8", "int8", arrow::int8(), one<arrow::Int8Builder>(int8_t{-8}), false},
        {"opt_int16", "int16", arrow::int16(), one<arrow::Int16Builder>(int16_t{-16}), false},
        {"opt_int32", "int32", arrow::int32(), one<arrow::Int32Builder>(int32_t{-32}), false},
        {"opt_int64", "int64", arrow::int64(), one<arrow::Int64Builder>(int64_t{-64}), false},
        {"opt_uint8", "uint8", arrow::uint8(), one<arrow::UInt8Builder>(uint8_t{8}), false},
        {"opt_uint16", "uint16", arrow::uint16(), one<arrow::UInt16Builder>(uint16_t{16}), false},
        {"opt_uint32", "uint32", arrow::uint32(), one<arrow::UInt32Builder>(uint32_t{32}), false},
        {"opt_uint64", "uint64", arrow::uint64(), one<arrow::UInt64Builder>(uint64_t{64}), false},
        {"opt_float32", "float32", arrow::float32(), one<arrow::FloatBuilder>(1.5f), false},
        {"opt_float64", "float64", arrow::float64(), one<arrow::DoubleBuilder>(2.5), false},
        {"opt_string", "UTF-8 string", arrow::utf8(), one<arrow::StringBuilder>("hello"), false},
        {"opt_blob", "Binary blob", arrow::binary(),
         one<arrow::BinaryBuilder>(std::string("\x00\x01\x02", 3)), false},
        {"opt_date", "Date", arrow::date32(), one<arrow::Date32Builder>(kDefaultDate), false},
        {"opt_time", "Time of day", arrow::time64(arrow::TimeUnit::MICRO),
         one_typed(arrow::time64(arrow::TimeUnit::MICRO), kDefaultTimeOfDay), false},
        {"opt_timestamp", "Naive timestamp", timestamp, one_typed(timestamp, kDefaultTimestamp),
         false},
        {"opt_timestamp_tz", "Timestamp with UTC tz", timestamp_tz,
         one_typed(timestamp_tz, kDefaultTimestamp), false},
        {"opt_decimal", "Decimal(18,4)", arrow::decimal128(18, 4), decimal_default(), false},
        {"opt_list", "List of int64", list_type(), list_default(), false},
        {"opt_struct", "Struct", struct_type(), struct_default(), false},
    };
}

// `echo_attach_options()` — one row, one column per declared option.
class EchoAttachOptions : public vgi::TableFunction {
public:
    std::string name() const override { return "echo_attach_options"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Echoes the options this attachment was made with";
        md.categories = {"diagnostic"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        arrow::FieldVector fields;
        for (const auto& option : option_specs()) {
            fields.push_back(arrow::field(option.name, option.type, /*nullable=*/true));
        }
        return arrow::schema(std::move(fields));
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override { return {1, 1}; }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        if (!params.attach_options) {
            throw std::runtime_error(
                "echo_attach_options: this attachment carries no options, so the ATTACH-time "
                "merge did not reach the function");
        }
        // Built in the bound order rather than the declared one, because the
        // engine may have narrowed and reordered it.
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(params.output_schema->num_fields()));
        for (const auto& field : params.output_schema->fields()) {
            auto column = params.attach_options->GetColumnByName(field->name());
            if (!column) {
                throw std::runtime_error("echo_attach_options: no option named '" + field->name() +
                                         "'");
            }
            columns.push_back(std::move(column));
        }
        return std::make_unique<OneRow>(arrow::RecordBatch::Make(params.output_schema, 1, columns));
    }

private:
    class OneRow : public vgi::TableProducer {
    public:
        explicit OneRow(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            auto batch = batch_;
            batch_ = nullptr;
            return batch;
        }

    private:
        std::shared_ptr<arrow::RecordBatch> batch_;
    };
};

}  // namespace

void register_attach_options(vgi::Worker& worker) {
    auto& model = worker.catalog(kCatalog);
    model.comment = "Catalog declaring one ATTACH option per wire type";
    model.attach_options = option_specs();
    model.schema("main");
    worker.register_table_in(kCatalog, "main", std::make_shared<EchoAttachOptions>());
}

}  // namespace example
