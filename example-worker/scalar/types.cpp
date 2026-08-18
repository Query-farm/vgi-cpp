// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Overload-resolution fixtures.
//
// Each of these registers the *same* function name several times with
// different argument types. The engine picks between them by exact Arrow type,
// which is why they declare `column_typed` rather than a VGI type name: the
// names collapse int32 and int64 into "int64" and the overloads would become
// indistinguishable.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Render row `i` of `array` for the any_mixed label. Only the two cases the
// fixture exercises are handled; anything else is empty, matching the Rust
// worker.
std::string render(const std::shared_ptr<arrow::Array>& array, int64_t i) {
    if (array->type()->id() == arrow::Type::STRING) {
        return static_cast<const arrow::StringArray&>(*array).GetString(i);
    }
    try {
        auto casted = cast_to(array, arrow::int64());
        return std::to_string(static_cast<const arrow::Int64Array&>(*casted).Value(i));
    } catch (const std::exception&) {
        // Neither string nor integer: the fixture has nothing to render, and
        // an empty label is what the Rust worker produces too.
        return {};
    }
}

// Emit one string per row, null where the source row is null.
template <typename Fn>
std::shared_ptr<arrow::Array> label_rows(const arrow::Array& source, Fn fn) {
    arrow::StringBuilder out;
    (void)out.Reserve(source.length());
    for (int64_t i = 0; i < source.length(); ++i) {
        if (source.IsNull(i)) {
            (void)out.AppendNull();
        } else {
            (void)out.Append(fn(i));
        }
    }
    std::shared_ptr<arrow::Array> array;
    (void)out.Finish(&array);
    return array;
}

// `type_info(v)` — returns the label of whichever overload was chosen.
class TypeInfo : public vgi::ScalarFunction {
public:
    TypeInfo(std::string label, std::shared_ptr<arrow::DataType> type)
        : label_(std::move(label)), type_(std::move(type)) {}

    std::string name() const override { return "type_info"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Return type name for " + label_ + " input";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column_typed("v", 0, type_, "Input value")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return result(params, label_rows(*batch->column(0), [this](int64_t) { return label_; }));
    }

private:
    std::string label_;
    std::shared_ptr<arrow::DataType> type_;
};

// `pair_type(a, b)` — resolution across two arguments at once.
class PairType : public vgi::ScalarFunction {
public:
    PairType(std::string label, std::shared_ptr<arrow::DataType> first,
             std::shared_ptr<arrow::DataType> second)
        : label_(std::move(label)), first_(std::move(first)), second_(std::move(second)) {}

    std::string name() const override { return "pair_type"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Return type pair name for " + label_;
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column_typed("a", 0, first_, "First value"),
                vgi::ArgSpec::column_typed("b", 1, second_, "Second value")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return result(params, label_rows(*batch->column(0), [this](int64_t) { return label_; }));
    }

private:
    std::string label_;
    std::shared_ptr<arrow::DataType> first_;
    std::shared_ptr<arrow::DataType> second_;
};

// `any_mixed(a, b)` — one polymorphic argument beside a concrete one, which is
// the case where resolution has to consider both.
class AnyMixed : public vgi::ScalarFunction {
public:
    AnyMixed(std::string label, std::shared_ptr<arrow::DataType> second)
        : label_(std::move(label)), second_(std::move(second)) {}

    std::string name() const override { return "any_mixed"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Any+" + label_ + " dispatch";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::any_column("a", 0, "Any type value"),
                vgi::ArgSpec::column_typed("b", 1, second_, "Second value")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto second = batch->column(1);
        return result(params, label_rows(*second, [this, &second](int64_t i) {
                          return "any+" + label_ + ": " + render(second, i);
                      }));
    }

private:
    std::string label_;
    std::shared_ptr<arrow::DataType> second_;
};

}  // namespace

void register_types(vgi::Worker& worker) {
    worker.register_scalar(std::make_shared<TypeInfo>("int32", arrow::int32()));
    worker.register_scalar(std::make_shared<TypeInfo>("int64", arrow::int64()));
    worker.register_scalar(std::make_shared<TypeInfo>("uint32", arrow::uint32()));
    worker.register_scalar(std::make_shared<TypeInfo>("uint64", arrow::uint64()));
    worker.register_scalar(std::make_shared<TypeInfo>("varchar", arrow::utf8()));

    worker.register_scalar(std::make_shared<PairType>("int+int", arrow::int64(), arrow::int64()));
    worker.register_scalar(std::make_shared<PairType>("str+str", arrow::utf8(), arrow::utf8()));
    worker.register_scalar(std::make_shared<PairType>("int+str", arrow::int64(), arrow::utf8()));

    worker.register_scalar(std::make_shared<AnyMixed>("int", arrow::int64()));
    worker.register_scalar(std::make_shared<AnyMixed>("str", arrow::utf8()));
}

}  // namespace example
