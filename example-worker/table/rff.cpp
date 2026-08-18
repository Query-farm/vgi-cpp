// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Scans behind the `rff_*` tables, which exercise `required_filters`.
//
// The worker only serves metadata and rows here — the engine is what enforces
// the WHERE requirement, and these fixtures exist to check that it does.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

constexpr int64_t kRows = 3;

std::shared_ptr<arrow::Array> counting_column(int64_t rows, int64_t scale) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(rows);
    for (int64_t i = 0; i < rows; ++i) (void)builder.Append(i * scale);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> int64_values(std::vector<int64_t> values) {
    arrow::Int64Builder builder;
    for (int64_t value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> float_values(std::vector<float> values) {
    arrow::FloatBuilder builder;
    for (float value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// Build one column of the *bound* type, named for the fixture's data.
//
// Driven by the type rather than a fixed shape, because projection narrows
// structs: a fixed two-child array against a one-field struct type disagrees
// with its own type, which Arrow does not check and the consumer does not
// survive.
std::shared_ptr<arrow::Array> build_named(const std::string& path,
                                          const std::shared_ptr<arrow::DataType>& type,
                                          int64_t rows) {
    if (type->id() == arrow::Type::STRUCT) {
        const auto& fields = static_cast<const arrow::StructType&>(*type);
        std::vector<std::shared_ptr<arrow::Array>> children;
        children.reserve(static_cast<size_t>(fields.num_fields()));
        for (int i = 0; i < fields.num_fields(); ++i) {
            children.push_back(build_named(path + "." + fields.field(i)->name(),
                                           fields.field(i)->type(), rows));
        }
        return std::make_shared<arrow::StructArray>(type, rows, children);
    }

    // Values the tests assert on directly, keyed by the dotted path so a
    // narrowed struct still gets the right column.
    if (path == "a" || path == "s.a" || path == "wrapper.mid.leaf") {
        return cast_to(int64_values({1, 2, 3}), type);
    }
    if (path == "b" || path == "s.b") return cast_to(int64_values({10, 20, 30}), type);
    if (path == "other") return cast_to(int64_values({100, 200, 300}), type);
    if (path == "top") return cast_to(int64_values({100, 200, 300}), type);
    if (path == "row_id") {
        std::vector<int64_t> ids;
        for (int64_t i = 0; i < rows; ++i) ids.push_back(i);
        return cast_to(int64_values(std::move(ids)), type);
    }
    if (path == "bbox.xmin") {
        std::vector<float> values;
        for (int64_t i = 0; i < rows; ++i) values.push_back(static_cast<float>(i));
        return float_values(std::move(values));
    }
    if (path == "bbox.ymin") return float_values(std::vector<float>(rows, 2.0f));
    if (path == "bbox.xmax") return float_values(std::vector<float>(rows, 3.0f));
    if (path == "bbox.ymax") return float_values(std::vector<float>(rows, 4.0f));

    return cast_to(counting_column(rows, 10), type);
}

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

// One scan shape for every `rff_*` table: the schema is what distinguishes
// them, and each carries a `required_filters` declaration on the *table*.
class RffScan : public vgi::TableFunction {
public:
    RffScan(std::string name, std::shared_ptr<arrow::Schema> schema, int64_t rows)
        : name_(std::move(name)), schema_(std::move(schema)), rows_(rows) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = name_ + " — required-filter fixture scan";
        md.categories = {"catalog"};
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return schema_;
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override {
        return {rows_, rows_};
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        // Exactly the bound schema, which projection may have narrowed — the
        // row_id virtual column in particular is only present when asked for.
        const auto& schema = params.output_schema;
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(schema->num_fields()));
        for (int i = 0; i < schema->num_fields(); ++i) {
            columns.push_back(
                build_named(schema->field(i)->name(), schema->field(i)->type(), rows_));
        }
        return std::make_unique<OneShot>(arrow::RecordBatch::Make(schema, rows_, columns));
    }

private:
    std::string name_;
    std::shared_ptr<arrow::Schema> schema_;
    int64_t rows_;
};

std::shared_ptr<arrow::Field> struct_ab(const std::string& name) {
    return arrow::field(name,
                        arrow::struct_({arrow::field("a", arrow::int64(), true),
                                        arrow::field("b", arrow::int64(), true)}),
                        true);
}

std::shared_ptr<arrow::Field> bbox() {
    return arrow::field("bbox",
                        arrow::struct_({arrow::field("xmin", arrow::float32(), true),
                                        arrow::field("ymin", arrow::float32(), true),
                                        arrow::field("xmax", arrow::float32(), true),
                                        arrow::field("ymax", arrow::float32(), true)}),
                        true);
}

}  // namespace

void register_rff(vgi::Worker& worker) {
    const auto ab = arrow::schema({arrow::field("a", arrow::int64(), true),
                                   arrow::field("b", arrow::int64(), true)});
    worker.register_table(std::make_shared<RffScan>("rff_simple_scan", ab, kRows));
    worker.register_table(std::make_shared<RffScan>("rff_none_scan", ab, kRows));
    worker.register_table(std::make_shared<RffScan>(
        "rff_struct_scan",
        arrow::schema({struct_ab("s"), arrow::field("other", arrow::int64(), true)}), kRows));
    // Two rows, not three: the fixture's mixed top-level/subfield requirement
    // is checked against a specific pair.
    worker.register_table(std::make_shared<RffScan>(
        "rff_multi_scan",
        arrow::schema({struct_ab("s"), arrow::field("top", arrow::int64(), true)}), 2));
    worker.register_table(std::make_shared<RffScan>(
        "rff_nested_scan",
        arrow::schema({arrow::field(
            "wrapper",
            arrow::struct_({arrow::field(
                "mid", arrow::struct_({arrow::field("leaf", arrow::int64(), true)}), true)}),
            true)}),
        kRows));

    // The row_id column is virtual: the engine only asks for it when a query
    // needs it, which is why this scan must honour projection.
    auto row_id = arrow::field("row_id", arrow::int64(), true)
                      ->WithMetadata(arrow::key_value_metadata({"is_row_id"}, {""}));
    worker.register_table(std::make_shared<RffScan>(
        "rff_rowid_scan",
        arrow::schema({row_id, bbox(), arrow::field("other", arrow::int64(), true)}), 10));
}

// The `rff_*` tables, declared by the catalog.
std::vector<vgi::CatalogTable> rff_tables() {
    const auto table = [](std::string name, std::shared_ptr<arrow::Schema> columns,
                          std::string scan,
                          std::vector<std::vector<std::string>> required) {
        vgi::CatalogTable entry;
        entry.name = std::move(name);
        entry.columns = std::move(columns);
        entry.scan_function = std::move(scan);
        entry.required_filters = std::move(required);
        entry.cardinality = kRows;
        return entry;
    };

    const auto ab = arrow::schema({arrow::field("a", arrow::int64(), true),
                                   arrow::field("b", arrow::int64(), true)});
    std::vector<vgi::CatalogTable> tables;
    tables.push_back(table("rff_simple", ab, "rff_simple_scan", {{"a"}}));
    // One OR-group: a filter on either column satisfies it.
    tables.push_back(table("rff_or", ab, "rff_simple_scan", {{"a", "b"}}));
    tables.push_back(table("rff_none", ab, "rff_none_scan", {}));
    tables.push_back(table(
        "rff_struct", arrow::schema({struct_ab("s"), arrow::field("other", arrow::int64(), true)}),
        "rff_struct_scan", {{"s.a"}, {"s.b"}}));
    auto multi = table(
        "rff_multi", arrow::schema({struct_ab("s"), arrow::field("top", arrow::int64(), true)}),
        "rff_multi_scan", {{"top"}, {"s.a"}});
    multi.cardinality = 2;
    tables.push_back(std::move(multi));
    tables.push_back(table(
        "rff_nested",
        arrow::schema({arrow::field(
            "wrapper",
            arrow::struct_({arrow::field(
                "mid", arrow::struct_({arrow::field("leaf", arrow::int64(), true)}), true)}),
            true)}),
        "rff_nested_scan", {{"wrapper.mid.leaf"}}));

    auto rowid_row_id = arrow::field("row_id", arrow::int64(), true)
                            ->WithMetadata(arrow::key_value_metadata({"is_row_id"}, {""}));
    auto rowid = table("rff_rowid",
                       arrow::schema({rowid_row_id, bbox(),
                                      arrow::field("other", arrow::int64(), true)}),
                       "rff_rowid_scan", {{"row_id"}, {"bbox.xmin"}});
    rowid.cardinality = 10;
    tables.push_back(std::move(rowid));
    return tables;
}

}  // namespace example
