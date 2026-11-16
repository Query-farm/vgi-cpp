// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The read-only half of the cross-SDK `example.contract` fixture.
//
// Every SDK's example worker serves the same `contract` schema, so the vgi
// suite's `visibility/example_table_only.test` runs unchanged against each:
//
//   contract.good    id, label, rowid (the row id, = id); scanned by
//                    `contract_good_scan`, which pushes projection down
//   contract.ranged  who; scanned by a function named exactly `range`
//
// Both scans are hidden (`table_only`): listed so the engine has their
// capabilities for the tables that name them, but not in duckdb_functions()
// and not callable from SQL. The one named `range` is what proves a hidden
// scan binds as the worker's function rather than DuckDB's `range()`.
//
// The schema has no comment: table/database_tags.test pins exactly the
// commented schemas (`data` and `main`). Write-path tables of the fixture
// (`notes`, `amounts`) are out of scope: this SDK serves read-only catalogs.

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>

#include <vgi/worker.h>

#include "registry.h"

namespace example {
namespace {

constexpr const char* kSchema = "contract";

using Columns = std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>;

std::shared_ptr<arrow::Array> int64_column(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    for (int64_t value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> string_column(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// One batch, then exhaustion.
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

// A no-argument, volatile scan over fixed rows. With projection pushdown the
// engine's `output_schema` names the columns it wants, in its order, and only
// those are emitted — by name, since pushdown reorders as freely as it drops.
class ContractScan : public vgi::TableFunction {
public:
    ContractScan(std::string name, std::shared_ptr<arrow::Schema> schema, Columns columns,
                 bool projection_pushdown)
        : name_(std::move(name)),
          schema_(std::move(schema)),
          columns_(std::move(columns)),
          projection_pushdown_(projection_pushdown) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "example.contract table scan";
        md.stability = vgi::Stability::Volatile;
        md.projection_pushdown = projection_pushdown_;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override { return schema_; }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const int64_t rows = columns_.front().second->length();
        std::vector<std::shared_ptr<arrow::Array>> projected;
        projected.reserve(static_cast<size_t>(params.output_schema->num_fields()));
        for (const auto& field : params.output_schema->fields()) {
            auto found = std::find_if(columns_.begin(), columns_.end(), [&](const auto& entry) {
                return entry.first == field->name();
            });
            if (found == columns_.end()) {
                throw std::invalid_argument(name_ + ": unknown column \"" + field->name() + "\"");
            }
            projected.push_back(found->second);
        }
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, rows, std::move(projected)));
    }

private:
    std::string name_;
    std::shared_ptr<arrow::Schema> schema_;
    Columns columns_;
    bool projection_pushdown_;
};

vgi::CatalogTable table(std::string name, std::string scan_function,
                        std::shared_ptr<arrow::Schema> columns) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.scan_function = std::move(scan_function);
    table.columns = std::move(columns);
    return table;
}

}  // namespace

void register_contract(vgi::Worker& worker) {
    const auto& primary = worker.catalog().name;

    // contract.good: the row id is a declared column marked `is_row_id`
    // (empty value: the key's presence is the signal), holding the id.
    const auto good_schema = arrow::schema({
        arrow::field("id", arrow::int64(), /*nullable=*/true),
        arrow::field("label", arrow::utf8(), /*nullable=*/true),
        arrow::field("rowid", arrow::int64(), /*nullable=*/true)
            ->WithMetadata(arrow::key_value_metadata({"is_row_id"}, {""})),
    });
    const auto ids = int64_column({1, 2, 3});
    worker.register_table_in(
        primary, kSchema,
        std::make_shared<ContractScan>(
            "contract_good_scan", good_schema,
            Columns{{"id", ids}, {"label", string_column({"a", "b", "c"})}, {"rowid", ids}},
            /*projection_pushdown=*/true));

    // contract.ranged: its scan is named `range`. Function identity is
    // (catalog, schema, name), so it lives only in `contract`; no public
    // function of that name is registered anywhere in this worker.
    const auto ranged_schema = arrow::schema({arrow::field("who", arrow::utf8(), true)});
    worker.register_table_in(
        primary, kSchema,
        std::make_shared<ContractScan>("range", ranged_schema,
                                       Columns{{"who", string_column({"vgi example worker"})}},
                                       /*projection_pushdown=*/false));

    // Hidden is keyed by bare name, worker-wide: no other registration uses
    // either name.
    worker.hide_function("contract_good_scan");
    worker.hide_function("range");

    auto& schema = worker.catalog().schema(kSchema);
    schema.tables.push_back(table("good", "contract_good_scan", good_schema));
    schema.tables.push_back(table("ranged", "range", ranged_schema));
}

}  // namespace example
