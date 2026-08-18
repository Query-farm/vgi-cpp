// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The catalogs beyond `example` that one worker process also serves.
//
// `twin_a` and `twin_b` both declare a `main` schema holding a scalar of the
// same name: neither the function name nor the schema name distinguishes them,
// so only the attachment can. `narrow_bind` is the fail-closed probe — one of
// its tables advertises two columns and binds to one, which the engine must
// refuse rather than read off the end of the worker's batch.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "registry.h"
#include "scalar/util.h"

namespace example {
namespace {

// `test_same_name_catalog(value)` — tags its answer with the catalog it was
// declared in, so a mis-routed call reads as the wrong tag rather than as a
// plausible answer.
class SameNameCatalog : public vgi::ScalarFunction {
public:
    explicit SameNameCatalog(std::string catalog) : catalog_(std::move(catalog)) {}

    std::string name() const override { return "test_same_name_catalog"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Catalog-disambiguation probe; the " + catalog_ + " implementation";
        md.return_type = arrow::utf8();
        md.categories = {"test"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Value to tag")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto values =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(0), arrow::int64()));
        arrow::StringBuilder tagged;
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)tagged.AppendNull();
            } else {
                (void)tagged.Append(catalog_ + ":" + std::to_string(values->Value(i)));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)tagged.Finish(&array);
        return result(params, array);
    }

private:
    std::string catalog_;
};

// The two `narrow_bind` scans. One binds to the columns its table advertises
// and one deliberately does not: what the second probes is that the engine
// refuses at bind rather than reading past the end of a one-column batch.
class RowScan : public vgi::TableFunction {
public:
    RowScan(std::string name, bool narrow) : name_(std::move(name)), narrow_(narrow) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = narrow_ ? "Binds to fewer columns than its table advertises"
                                 : "Binds to exactly the columns its table advertises";
        md.categories = {"test"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        if (narrow_) return arrow::schema({arrow::field("id", arrow::int64(), true)});
        return arrow::schema(
            {arrow::field("id", arrow::int64(), true), arrow::field("val", arrow::int64(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override {
        return {kRows, kRows};
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        std::vector<std::shared_ptr<arrow::Array>> columns;
        for (const auto& field : params.output_schema->fields()) {
            arrow::Int64Builder builder;
            (void)builder.Reserve(kRows);
            for (int64_t i = 0; i < kRows; ++i) {
                (void)builder.Append(field->name() == "val" ? (i + 1) * 10 : i);
            }
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            columns.push_back(std::move(array));
        }
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, kRows, columns));
    }

private:
    static constexpr int64_t kRows = 3;

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

    std::string name_;
    bool narrow_;
};

std::shared_ptr<arrow::Schema> id_val_columns() {
    return arrow::schema({arrow::field("id", arrow::int64(), /*nullable=*/true),
                          arrow::field("val", arrow::int64(), /*nullable=*/true)});
}

vgi::CatalogTable scanned_by(std::string name, std::string scan_function) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.scan_function = std::move(scan_function);
    table.columns = id_val_columns();
    table.cardinality = 3;
    return table;
}

}  // namespace

void register_extra_catalogs(vgi::Worker& worker) {
    for (const char* name : {"twin_a", "twin_b"}) {
        auto& model = worker.catalog(name);
        model.comment = std::string("Catalog-disambiguation twin ") + name;
        worker.register_scalar_in(name, "main", std::make_shared<SameNameCatalog>(name));
    }

    auto& narrow = worker.catalog("narrow_bind");
    narrow.comment = "Fail-closed probe: a bind that disagrees with the table's columns";
    worker.register_table_in("narrow_bind", "main",
                             std::make_shared<RowScan>("narrow_scan", /*narrow=*/true));
    worker.register_table_in("narrow_bind", "main",
                             std::make_shared<RowScan>("wide_scan", /*narrow=*/false));
    auto& main = narrow.schema("main");
    main.tables.push_back(scanned_by("mismatch", "narrow_scan"));
    main.tables.push_back(scanned_by("consistent", "wide_scan"));
}

}  // namespace example
