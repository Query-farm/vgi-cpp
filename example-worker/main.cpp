// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The VGI example worker: the fixture set the integration suite in
// ~/Development/vgi runs against.  One binary serves whichever catalog
// VGI_WORKER_CATALOG_NAME names, mirroring vgi-rust's example worker so the
// same wrapper scripts drive either.

#include <cstdlib>
#include <memory>
#include <string>

#include <arrow/array/builder_binary.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <vgi/worker.h>

namespace {

// `upper_case(varchar) -> varchar`.  The smallest function that exercises the
// whole path: bind settles a fixed schema, process maps one column.
class UpperCase : public vgi::ScalarFunction {
public:
    std::string name() const override { return "upper_case"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Convert string values to uppercase";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "varchar", "String to uppercase")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto& col = std::static_pointer_cast<arrow::StringArray>(batch->column(0));
        arrow::StringBuilder out;
        for (int64_t i = 0; i < col->length(); ++i) {
            if (col->IsNull(i)) {
                (void)out.AppendNull();
                continue;
            }
            std::string v = col->GetString(i);
            for (auto& c : v) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            (void)out.Append(v);
        }
        std::shared_ptr<arrow::Array> arr;
        (void)out.Finish(&arr);
        return arrow::RecordBatch::Make(params.output_schema, arr->length(), {arr});
    }
};

}  // namespace

int main(int argc, char** argv) {
    vgi::Worker worker;

    vgi::CatalogModel catalog;
    if (const char* name = std::getenv("VGI_WORKER_CATALOG_NAME")) {
        catalog.name = name;
    } else {
        catalog.name = "example";
    }
    worker.set_catalog(std::move(catalog));

    worker.register_scalar(std::make_shared<UpperCase>());

    worker.run(argc, argv);
}
