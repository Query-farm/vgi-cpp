// © Copyright 2025, 2026 Query Farm LLC - https://query.farm

// Two identically-named scans in different schemas. Each backs a declarative
// table in its own schema, so dispatch has to retain the full schema path.

#include <memory>
#include <string>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/record_batch.h>

#include <vgi/worker.h>

namespace example {
namespace {

class SameNameTableScan : public vgi::TableFunction {
public:
    explicit SameNameTableScan(std::string schema) : schema_(std::move(schema)) {}

    std::string name() const override { return "test_same_name_table_scan"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Schema-disambiguation probe; the " + schema_ + "-schema producer";
        md.categories = {"generator", "testing"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("tag", arrow::utf8(), /*nullable=*/true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        arrow::StringBuilder builder;
        (void)builder.Append(schema_);
        std::shared_ptr<arrow::Array> tag;
        (void)builder.Finish(&tag);
        return std::make_unique<Producer>(arrow::RecordBatch::Make(params.output_schema, 1, {tag}));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        explicit Producer(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            auto result = batch_;
            batch_ = nullptr;
            return result;
        }

    private:
        std::shared_ptr<arrow::RecordBatch> batch_;
    };

    std::string schema_;
};

}  // namespace

void register_same_name_tables(vgi::Worker& worker) {
    worker.register_table_in("example", "main", std::make_shared<SameNameTableScan>("main"));
    worker.register_table_in("example", "data", std::make_shared<SameNameTableScan>("data"));
}

}  // namespace example
