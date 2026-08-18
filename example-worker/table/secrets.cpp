// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Fixtures that resolve secrets.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// The parameter types, echoed back so a test can confirm the secret's schema
// survived the round trip rather than being flattened to strings.
const char* arrow_type_of(const std::string& field) {
    if (field == "port") return "int32";
    if (field == "use_ssl") return "bool";
    if (field == "timeout") return "double";
    return "string";
}

// `secret_demo()` — one row per field of the resolved `vgi_example` secret.
class SecretDemo : public vgi::TableFunction {
public:
    std::string name() const override { return "secret_demo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Outputs secret contents as key-value rows";
        md.categories = {"generator", "utility"};
        // Asking here is what triggers the engine's two-phase resolution: the
        // first bind returns this list, the engine resolves it, and binds
        // again with the values in `params.secrets`.
        md.required_secrets = {{"vgi_example", std::nullopt, std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("key", arrow::utf8(), true),
                              arrow::field("value", arrow::utf8(), true),
                              arrow::field("arrow_type", arrow::utf8(), true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        std::vector<std::array<std::string, 3>> rows;
        // By *type*, not by name: the user chose the secret's name in
        // `CREATE SECRET`, so a fixture cannot know it. Looking it up by name
        // silently found nothing and emitted zero rows.
        if (const auto* secret = params.secrets.of_type("vgi_example")) {
            for (const auto& [field, value] : *secret) {
                rows.push_back({field, value, arrow_type_of(field)});
            }
        }
        return std::make_unique<Producer>(params.output_schema, std::move(rows));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema,
                 std::vector<std::array<std::string, 3>> rows)
            : schema_(std::move(schema)), rows_(std::move(rows)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (emitted_) return nullptr;
            emitted_ = true;

            std::vector<std::shared_ptr<arrow::Array>> columns(3);
            for (size_t column = 0; column < 3; ++column) {
                arrow::StringBuilder builder;
                (void)builder.Reserve(static_cast<int64_t>(rows_.size()));
                for (const auto& row : rows_) (void)builder.Append(row[column]);
                (void)builder.Finish(&columns[column]);
            }
            return arrow::RecordBatch::Make(schema_, static_cast<int64_t>(rows_.size()),
                                            columns);
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        std::vector<std::array<std::string, 3>> rows_;
        bool emitted_ = false;
    };
};

}  // namespace

void register_secret_fixtures(vgi::Worker& worker) {
    worker.register_table(std::make_shared<SecretDemo>());
}

}  // namespace example
