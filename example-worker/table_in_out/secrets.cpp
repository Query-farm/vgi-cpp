// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The secret × table-in-out intersection: a transform whose extra column is a
// resolved secret's value.

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

constexpr const char* kSecretType = "vgi_example";
// The column the fixture appends, and the secret field it carries.
constexpr const char* kSecretColumn = "secret_string";

// `secret_in_out(data)` — every input row, plus the secret's `secret_string`.
//
// What it probes that the table-function fixtures do not: the secret is
// resolved during a bind that must *also* settle an output schema derived from
// the input relation. A bind that answered the secret request alone would lose
// the input columns, leaving the caller with a relation of no columns at all.
class SecretInOut : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "secret_in_out"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Append a resolved secret value to each input row";
        md.categories = {"transform", "secret"};
        md.examples = {{"SELECT * FROM secret_in_out((SELECT 1 AS n))",
                        "Append the secret_string value to each input row", std::nullopt}};
        md.required_secrets = {{kSecretType, std::nullopt, std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Input relation")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema) {
            throw std::invalid_argument("secret_in_out: input_schema is required");
        }
        auto fields = params.input_schema->fields();
        fields.push_back(arrow::field(kSecretColumn, arrow::utf8(), /*nullable=*/true));
        return arrow::schema(std::move(fields));
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        // By type, since the secret's name is the user's to choose. Absent
        // rather than fatal when nothing resolved: the column is nullable and
        // a NULL says "no secret" more usefully than a failed query.
        const auto value = params.secrets.typed_field(kSecretType, kSecretColumn);

        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(params.output_schema->num_fields()));
        for (const auto& field : params.output_schema->fields()) {
            if (field->name() == kSecretColumn) {
                columns.push_back(repeated(value, batch->num_rows()));
                continue;
            }
            // By name, not by position: a pushed-down projection keeps the
            // surviving columns' names but not their indices.
            auto column = batch->GetColumnByName(field->name());
            if (!column) {
                throw std::runtime_error("secret_in_out: no input column named '" +
                                         field->name() + "'");
            }
            columns.push_back(cast_to(column, field->type()));
        }
        return {arrow::RecordBatch::Make(params.output_schema, batch->num_rows(), columns)};
    }

private:
    static std::shared_ptr<arrow::Array> repeated(const std::optional<std::string>& value,
                                                  int64_t rows) {
        arrow::StringBuilder builder;
        (void)builder.Reserve(rows);
        for (int64_t i = 0; i < rows; ++i) {
            if (value) {
                (void)builder.Append(*value);
            } else {
                (void)builder.AppendNull();
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        return array;
    }
};

}  // namespace

void register_secret_table_in_out(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<SecretInOut>());
}

}  // namespace example
