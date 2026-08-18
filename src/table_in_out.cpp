// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/table_in_out.h"

#include <stdexcept>

#include <arrow/array.h>
#include <arrow/array/builder_base.h>
#include <arrow/compute/cast.h>

namespace vgi {

std::shared_ptr<arrow::Schema> TableInOutFunction::bind(const BindParams& params) const {
    if (!params.input_schema) {
        throw std::invalid_argument("table-in-out function '" + name() +
                                    "' requires an input schema");
    }
    return params.input_schema;
}

std::shared_ptr<arrow::RecordBatch> project_batch(const std::shared_ptr<arrow::RecordBatch>& batch,
                                                  const std::shared_ptr<arrow::Schema>& schema) {
    if (!schema) throw std::invalid_argument("project_batch: null schema");
    if (!batch) throw std::invalid_argument("project_batch: null batch");

    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(static_cast<size_t>(schema->num_fields()));

    for (int i = 0; i < schema->num_fields(); ++i) {
        const auto& field = schema->field(i);
        auto column = batch->GetColumnByName(field->name());
        if (!column) {
            // A column the output schema declares but the input does not carry
            // is all-nulls rather than an error: a projection may name a
            // column the producing side chose not to emit.
            std::unique_ptr<arrow::ArrayBuilder> builder;
            auto status = arrow::MakeBuilder(arrow::default_memory_pool(), field->type(), &builder);
            if (!status.ok()) {
                throw std::runtime_error("project_batch: cannot build nulls for '" + field->name() +
                                         "': " + status.ToString());
            }
            if (!builder->AppendNulls(batch->num_rows()).ok()) {
                throw std::runtime_error("project_batch: cannot append nulls for '" +
                                         field->name() + "'");
            }
            auto result = builder->Finish();
            if (!result.ok()) {
                throw std::runtime_error("project_batch: cannot finish nulls for '" +
                                         field->name() + "'");
            }
            columns.push_back(result.MoveValueUnsafe());
            continue;
        }
        if (!column->type()->Equals(*field->type())) {
            auto casted = arrow::compute::Cast(*column, field->type());
            if (!casted.ok()) {
                throw std::runtime_error("project_batch: cannot cast '" + field->name() +
                                         "' from " + column->type()->ToString() + " to " +
                                         field->type()->ToString());
            }
            column = casted.MoveValueUnsafe();
        }
        columns.push_back(std::move(column));
    }

    return arrow::RecordBatch::Make(schema, batch->num_rows(), columns);
}

}  // namespace vgi
