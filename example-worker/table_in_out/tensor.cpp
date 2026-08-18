// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// `unnest_tensor_rows(data)` — the streaming inverse of the `nest_tensor`
// aggregate: one output *row* per tensor cell.
//
// The scalar `unnest_tensor` in `scalar/secrets.cpp` inverts the same
// structure but answers with a list per input row, so a caller has to UNNEST it
// again. This shape exists because a table-in-out can be driven by LATERAL,
// which is what lets one tensor per group be exploded against its group's outer
// columns — the call shape the fixture is really for.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/util.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

[[noreturn]] void rows_error(const std::string& message) {
    throw std::runtime_error("unnest_tensor_rows: " + message);
}

// One list level stripped, or the type unchanged when it is not a list.
std::shared_ptr<arrow::DataType> element_type(const std::shared_ptr<arrow::DataType>& type) {
    if (type->id() == arrow::Type::LIST || type->id() == arrow::Type::LARGE_LIST) {
        return type->field(0)->type();
    }
    return type;
}

int list_depth(const arrow::DataType& type) {
    int depth = 0;
    const arrow::DataType* current = &type;
    while (current->id() == arrow::Type::LIST || current->id() == arrow::Type::LARGE_LIST) {
        current = current->field(0)->type().get();
        ++depth;
    }
    return depth;
}

std::shared_ptr<arrow::Array> take_indices(const std::shared_ptr<arrow::Array>& values,
                                           const std::shared_ptr<arrow::Array>& indices) {
    // Short-circuited rather than handed to `Take`: gathering nothing walks
    // into Arrow's gather kernel with a zero-length source and index, whose
    // data pointers are null, and it asserts on them. Release builds compile
    // the assert out; a debug or sanitizer build aborts the worker.
    if (indices->length() == 0) {
        auto empty = arrow::MakeArrayOfNull(values->type(), 0);
        if (!empty.ok()) rows_error("gather failed: " + empty.status().ToString());
        return empty.MoveValueUnsafe();
    }
    auto taken = arrow::compute::Take(*values, *indices);
    if (!taken.ok()) rows_error("gather failed: " + taken.status().ToString());
    return taken.MoveValueUnsafe();
}

// `{value, axes}` for a `{tensor, axes}` input column.
//
// The axes struct decides the nesting: one coordinate list per axis, and a
// tensor that must be exactly that many list levels deep. An output axis
// carries one coordinate per cell, so it drops the list wrapper.
std::shared_ptr<arrow::Schema> unnest_row_schema(const std::shared_ptr<arrow::DataType>& input) {
    if (input->id() != arrow::Type::STRUCT) {
        rows_error("input column must be a struct, got " + input->ToString());
    }
    const auto& members = static_cast<const arrow::StructType&>(*input);
    auto tensor = members.GetFieldByName("tensor");
    auto axes = members.GetFieldByName("axes");
    if (!tensor || !axes) rows_error("struct must have 'tensor' and 'axes' fields");
    if (axes->type()->id() != arrow::Type::STRUCT) {
        rows_error("'axes' field must be a struct, got " + axes->type()->ToString());
    }

    const auto& axis_fields = static_cast<const arrow::StructType&>(*axes->type());
    const int n_axes = axis_fields.num_fields();
    const int depth = list_depth(*tensor->type());
    if (depth != n_axes) {
        rows_error("tensor nesting depth " + std::to_string(depth) +
                   " does not match number of axes " + std::to_string(n_axes));
    }

    auto cell = tensor->type();
    for (int i = 0; i < depth; ++i) cell = element_type(cell);

    std::vector<std::shared_ptr<arrow::Field>> out_axes;
    out_axes.reserve(static_cast<size_t>(n_axes));
    for (int a = 0; a < n_axes; ++a) {
        out_axes.push_back(arrow::field(axis_fields.field(a)->name(),
                                        element_type(axis_fields.field(a)->type()),
                                        /*nullable=*/true));
    }
    return arrow::schema({arrow::field("value", cell, /*nullable=*/true),
                          arrow::field("axes", arrow::struct_(out_axes), /*nullable=*/true)});
}

class UnnestTensorRows : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "unnest_tensor_rows"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Invert nest_tensor, streaming one row per cell (LATERAL-friendly)";
        md.categories = {"transform", "tensor"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::table("data", 0, "Relation of nest_tensor structs")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema || params.input_schema->num_fields() != 1) {
            rows_error("input table must have exactly one column (the nest_tensor struct)");
        }
        return unnest_row_schema(params.input_schema->field(0)->type());
    }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto cells = std::dynamic_pointer_cast<arrow::StructArray>(batch->column(0));
        if (!cells) rows_error("input column must be a struct array");
        auto tensor = cells->GetFieldByName("tensor");
        auto axes = std::dynamic_pointer_cast<arrow::StructArray>(cells->GetFieldByName("axes"));
        if (!tensor || !axes) rows_error("input must have 'tensor' and 'axes' fields");
        const int n_axes = axes->num_fields();

        std::vector<std::shared_ptr<arrow::ListArray>> axis_lists;
        std::vector<std::shared_ptr<arrow::Array>> axis_values;
        for (int a = 0; a < n_axes; ++a) {
            auto list = std::dynamic_pointer_cast<arrow::ListArray>(axes->field(a));
            if (!list) rows_error("each axis must be a list of coordinates");
            axis_values.push_back(list->values());
            axis_lists.push_back(std::move(list));
        }

        // Descend one list level per axis to reach the flat leaf values; the
        // levels are kept because each row's leaf window is found by walking
        // their offsets, not by assuming the rows are packed in order.
        std::vector<std::shared_ptr<arrow::ListArray>> levels;
        std::shared_ptr<arrow::Array> leaf = tensor;
        for (int a = 0; a < n_axes; ++a) {
            auto list = std::dynamic_pointer_cast<arrow::ListArray>(leaf);
            if (!list) rows_error("tensor nesting is shallower than the axis count");
            leaf = list->values();
            levels.push_back(std::move(list));
        }

        arrow::Int64Builder value_take;
        std::vector<arrow::Int64Builder> axis_take(static_cast<size_t>(n_axes));
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            // A null tensor names no cells, so the row contributes none — the
            // LATERAL shape of "this outer row has nothing to explode".
            if (cells->IsNull(i)) continue;

            // The shape comes from the axes rather than the tensor: a cell may
            // be null but its coordinates always exist.
            std::vector<int64_t> shape(static_cast<size_t>(n_axes));
            std::vector<int64_t> axis_start(static_cast<size_t>(n_axes));
            int64_t total = 1;
            for (int a = 0; a < n_axes; ++a) {
                const auto axis = static_cast<size_t>(a);
                shape[axis] = axis_lists[axis]->value_length(i);
                axis_start[axis] = axis_lists[axis]->value_offset(i);
                total *= shape[axis];
            }
            if (total == 0) continue;

            int64_t leaf_start = i;
            for (int a = 0; a < n_axes; ++a) leaf_start = levels[a]->value_offset(leaf_start);

            for (int64_t k = 0; k < total; ++k) {
                (void)value_take.Append(leaf_start + k);
                // Decode the row-major cell index back into per-axis
                // coordinates, innermost axis varying fastest.
                int64_t rest = k;
                for (int a = n_axes - 1; a >= 0; --a) {
                    const auto axis = static_cast<size_t>(a);
                    (void)axis_take[axis].Append(axis_start[axis] + rest % shape[axis]);
                    rest /= shape[axis];
                }
            }
        }

        // Built against the *bound* schema, not against the input: the engine
        // owns the output types and an array that disagrees with its own type
        // crashes whatever consumes it later.
        const auto& schema = params.output_schema;
        auto axes_type = schema->GetFieldByName("axes");
        if (!axes_type || axes_type->type()->id() != arrow::Type::STRUCT) {
            rows_error("bound output has no 'axes' struct");
        }
        const auto& out_axes = static_cast<const arrow::StructType&>(*axes_type->type());
        if (out_axes.num_fields() != n_axes) rows_error("bound axes count does not match input");

        std::shared_ptr<arrow::Array> value_indices;
        (void)value_take.Finish(&value_indices);
        auto values = cast_to(take_indices(leaf, value_indices),
                              schema->GetFieldByName("value")->type());

        std::vector<std::shared_ptr<arrow::Array>> axis_columns;
        std::vector<std::shared_ptr<arrow::Field>> axis_fields;
        for (int a = 0; a < n_axes; ++a) {
            std::shared_ptr<arrow::Array> indices;
            (void)axis_take[static_cast<size_t>(a)].Finish(&indices);
            axis_columns.push_back(
                cast_to(take_indices(axis_values[static_cast<size_t>(a)], indices),
                        out_axes.field(a)->type()));
            axis_fields.push_back(out_axes.field(a));
        }
        auto axes_struct = arrow::StructArray::Make(axis_columns, axis_fields);
        if (!axes_struct.ok()) rows_error(axes_struct.status().ToString());

        return {arrow::RecordBatch::Make(schema, values->length(), {values, *axes_struct})};
    }
};

}  // namespace

void register_unnest_tensor_rows(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<UnnestTensorRows>());
}

}  // namespace example
