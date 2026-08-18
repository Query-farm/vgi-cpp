// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// `nest_tensor(value, {axis: coord, …})` — collect a group's rows into a dense
// N-D tensor plus per-axis sorted coordinate lists.
//
// The one aggregate here whose state is a *table* rather than a summary: the
// cells cannot be placed until every coordinate is known, so update keeps the
// rows and finalize is where the tensor is built. That also puts the two
// error cases — a null coordinate, and two rows claiming one cell — where the
// whole group is visible, which is the only place a duplicate arriving from
// two parallel partitions can be seen at all.
//
// `unnest_tensor` in scalar/secrets.cpp is the inverse and pairs with this.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/util.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/concatenate.h>
#include <arrow/buffer_builder.h>
#include <arrow/compute/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// The prefix the fixture's errors carry, matching the Python and Rust ports so
// the same integration assertions hold across all three.
[[noreturn]] void nest_error(const std::string& message) {
    throw std::runtime_error("NestTensorError: nest_tensor: " + message);
}

std::string encode_batch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(sink, batch->schema()).ValueOrDie();
    (void)writer->WriteRecordBatch(*batch);
    (void)writer->Close();
    return sink->Finish().ValueOrDie()->ToString();
}

std::shared_ptr<arrow::RecordBatch> decode_batch(const std::string& bytes) {
    if (bytes.empty()) return nullptr;
    auto source = std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(bytes));
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!reader.ok()) nest_error("state is not a readable batch");
    std::shared_ptr<arrow::RecordBatch> batch;
    (void)reader.ValueUnsafe()->ReadNext(&batch);
    return batch;
}

std::shared_ptr<arrow::Array> take(const std::shared_ptr<arrow::Array>& values,
                                   const std::shared_ptr<arrow::Array>& indices) {
    // Short-circuited rather than handed to `Take`: gathering nothing walks
    // into Arrow's gather kernel with a zero-length source and index, whose
    // data pointers are null, and it asserts on them. Release builds compile
    // the assert out; a debug or sanitizer build aborts the worker.
    if (indices->length() == 0) {
        auto empty = arrow::MakeArrayOfNull(values->type(), 0);
        if (!empty.ok()) nest_error("gather failed: " + empty.status().ToString());
        return empty.MoveValueUnsafe();
    }
    auto taken = arrow::compute::Take(*values, *indices);
    if (!taken.ok()) nest_error("gather failed: " + taken.status().ToString());
    return taken.MoveValueUnsafe();
}

std::shared_ptr<arrow::Array> concatenate(const std::vector<std::shared_ptr<arrow::Array>>& parts) {
    auto joined = arrow::Concatenate(parts);
    if (!joined.ok()) nest_error("concatenate failed: " + joined.status().ToString());
    return joined.MoveValueUnsafe();
}

std::shared_ptr<arrow::RecordBatch> concatenate_batches(
    const std::shared_ptr<arrow::RecordBatch>& first,
    const std::shared_ptr<arrow::RecordBatch>& second) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(static_cast<size_t>(first->num_columns()));
    for (int i = 0; i < first->num_columns(); ++i) {
        columns.push_back(concatenate({first->column(i), second->column(i)}));
    }
    return arrow::RecordBatch::Make(first->schema(), first->num_rows() + second->num_rows(),
                                    columns);
}

// How many list levels wrap the leaf values.
int list_depth(const arrow::DataType& type) {
    int depth = 0;
    const arrow::DataType* current = &type;
    while (current->id() == arrow::Type::LIST || current->id() == arrow::Type::LARGE_LIST) {
        current = current->field(0)->type().get();
        ++depth;
    }
    return depth;
}

std::shared_ptr<arrow::DataType> innermost(const std::shared_ptr<arrow::DataType>& type) {
    auto current = type;
    while (current->id() == arrow::Type::LIST || current->id() == arrow::Type::LARGE_LIST) {
        current = current->field(0)->type();
    }
    return current;
}

std::shared_ptr<arrow::DataType> nested_list(std::shared_ptr<arrow::DataType> inner, int depth) {
    for (int i = 0; i < depth; ++i) inner = arrow::list(inner);
    return inner;
}

// One list level over `values`, partitioning it by per-entry lengths.
std::shared_ptr<arrow::Array> wrap_one_level(const std::shared_ptr<arrow::Array>& values,
                                             const std::vector<int64_t>& lengths) {
    arrow::TypedBufferBuilder<int32_t> offsets;
    (void)offsets.Reserve(static_cast<int64_t>(lengths.size()) + 1);
    int32_t running = 0;
    (void)offsets.Append(running);
    for (int64_t length : lengths) {
        running += static_cast<int32_t>(length);
        (void)offsets.Append(running);
    }
    std::shared_ptr<arrow::Buffer> buffer;
    (void)offsets.Finish(&buffer);

    // Assembled from buffers rather than a ListBuilder: the child values are
    // already one contiguous array, and a builder would copy every cell again.
    auto data =
        arrow::ArrayData::Make(arrow::list(values->type()), static_cast<int64_t>(lengths.size()),
                               {nullptr, buffer}, {values->data()}, 0);
    return std::make_shared<arrow::ListArray>(data);
}

// The depth-`axes` nested list, one entry per group, over the flat row-major
// leaf values. The innermost level runs fastest, so each level's offsets step
// by that level's extent as many times as the levels outside it have entries.
std::shared_ptr<arrow::Array> build_tensor(std::shared_ptr<arrow::Array> leaf,
                                           const std::vector<std::vector<int64_t>>& shapes,
                                           int axes) {
    for (int level = axes - 1; level >= 0; --level) {
        std::vector<int64_t> lengths;
        for (const auto& shape : shapes) {
            int64_t outer = 1;
            for (int i = 0; i < level; ++i) outer *= shape[static_cast<size_t>(i)];
            lengths.insert(lengths.end(), static_cast<size_t>(outer),
                           shape[static_cast<size_t>(level)]);
        }
        leaf = wrap_one_level(leaf, lengths);
    }
    return leaf;
}

// Per-axis ordinals for one group's rows, plus the row that first carries each
// distinct coordinate.
//
// Ordered by value rather than by arrival so the tensor is the same whichever
// way parallel partitions were combined. Arrow's sort settles the ordering for
// any scalar coord type — integers, strings, dates — which is why the
// comparison goes through it rather than through a type switch here.
struct AxisOrdinals {
    std::vector<int64_t> rank;
    std::vector<int64_t> representatives;
};

AxisOrdinals rank_axis(const std::shared_ptr<arrow::Array>& coords, const std::string& name) {
    const int64_t rows = coords->length();
    for (int64_t i = 0; i < rows; ++i) {
        if (coords->IsNull(i)) nest_error("null coord value for axis '" + name + "'");
    }

    auto sorted = arrow::compute::SortIndices(*coords);
    if (!sorted.ok()) {
        nest_error("axis '" + name + "' is not sortable: " + sorted.status().ToString());
    }
    auto order = std::dynamic_pointer_cast<arrow::UInt64Array>(sorted.MoveValueUnsafe());
    if (!order) nest_error("axis '" + name + "' did not sort to an index array");

    AxisOrdinals ordinals;
    ordinals.rank.assign(static_cast<size_t>(rows), 0);
    std::shared_ptr<arrow::Scalar> previous;
    for (int64_t i = 0; i < rows; ++i) {
        const auto row = static_cast<int64_t>(order->Value(i));
        auto scalar = coords->GetScalar(row);
        if (!scalar.ok()) nest_error("axis '" + name + "' has an unreadable coordinate");
        if (!previous || !scalar.ValueUnsafe()->Equals(*previous)) {
            ordinals.representatives.push_back(row);
            previous = scalar.MoveValueUnsafe();
        }
        ordinals.rank[static_cast<size_t>(row)] =
            static_cast<int64_t>(ordinals.representatives.size()) - 1;
    }
    return ordinals;
}

std::shared_ptr<arrow::Array> index_array(const std::vector<int64_t>& indices) {
    arrow::Int64Builder out;
    (void)out.AppendValues(indices);
    std::shared_ptr<arrow::Array> array;
    (void)out.Finish(&array);
    return array;
}

class NestTensor : public vgi::AggregateFunction {
public:
    std::string name() const override { return "nest_tensor"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Collect rows into a dense N-D tensor plus per-axis coordinates";
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::any_column("value", 0, "Tensor cell value"),
                vgi::ArgSpec::any_column("axes", 1, "Struct of axis coordinates")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        // Registration binds with no input schema; throwing is how the catalog
        // is told to advertise ANY, since both halves of the result type come
        // from the call site.
        if (!params.input_schema || params.input_schema->num_fields() < 2) {
            nest_error("expected 2 arguments (value, axes struct)");
        }
        auto value_type = params.input_schema->field(0)->type();
        auto axes_type = params.input_schema->field(1)->type();
        if (axes_type->id() != arrow::Type::STRUCT) {
            nest_error("second argument must be a struct, got " + axes_type->ToString());
        }
        const auto& axes = static_cast<const arrow::StructType&>(*axes_type);
        if (axes.num_fields() == 0) nest_error("axes struct must have at least one field");

        std::vector<std::shared_ptr<arrow::Field>> axis_fields;
        for (int a = 0; a < axes.num_fields(); ++a) {
            const auto& axis = axes.field(a);
            if (arrow::is_floating(axis->type()->id())) {
                nest_error("axis '" + axis->name() + "' has floating-point type " +
                           axis->type()->ToString() +
                           "; floats are not supported as coord types (NaN breaks equality)");
            }
            if (arrow::is_nested(axis->type()->id())) {
                nest_error("axis '" + axis->name() + "' has nested type " +
                           axis->type()->ToString() + "; only scalar coord types are supported");
            }
            axis_fields.push_back(
                arrow::field(axis->name(), arrow::list(axis->type()), /*nullable=*/true));
        }

        auto result =
            arrow::struct_({arrow::field("tensor", nested_list(value_type, axes.num_fields()),
                                         /*nullable=*/true),
                            arrow::field("axes", arrow::struct_(axis_fields), /*nullable=*/true)});
        return arrow::schema({arrow::field("result", result, /*nullable=*/true)});
    }

    // The rows themselves, as a serialized batch. A cell's position depends on
    // every other row's coordinates, so nothing can be placed until the group
    // is complete.
    void update(std::map<int64_t, std::string>& states, const arrow::Int64Array& group_ids,
                const std::vector<std::shared_ptr<arrow::Array>>& columns) const override {
        if (columns.size() < 2) nest_error("expected 2 arguments (value, axes struct)");
        const auto& values = columns[0];
        auto axes = std::dynamic_pointer_cast<arrow::StructArray>(columns[1]);
        if (!axes) nest_error("axes argument must be a struct array");

        // A null axes struct names no cell, so the row is dropped here rather
        // than carried to finalize: everything the state holds is placeable.
        std::map<int64_t, std::vector<int64_t>> rows_by_group;
        for (int64_t i = 0; i < group_ids.length(); ++i) {
            if (axes->IsNull(i)) continue;
            rows_by_group[group_ids.Value(i)].push_back(i);
        }

        auto schema = arrow::schema({arrow::field("value", values->type(), /*nullable=*/true),
                                     arrow::field("axes", axes->type(), /*nullable=*/true)});
        for (const auto& [group, rows] : rows_by_group) {
            auto indices = index_array(rows);
            auto batch = arrow::RecordBatch::Make(schema, static_cast<int64_t>(rows.size()),
                                                  {take(values, indices), take(axes, indices)});
            auto& state = states[group];
            if (!state.empty()) batch = concatenate_batches(decode_batch(state), batch);
            state = encode_batch(batch);
        }
    }

    std::string combine(const std::string& target, const std::string& source) const override {
        if (source.empty()) return target;
        if (target.empty()) return source;
        return encode_batch(concatenate_batches(decode_batch(target), decode_batch(source)));
    }

    std::shared_ptr<arrow::RecordBatch> finalize(
        const std::shared_ptr<arrow::Schema>& output_schema, const arrow::Int64Array& group_ids,
        const std::vector<std::optional<std::string>>& states) const override {
        // Every array below is shaped from the *bound* type rather than from
        // the input: an array that disagrees with its own type is not caught
        // by Arrow and crashes whatever consumes it later.
        auto result_type = output_schema->field(0)->type();
        if (result_type->id() != arrow::Type::STRUCT) {
            nest_error("bound result is not a struct");
        }
        const auto& result_fields = static_cast<const arrow::StructType&>(*result_type);
        auto tensor_field = result_fields.GetFieldByName("tensor");
        auto axes_field = result_fields.GetFieldByName("axes");
        if (!tensor_field || !axes_field || axes_field->type()->id() != arrow::Type::STRUCT) {
            nest_error("bound result must have 'tensor' and an 'axes' struct");
        }
        const auto& axes_fields = static_cast<const arrow::StructType&>(*axes_field->type());
        const int n_axes = axes_fields.num_fields();
        auto cell_type = innermost(tensor_field->type());
        if (list_depth(*tensor_field->type()) != n_axes) {
            nest_error("bound tensor nesting does not match the axis count");
        }

        std::vector<std::shared_ptr<arrow::Array>> leaves;
        std::vector<std::vector<int64_t>> shapes;
        std::vector<std::vector<std::shared_ptr<arrow::Array>>> coords(static_cast<size_t>(n_axes));

        for (const auto& state : states) {
            auto rows = state && !state->empty() ? decode_batch(*state) : nullptr;
            if (!rows || rows->num_rows() == 0) {
                // A group that folded nothing is a zero-extent tensor with
                // empty axes, not a NULL: the struct itself still exists.
                shapes.emplace_back(static_cast<size_t>(n_axes), 0);
                leaves.push_back(arrow::MakeEmptyArray(cell_type).ValueOrDie());
                for (int a = 0; a < n_axes; ++a) {
                    coords[static_cast<size_t>(a)].push_back(
                        arrow::MakeEmptyArray(innermost(axes_fields.field(a)->type()))
                            .ValueOrDie());
                }
                continue;
            }
            append_group(*rows, axes_fields, n_axes, leaves, shapes, coords);
        }

        auto leaf = cast_to(concatenate(leaves), cell_type);
        std::vector<std::shared_ptr<arrow::Array>> axis_columns;
        for (int a = 0; a < n_axes; ++a) {
            std::vector<int64_t> lengths;
            lengths.reserve(shapes.size());
            for (const auto& shape : shapes) lengths.push_back(shape[static_cast<size_t>(a)]);
            auto flat = cast_to(concatenate(coords[static_cast<size_t>(a)]),
                                innermost(axes_fields.field(a)->type()));
            axis_columns.push_back(wrap_one_level(flat, lengths));
        }

        std::vector<std::shared_ptr<arrow::Field>> axis_field_list;
        for (int a = 0; a < n_axes; ++a) axis_field_list.push_back(axes_fields.field(a));
        auto axes_struct = arrow::StructArray::Make(axis_columns, axis_field_list);
        if (!axes_struct.ok()) nest_error(axes_struct.status().ToString());
        auto result = arrow::StructArray::Make({build_tensor(leaf, shapes, n_axes), *axes_struct},
                                               {tensor_field, axes_field});
        if (!result.ok()) nest_error(result.status().ToString());
        return arrow::RecordBatch::Make(output_schema, group_ids.length(), {*result});
    }

private:
    // One group's rows placed into row-major order, appending its leaf values,
    // its shape and its per-axis coordinates to the running lists.
    static void append_group(const arrow::RecordBatch& rows, const arrow::StructType& axes_fields,
                             int n_axes, std::vector<std::shared_ptr<arrow::Array>>& leaves,
                             std::vector<std::vector<int64_t>>& shapes,
                             std::vector<std::vector<std::shared_ptr<arrow::Array>>>& coords) {
        auto axes = std::dynamic_pointer_cast<arrow::StructArray>(rows.column(1));
        if (!axes || axes->num_fields() != n_axes) nest_error("held rows have the wrong axes");

        std::vector<int64_t> shape;
        std::vector<AxisOrdinals> ordinals;
        int64_t cells = 1;
        for (int a = 0; a < n_axes; ++a) {
            ordinals.push_back(rank_axis(axes->field(a), axes_fields.field(a)->name()));
            const auto extent = static_cast<int64_t>(ordinals.back().representatives.size());
            shape.push_back(extent);
            cells *= extent;
            coords[static_cast<size_t>(a)].push_back(
                take(axes->field(a), index_array(ordinals.back().representatives)));
        }

        // A null index gathers a null cell, which is how a sparse tensor's
        // unfilled slots come out NULL rather than absent.
        arrow::Int64Builder gather;
        std::vector<int64_t> source(static_cast<size_t>(cells), -1);
        for (int64_t row = 0; row < rows.num_rows(); ++row) {
            int64_t flat = 0;
            for (int a = 0; a < n_axes; ++a) {
                flat = flat * shape[static_cast<size_t>(a)] +
                       ordinals[static_cast<size_t>(a)].rank[static_cast<size_t>(row)];
            }
            if (source[static_cast<size_t>(flat)] >= 0) {
                nest_error("duplicate coordinate at cell " + std::to_string(flat));
            }
            source[static_cast<size_t>(flat)] = row;
        }
        for (int64_t index : source) {
            if (index < 0) {
                (void)gather.AppendNull();
            } else {
                (void)gather.Append(index);
            }
        }
        std::shared_ptr<arrow::Array> indices;
        (void)gather.Finish(&indices);

        leaves.push_back(take(rows.column(0), indices));
        shapes.push_back(std::move(shape));
    }
};

}  // namespace

void register_nest_tensor(vgi::Worker& worker) {
    worker.register_aggregate(std::make_shared<NestTensor>());
}

}  // namespace example
