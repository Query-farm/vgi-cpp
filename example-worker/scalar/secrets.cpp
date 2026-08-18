// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Probe fixtures: global registration, the auth principal, constant-argument
// packet assembly, and the tensor inverter.

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/buffer_builder.h>
#include <arrow/compute/api.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Emit `rows` copies of one string — the shape of every probe that answers the
// same value for the whole batch.
std::shared_ptr<arrow::Array> repeat_string(const std::string& value, int64_t rows) {
    arrow::StringBuilder out;
    (void)out.Reserve(rows);
    for (int64_t i = 0; i < rows; ++i) (void)out.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)out.Finish(&array);
    return array;
}

// `global_scalar(value)` — the scalar half of the global-registration probes.
//
// Deliberately its own fixture rather than a reuse of `double` or `passthru`:
// the example catalog is a cross-language contract, and a probe that shares an
// implementation forces every other worker to change a function it already
// ships. The label names this function so a test can prove the globally
// published name reached it and not a same-named function in another catalog.
class GlobalScalar : public vgi::ScalarFunction {
public:
    std::string name() const override { return "global_scalar"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Global-registration probe (scalar)";
        md.return_type = arrow::utf8();
        md.categories = {"test", "global"};
        md.examples = {{"SELECT vgi_example_global_scalar(7)",
                        "Scalar probe published into system.main", std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Value to label")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto values = std::static_pointer_cast<arrow::Int64Array>(
            cast_to(batch->column(0), arrow::int64()));

        arrow::StringBuilder out;
        (void)out.Reserve(values->length());
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append("global_scalar:" + std::to_string(values->Value(i)));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }
};

// `whoami(x)` — the authenticated principal.
//
// Always "anonymous" here: nothing on the call path hands a principal down to
// a scalar, which is also what the Rust worker answers over the subprocess
// transport the integration tests attach. The `x` argument exists only to give
// the call a row count.
class WhoAmI : public vgi::ScalarFunction {
public:
    std::string name() const override { return "whoami"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Return the authenticated principal";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("x", 0, "int64", "dummy input")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return result(params, repeat_string("anonymous", batch->num_rows()));
    }
};

// The `config` constant of `binary_packet`, declared as an Arrow type because
// no VGI type name spells a struct.
std::shared_ptr<arrow::DataType> packet_config_type() {
    return arrow::struct_({arrow::field("label", arrow::utf8(), /*nullable=*/true),
                           arrow::field("version", arrow::int64(), /*nullable=*/true)});
}

// Row 0 of a one-row constant argument, as bytes.
//
// Cast first: a BLOB literal can reach a worker as `large_binary`, and reading
// that through a `BinaryArray&` walks 32-bit offsets over a 64-bit buffer.
std::string const_bytes(const std::shared_ptr<arrow::Array>& array) {
    if (!array || array->length() == 0 || array->IsNull(0)) return {};
    const auto& values =
        static_cast<const arrow::BinaryArray&>(*cast_to(array, arrow::binary()));
    return values.GetString(0);
}

// `binary_packet(header, payload, config)` — header + payload + label + a
// version byte.
//
// Two of its three parameters are constants, which is the point: the header
// and the config struct are read once during process from the argument list,
// while only the payload arrives as a column.
class BinaryPacket : public vgi::ScalarFunction {
public:
    std::string name() const override { return "binary_packet"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Build binary packets with header, payload, and config";
        md.return_type = arrow::binary();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_typed("header", 0, arrow::binary(),
                                         "Header bytes to prepend"),
            vgi::ArgSpec::column("payload", 1, "binary", "Binary payload data"),
            vgi::ArgSpec::constant_typed("config", 2, packet_config_type(),
                                         "Config {label, version}"),
        };
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const std::string header = const_bytes(params.arguments.positional(0));
        const std::string suffix = packet_suffix(params.arguments.positional(2));

        const auto& payload = static_cast<const arrow::BinaryArray&>(
            *cast_to(batch->column(0), arrow::binary()));

        arrow::BinaryBuilder out;
        (void)out.Reserve(payload.length());
        for (int64_t i = 0; i < payload.length(); ++i) {
            std::string packet = header;
            // A null payload contributes nothing but still yields a packet:
            // the header and suffix are constants, so the row is not null.
            if (!payload.IsNull(i)) packet += payload.GetString(i);
            packet += suffix;
            (void)out.Append(packet);
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }

private:
    // The trailer: the config's label, then its version truncated to one byte.
    static std::string packet_suffix(const std::shared_ptr<arrow::Array>& config) {
        std::string suffix;
        int64_t version = 0;

        auto fields = std::dynamic_pointer_cast<arrow::StructArray>(config);
        if (fields && fields->length() > 0) {
            if (auto label = fields->GetFieldByName("label")) {
                const auto& labels =
                    static_cast<const arrow::StringArray&>(*cast_to(label, arrow::utf8()));
                if (!labels.IsNull(0)) suffix = labels.GetString(0);
            }
            if (auto declared = fields->GetFieldByName("version")) {
                const auto& versions = static_cast<const arrow::Int64Array&>(
                    *cast_to(declared, arrow::int64()));
                if (!versions.IsNull(0)) version = versions.Value(0);
            }
        }
        suffix.push_back(static_cast<char>(version & 0xFF));
        return suffix;
    }
};

[[noreturn]] void tensor_error(const std::string& message) {
    throw std::runtime_error("unnest_tensor: " + message);
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

// One list level stripped, or the type unchanged when it is not a list.
std::shared_ptr<arrow::DataType> element_type(const std::shared_ptr<arrow::DataType>& type) {
    if (type->id() == arrow::Type::LIST || type->id() == arrow::Type::LARGE_LIST) {
        return type->field(0)->type();
    }
    return type;
}

// `list<struct{value, axes}>` for a `{tensor, axes}` input.
//
// The axes struct decides the nesting: one coordinate list per axis, and a
// tensor that must be exactly that many list levels deep.
std::shared_ptr<arrow::DataType> unnest_result_type(
    const std::shared_ptr<arrow::DataType>& input) {
    if (input->id() != arrow::Type::STRUCT) {
        tensor_error("argument must be a struct, got " + input->ToString());
    }
    const auto& members = static_cast<const arrow::StructType&>(*input);
    auto tensor = members.GetFieldByName("tensor");
    auto axes = members.GetFieldByName("axes");
    if (!tensor || !axes) tensor_error("struct must have 'tensor' and 'axes' fields");
    if (axes->type()->id() != arrow::Type::STRUCT) {
        tensor_error("'axes' field must be a struct");
    }

    const auto& axis_fields = static_cast<const arrow::StructType&>(*axes->type());
    const int n_axes = axis_fields.num_fields();
    const int depth = list_depth(*tensor->type());
    if (depth != n_axes) {
        tensor_error("tensor nesting depth " + std::to_string(depth) +
                     " does not match number of axes " + std::to_string(n_axes));
    }

    auto cell = tensor->type();
    for (int i = 0; i < depth; ++i) cell = element_type(cell);

    // An output axis carries one coordinate per cell, so it drops the list
    // wrapper the input axis has.
    std::vector<std::shared_ptr<arrow::Field>> out_axes;
    out_axes.reserve(static_cast<size_t>(n_axes));
    for (int a = 0; a < n_axes; ++a) {
        out_axes.push_back(arrow::field(axis_fields.field(a)->name(),
                                        element_type(axis_fields.field(a)->type()),
                                        /*nullable=*/true));
    }

    auto row = arrow::struct_(
        {arrow::field("value", cell, /*nullable=*/true),
         arrow::field("axes", arrow::struct_(out_axes), /*nullable=*/true)});
    return arrow::list(arrow::field("item", row, /*nullable=*/true));
}

std::shared_ptr<arrow::Array> take_indices(const std::shared_ptr<arrow::Array>& values,
                                           const std::shared_ptr<arrow::Array>& indices) {
    auto taken = arrow::compute::Take(*values, *indices);
    if (!taken.ok()) tensor_error("gather failed: " + taken.status().ToString());
    return taken.MoveValueUnsafe();
}

// `unnest_tensor(t)` — invert `nest_tensor`, enumerating every cell of the
// axes' Cartesian product in row-major order.
//
// Null cells are emitted rather than dropped: the point of the fixture is that
// a sparse tensor round-trips, and a missing cell is a coordinate with a null
// value, not an absent row.
class UnnestTensor : public vgi::ScalarFunction {
public:
    std::string name() const override { return "unnest_tensor"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Invert nest_tensor: list of {value, axes} structs per cell";
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::any_column("t", 0, "Struct produced by nest_tensor")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        auto input = params.input_type(0);
        if (!input) tensor_error("requires an input schema");
        return arrow::schema(
            {arrow::field("result", unnest_result_type(input), /*nullable=*/true)});
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto cells = std::dynamic_pointer_cast<arrow::StructArray>(batch->column(0));
        if (!cells) tensor_error("input must be a struct array");
        auto tensor = cells->GetFieldByName("tensor");
        auto axes = std::dynamic_pointer_cast<arrow::StructArray>(
            cells->GetFieldByName("axes"));
        if (!tensor || !axes) tensor_error("input must have 'tensor' and 'axes' fields");
        const int n_axes = axes->num_fields();

        std::vector<std::shared_ptr<arrow::ListArray>> axis_lists;
        std::vector<std::shared_ptr<arrow::Array>> axis_values;
        for (int a = 0; a < n_axes; ++a) {
            auto list = std::dynamic_pointer_cast<arrow::ListArray>(axes->field(a));
            if (!list) tensor_error("each axis must be a list of coordinates");
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
            if (!list) tensor_error("tensor nesting is shallower than the axis count");
            leaf = list->values();
            levels.push_back(std::move(list));
        }

        arrow::Int64Builder value_take;
        std::vector<arrow::Int64Builder> axis_take(static_cast<size_t>(n_axes));
        arrow::TypedBufferBuilder<int32_t> offsets;
        arrow::TypedBufferBuilder<bool> validity;
        (void)offsets.Append(0);

        int32_t emitted = 0;
        int64_t null_rows = 0;
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            if (cells->IsNull(i)) {
                (void)validity.Append(false);
                ++null_rows;
                (void)offsets.Append(emitted);
                continue;
            }
            (void)validity.Append(true);

            // The shape comes from the axes rather than the tensor: a cell may
            // be null but its coordinates always exist.
            std::vector<int64_t> shape(static_cast<size_t>(n_axes));
            std::vector<int64_t> axis_start(static_cast<size_t>(n_axes));
            int64_t total = 1;
            for (int a = 0; a < n_axes; ++a) {
                shape[static_cast<size_t>(a)] = axis_lists[a]->value_length(i);
                axis_start[static_cast<size_t>(a)] = axis_lists[a]->value_offset(i);
                total *= shape[static_cast<size_t>(a)];
            }

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
            emitted += static_cast<int32_t>(total);
            (void)offsets.Append(emitted);
        }

        auto list_type = output_type(params);
        if (list_type->id() != arrow::Type::LIST) tensor_error("bound output is not a list");
        auto row_type = list_type->field(0)->type();
        if (row_type->id() != arrow::Type::STRUCT) {
            tensor_error("bound list element is not a struct");
        }
        const auto& row_fields = static_cast<const arrow::StructType&>(*row_type);

        std::shared_ptr<arrow::Array> value_indices;
        (void)value_take.Finish(&value_indices);
        auto values = cast_to(take_indices(leaf, value_indices), row_fields.field(0)->type());

        const auto& out_axes =
            static_cast<const arrow::StructType&>(*row_fields.field(1)->type());
        std::vector<std::shared_ptr<arrow::Array>> axis_columns;
        std::vector<std::shared_ptr<arrow::Field>> axis_fields;
        for (int a = 0; a < n_axes; ++a) {
            std::shared_ptr<arrow::Array> indices;
            (void)axis_take[static_cast<size_t>(a)].Finish(&indices);
            axis_columns.push_back(
                cast_to(take_indices(axis_values[a], indices), out_axes.field(a)->type()));
            axis_fields.push_back(out_axes.field(a));
        }

        auto axes_struct = arrow::StructArray::Make(axis_columns, axis_fields);
        if (!axes_struct.ok()) tensor_error(axes_struct.status().ToString());
        auto rows = arrow::StructArray::Make({values, *axes_struct},
                                             {row_fields.field(0), row_fields.field(1)});
        if (!rows.ok()) tensor_error(rows.status().ToString());

        std::shared_ptr<arrow::Buffer> offsets_buffer;
        (void)offsets.Finish(&offsets_buffer);
        std::shared_ptr<arrow::Buffer> validity_buffer;
        (void)validity.Finish(&validity_buffer);

        // Assembled from buffers rather than a ListBuilder because the child
        // rows are gathered in one Take: re-appending them through a builder
        // would copy every cell a second time.
        auto data = arrow::ArrayData::Make(
            list_type, batch->num_rows(),
            {null_rows == 0 ? std::shared_ptr<arrow::Buffer>() : validity_buffer,
             offsets_buffer},
            {(*rows)->data()}, null_rows);
        return result(params, std::make_shared<arrow::ListArray>(data));
    }
};

}  // namespace

void register_secret_scalars(vgi::Worker& worker) {
    worker.register_scalar(std::make_shared<GlobalScalar>());
    worker.register_scalar(std::make_shared<WhoAmI>());
    worker.register_scalar(std::make_shared<BinaryPacket>());
    worker.register_scalar(std::make_shared<UnnestTensor>());
}

}  // namespace example
