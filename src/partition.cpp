// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/partition.h"

#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_base.h>
#include <arrow/compute/api.h>
#include <arrow/util/base64.h>
#include <arrow/util/key_value_metadata.h>

#include "wire.h"

namespace vgi {
namespace {

bool is_partition_field(const arrow::Field& field) {
    const auto& metadata = field.metadata();
    if (!metadata) return false;
    const auto index = metadata->FindKey(kPartitionColumnKey);
    return index >= 0 && metadata->value(index) == "true";
}

}  // namespace

std::shared_ptr<arrow::Field> partition_field(const std::string& name,
                                              std::shared_ptr<arrow::DataType> type) {
    return arrow::field(name, std::move(type), /*nullable=*/true)
        ->WithMetadata(arrow::key_value_metadata({kPartitionColumnKey}, {"true"}));
}

std::map<std::string, std::string> partition_metadata(
    const std::shared_ptr<arrow::Schema>& full_schema,
    const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (!full_schema || !batch || batch->num_rows() == 0) return {};

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> bounds;
    for (int i = 0; i < full_schema->num_fields(); ++i) {
        const auto& field = full_schema->field(i);
        if (!is_partition_field(*field)) continue;
        auto column = batch->GetColumnByName(field->name());
        if (!column || column->length() == 0) continue;

        auto extremes = arrow::compute::MinMax(column);
        if (!extremes.ok()) continue;
        // MinMax gives a struct scalar {min, max}; the wire wants them as two
        // rows of one column, so they are unpacked rather than passed through.
        const auto& pair =
            static_cast<const arrow::StructScalar&>(*extremes.MoveValueUnsafe().scalar());
        if (pair.value.size() < 2) continue;

        // Built as one two-element array rather than concatenating two
        // one-element ones: `take` with indices [0, 0] over a min/max pair is
        // awkward, and building directly avoids depending on which Arrow
        // header Concatenate happens to live in.
        std::unique_ptr<arrow::ArrayBuilder> builder;
        if (!arrow::MakeBuilder(arrow::default_memory_pool(), column->type(), &builder).ok()) {
            continue;
        }
        if (!builder->AppendScalar(*pair.value[0]).ok()) continue;
        if (!builder->AppendScalar(*pair.value[1]).ok()) continue;
        auto both = builder->Finish();
        if (!both.ok()) continue;

        fields.push_back(arrow::field(field->name(), column->type(), /*nullable=*/true));
        bounds.push_back(both.MoveValueUnsafe());
    }
    if (fields.empty()) return {};

    auto encoded = wire::encode_ipc(arrow::RecordBatch::Make(arrow::schema(fields), 2, bounds));
    return {{kPartitionValuesKey, arrow::util::base64_encode(encoded)}};
}

}  // namespace vgi
