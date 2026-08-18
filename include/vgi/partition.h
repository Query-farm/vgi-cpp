// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <string>

#include <arrow/record_batch.h>
#include <arrow/type.h>

namespace vgi {

// Field metadata marking a column as a partition key.
inline constexpr const char* kPartitionColumnKey = "vgi.partition_column";
// Per-batch metadata carrying each partition column's (min, max) for the batch.
inline constexpr const char* kPartitionValuesKey = "vgi_partition_values#b64";

// A field declared as a partition column.
std::shared_ptr<arrow::Field> partition_field(const std::string& name,
                                              std::shared_ptr<arrow::DataType> type);

// The `vgi_partition_values#b64` metadata for `batch`, or empty when the schema
// declares no partition columns or the batch has no rows.
//
// The value is base64 of an IPC batch holding two rows — the min and max of
// each partition column. Two rows rather than the distinct values, because the
// engine only needs to know the range a batch covers to decide whether a
// filtered query can skip it.
std::map<std::string, std::string> partition_metadata(
    const std::shared_ptr<arrow::Schema>& full_schema,
    const std::shared_ptr<arrow::RecordBatch>& batch);

}  // namespace vgi
