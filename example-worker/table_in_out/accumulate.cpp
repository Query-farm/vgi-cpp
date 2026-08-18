// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The `accumulate` family: a name-keyed row accumulator whose collections
// outlive the query that filled them.
//
// What the fixture probes is persistence. Under the subprocess transport every
// query spawns a fresh worker, and the engine spreads one buffering call across
// several of them, so a collection that is still there for the next statement
// proves the rows went through FunctionStorage rather than through a member.

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/table.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// The catalog these three serve. Named rather than spelled at each use: the
// registrations and the model must agree or the functions are invisible.
constexpr const char* kCatalog = "accumulate";

// Appended to every output row, carrying the call time. Underscore-prefixed so
// it is unlikely to collide with a column a caller already has.
constexpr const char* kTimestampColumn = "_timestamp";

// The name becomes part of a storage scope, so its length is bounded.
constexpr size_t kMaxNameBytes = 255;

// Rows per emitted batch. The collection is bounded only by what callers have
// appended, so it goes back in slices rather than as one batch.
constexpr int64_t kOutBatchRows = 65536;

// Execution-scoped log namespaces. The sink stages its input under the first
// and combine stages the chosen result under the second; the phases run in
// different worker processes, so the log is the only channel between them.
constexpr const char* kInputNamespace = "acc_in";
constexpr const char* kOutputNamespace = "acc_out";

// Keys within one collection's persistent scope.
constexpr const char* kSchemaKey = "schema";
constexpr const char* kRowsKey = "rows";

// Tz-naive microseconds, which is what DuckDB surfaces as TIMESTAMP rather than
// TIMESTAMP WITH TIME ZONE.
std::shared_ptr<arrow::DataType> timestamp_type() {
    return arrow::timestamp(arrow::TimeUnit::MICRO);
}

std::shared_ptr<arrow::Schema> clear_schema() {
    return arrow::schema({arrow::field("name", arrow::utf8(), /*nullable=*/false),
                          arrow::field("rows_cleared", arrow::int64(), /*nullable=*/false)});
}

// A collection's storage scope.
//
// Deliberately not the execution id: the engine discards that when the query
// ends, and a collection has to still be there for the next one. Keyed by the
// attachment as well as the name, because two sessions may use the same
// collection name and must not see each other's rows.
std::string collection_scope(const std::string& attachment, const std::string& name) {
    return "vgi-accumulate:" + attachment + ":" + name;
}

int64_t now_micros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
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
    if (!reader.ok()) return nullptr;
    std::shared_ptr<arrow::RecordBatch> batch;
    (void)reader.ValueUnsafe()->ReadNext(&batch);
    return batch;
}

std::string encode_table(const std::shared_ptr<arrow::Table>& table) {
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(sink, table->schema()).ValueOrDie();
    (void)writer->WriteTable(*table);
    (void)writer->Close();
    return sink->Finish().ValueOrDie()->ToString();
}

std::shared_ptr<arrow::Table> decode_table(const std::string& bytes) {
    if (bytes.empty()) return nullptr;
    auto source = std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(bytes));
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!reader.ok()) return nullptr;
    auto table = reader.ValueUnsafe()->ToTable();
    return table.ok() ? table.MoveValueUnsafe() : nullptr;
}

std::shared_ptr<arrow::Schema> output_schema_of(const std::shared_ptr<arrow::Schema>& input) {
    auto fields = input->fields();
    fields.push_back(arrow::field(kTimestampColumn, timestamp_type(), /*nullable=*/false));
    return arrow::schema(std::move(fields));
}

std::shared_ptr<arrow::Schema> input_schema_of(const std::shared_ptr<arrow::Schema>& output) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& field : output->fields()) {
        if (field->name() != kTimestampColumn) fields.push_back(field);
    }
    return arrow::schema(std::move(fields));
}

// Names and types only. Nullability is deliberately left out: DuckDB derives it
// from the call site's expression, so two calls that ship the same column can
// still disagree on it, and rejecting that would make the pin unusable.
bool input_fields_match(const std::shared_ptr<arrow::Schema>& pinned,
                        const std::shared_ptr<arrow::Schema>& incoming) {
    if (pinned->num_fields() != incoming->num_fields()) return false;
    for (int i = 0; i < pinned->num_fields(); ++i) {
        if (pinned->field(i)->name() != incoming->field(i)->name()) return false;
        if (!pinned->field(i)->type()->Equals(*incoming->field(i)->type())) return false;
    }
    return true;
}

void validate_name(const std::string& name) {
    if (name.find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
        throw std::invalid_argument("collection name must be a non-empty string");
    }
    if (name.size() > kMaxNameBytes) {
        throw std::invalid_argument("collection name must be at most " +
                                    std::to_string(kMaxNameBytes) + " bytes");
    }
}

// The schema pinned by the first call to accumulate under `name`, or null when
// nothing has been accumulated under it.
std::shared_ptr<arrow::Schema> pinned_schema(vgi::FunctionStorage& storage,
                                             const std::string& attachment,
                                             const std::string& name) {
    auto blob = storage.kv_get(collection_scope(attachment, name), kSchemaKey);
    if (!blob) return nullptr;
    auto table = decode_table(*blob);
    return table ? table->schema() : nullptr;
}

void pin_schema(vgi::FunctionStorage& storage, const std::string& attachment,
                const std::string& name, const std::shared_ptr<arrow::Schema>& schema) {
    storage.kv_put(collection_scope(attachment, name), kSchemaKey,
                   encode_table(arrow::Table::MakeEmpty(schema).ValueOrDie()));
}

std::shared_ptr<arrow::Table> read_collection(vgi::FunctionStorage& storage,
                                              const std::string& attachment,
                                              const std::string& name,
                                              const std::shared_ptr<arrow::Schema>& fallback) {
    if (auto blob = storage.kv_get(collection_scope(attachment, name), kRowsKey)) {
        if (auto table = decode_table(*blob)) return table;
    }
    return arrow::Table::MakeEmpty(fallback).ValueOrDie();
}

// The whole collection is rewritten on every append. The reference workers keep
// time-keyed segments and evict with a ranged delete, but FunctionStorage
// offers neither ranged deletes nor key enumeration, and a hand-rolled segment
// index buys nothing for a fixture whose largest collection is ten thousand
// rows.
void write_collection(vgi::FunctionStorage& storage, const std::string& attachment,
                      const std::string& name, const std::shared_ptr<arrow::Table>& table) {
    storage.kv_put(collection_scope(attachment, name), kRowsKey, encode_table(table));
}

// A DuckDB INTERVAL as microseconds, or nullopt when the argument was omitted.
// Calendar months have no fixed length, so a month counts as 30 days — the same
// approximation the Python and Rust fixtures make.
std::optional<int64_t> ttl_micros(const vgi::Arguments& arguments) {
    auto array = arguments.named("ttl");
    if (!array || array->type_id() != arrow::Type::INTERVAL_MONTH_DAY_NANO) return std::nullopt;
    if (array->length() == 0 || array->IsNull(0)) return std::nullopt;
    const auto interval =
        std::static_pointer_cast<arrow::MonthDayNanoIntervalArray>(array)->Value(0);
    return (static_cast<int64_t>(interval.months) * 30 + interval.days) * 86'400'000'000LL +
           interval.nanoseconds / 1000;
}

// Index of the first row stamped at or after `cutoff`. Every row of one call
// carries that call's time and calls append in order, so the column is
// nondecreasing and a walk from the front finds the boundary.
int64_t first_row_at_or_after(const std::shared_ptr<arrow::Table>& table, int64_t cutoff) {
    auto column = table->GetColumnByName(kTimestampColumn);
    if (!column) return 0;
    int64_t index = 0;
    for (const auto& chunk : column->chunks()) {
        auto stamps =
            std::static_pointer_cast<arrow::TimestampArray>(cast_to(chunk, timestamp_type()));
        for (int64_t row = 0; row < stamps->length(); ++row, ++index) {
            if (!stamps->IsNull(row) && stamps->Value(row) >= cutoff) return index;
        }
    }
    return index;
}

// Append the call's `_timestamp` column, normalizing the input on the way.
//
// The input is projected onto the pinned shape rather than taken as it arrives:
// DuckDB pushes projection down *into* structs, so a column declared
// `struct<a,b>` can turn up as `struct<b>`, and a batch whose schema disagrees
// with its own arrays does not fail here — it corrupts whatever reads it next.
std::shared_ptr<arrow::RecordBatch> stamp(const std::shared_ptr<arrow::RecordBatch>& batch,
                                          const std::shared_ptr<arrow::Schema>& output_schema,
                                          int64_t micros) {
    auto projected = vgi::project_batch(batch, input_schema_of(output_schema));
    const std::vector<int64_t> values(static_cast<size_t>(projected->num_rows()), micros);
    arrow::TimestampBuilder builder(timestamp_type(), arrow::default_memory_pool());
    (void)builder.AppendValues(values);
    std::shared_ptr<arrow::Array> stamps;
    (void)builder.Finish(&stamps);

    auto columns = projected->columns();
    columns.push_back(std::move(stamps));
    return arrow::RecordBatch::Make(output_schema, projected->num_rows(), std::move(columns));
}

// Emits a table in bounded slices, projected onto the schema the call was bound
// to.
class TableDrain : public vgi::TableProducer {
public:
    TableDrain(std::shared_ptr<arrow::Table> table, std::shared_ptr<arrow::Schema> output_schema)
        : reader_(std::move(table)), output_schema_(std::move(output_schema)) {
        reader_.set_chunksize(kOutBatchRows);
    }

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        std::shared_ptr<arrow::RecordBatch> batch;
        if (!reader_.ReadNext(&batch).ok() || !batch) return nullptr;
        return vgi::project_batch(batch, output_schema_);
    }

private:
    arrow::TableBatchReader reader_;
    std::shared_ptr<arrow::Schema> output_schema_;
};

// Replays an execution's staged result, one batch per tick.
class LogDrain : public vgi::TableProducer {
public:
    LogDrain(std::shared_ptr<vgi::FunctionStorage> storage, std::string scope,
             std::shared_ptr<arrow::Schema> output_schema)
        : storage_(std::move(storage)),
          scope_(std::move(scope)),
          output_schema_(std::move(output_schema)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        // One entry per tick rather than draining up front: the collection a
        // caller asked for may be far larger than the rows they just added.
        auto entries = storage_->scan(scope_, kOutputNamespace, "", after_id_, 1);
        if (entries.empty()) return nullptr;
        after_id_ = entries.front().first;
        auto batch = decode_batch(entries.front().second);
        return batch ? vgi::project_batch(batch, output_schema_) : nullptr;
    }

private:
    std::shared_ptr<vgi::FunctionStorage> storage_;
    std::string scope_;
    std::shared_ptr<arrow::Schema> output_schema_;
    int64_t after_id_ = 0;
};

// Stage a result for the source phase, in batches it can stream back.
void stage(const vgi::ProcessParams& params, const std::shared_ptr<arrow::Table>& table) {
    arrow::TableBatchReader reader(table);
    reader.set_chunksize(kOutBatchRows);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (reader.ReadNext(&batch).ok() && batch) {
        params.storage->append(params.execution_id, kOutputNamespace, "", encode_batch(batch));
    }
}

// `accumulate(name, data, ttl :=, max_row_size :=, result :=)`
//
// A buffering function rather than a table-in-out one: every row of a call has
// to carry the same `_timestamp`, so nothing can be emitted until the whole
// input has arrived.
class Accumulate : public vgi::TableBufferingFunction {
public:
    std::string name() const override { return "accumulate"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Append rows to a named collection; return all/new/no rows with a _timestamp column";
        md.categories = {"stateful", "utility"};
        md.tags = {{"category", "stateful"}, {"type", "accumulator"}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto ttl = vgi::ArgSpec::named("ttl", "",
                                       "Evict rows older than this INTERVAL before returning "
                                       "(months are treated as 30 days)");
        // Given as a concrete Arrow type: the VGI type names have no interval,
        // and an untyped named argument resolves as `any`.
        ttl.arrow_type = arrow::month_day_nano_interval();
        return {vgi::ArgSpec::constant_arg("name", 0, "varchar",
                                           "Name of the collection to accumulate into"),
                vgi::ArgSpec::table("data", 1, "Rows to accumulate (any table expression)"),
                std::move(ttl),
                vgi::ArgSpec::named("max_row_size", "int64",
                                    "Maximum rows retained per name; oldest dropped first "
                                    "(0 = unlimited)"),
                vgi::ArgSpec::named("result", "varchar",
                                    "What to return: 'all' (default), 'new', or 'none'")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        const auto name = params.arguments.const_string(0).value_or("");
        validate_name(name);
        if (!params.input_schema) {
            throw std::invalid_argument("accumulate requires a table input");
        }
        if (params.input_schema->GetFieldIndex(kTimestampColumn) >= 0) {
            // Refused rather than overwritten: a caller shipping a column of
            // this name means something by it, and quietly replacing their
            // values with call times is worse than telling them.
            throw std::invalid_argument(
                "input may not contain a reserved '_timestamp' column; accumulate adds this "
                "column to its output");
        }
        auto output_schema = output_schema_of(params.input_schema);

        // Bind is the one phase that sees no ProcessParams, so it reaches the
        // process-wide store directly; it is the same object the later phases
        // are handed.
        auto storage = vgi::default_storage();
        auto pinned = pinned_schema(*storage, params.attachment_id, name);
        if (!pinned) {
            // Pinned on first use, then enforced. Two simultaneous first
            // appends of incompatible schemas race, and the loser gets a
            // confusing validation error — but never a mixed collection, since
            // every append is normalized to whichever schema won.
            pin_schema(*storage, params.attachment_id, name, output_schema);
        } else if (!input_fields_match(input_schema_of(pinned), params.input_schema)) {
            throw std::invalid_argument(
                "input schema for accumulate('" + name + "', ...) does not match the schema " +
                "already accumulated under that name.\n  accumulated: " +
                input_schema_of(pinned)->ToString() +
                "\n  received:    " + params.input_schema->ToString());
        }
        return output_schema;
    }

    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>& batch) override {
        if (batch && batch->num_rows() > 0) {
            params.storage->append(params.execution_id, kInputNamespace, "", encode_batch(batch));
        }
        // Every sink writes into the one execution-scoped log, so the state id
        // is the execution id and combine has nothing to reconcile.
        return params.execution_id;
    }

    std::vector<std::string> combine(const vgi::ProcessParams& params,
                                     const std::vector<std::string>& state_ids) override {
        (void)state_ids;
        const auto name = params.arguments.const_string(0).value_or("");

        // The pinned schema, not the bound one: all of a collection's rows must
        // share one exact schema for the append to concatenate, and only the
        // pin is invariant across the call sites that write to the name.
        auto schema = pinned_schema(*params.storage, params.attachment_id, name);
        if (!schema) schema = params.output_schema;

        const int64_t micros = now_micros();
        std::vector<std::shared_ptr<arrow::RecordBatch>> added;
        for (const auto& entry : params.storage->scan(params.execution_id, kInputNamespace, "",
                                                      /*after_id=*/0,
                                                      std::numeric_limits<size_t>::max())) {
            if (auto batch = decode_batch(entry.second)) {
                added.push_back(stamp(batch, schema, micros));
            }
        }
        auto new_rows = added.empty()
                            ? arrow::Table::MakeEmpty(schema).ValueOrDie()
                            : arrow::Table::FromRecordBatches(schema, added).ValueOrDie();

        auto collection = read_collection(*params.storage, params.attachment_id, name, schema);
        if (new_rows->num_rows() > 0) {
            collection = arrow::ConcatenateTables({collection, new_rows}).ValueOrDie();
        }
        // The TTL is measured from this call's time, so a zero interval keeps
        // exactly the rows this call added. The row cap then trims whatever
        // survived it.
        if (const auto ttl = ttl_micros(params.arguments)) {
            const int64_t cutoff = micros - *ttl;
            if (cutoff > 0) {
                collection = collection->Slice(first_row_at_or_after(collection, cutoff));
            }
        }
        const int64_t max_rows = params.arguments.named_int64("max_row_size").value_or(0);
        if (max_rows > 0 && collection->num_rows() > max_rows) {
            collection = collection->Slice(collection->num_rows() - max_rows);
        }
        write_collection(*params.storage, params.attachment_id, name, collection);

        const auto mode = params.arguments.named_string("result").value_or("all");
        if (mode == "new") {
            // The rows this call added, before eviction had a say in them.
            stage(params, new_rows);
        } else if (mode != "none") {
            stage(params, collection);
        }
        return {params.execution_id};
    }

    std::unique_ptr<vgi::TableProducer> finalize_producer(
        const vgi::ProcessParams& params, const std::string& finalize_state_id) override {
        const auto scope = finalize_state_id.empty() ? params.execution_id : finalize_state_id;
        return std::make_unique<LogDrain>(params.storage, scope, params.output_schema);
    }
};

// `accumulate_read(name)` — the collection's rows, without touching it.
class AccumulateRead : public vgi::TableFunction {
public:
    std::string name() const override { return "accumulate_read"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Read an accumulated collection's rows without modifying it";
        md.categories = {"stateful", "utility"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("name", 0, "varchar",
                                           "Name of the collection to read")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        const auto name = params.arguments.const_string(0).value_or("");
        validate_name(name);
        auto pinned = pinned_schema(*vgi::default_storage(), params.attachment_id, name);
        if (!pinned) {
            throw std::invalid_argument("no accumulation named '" + name + "' in this session");
        }
        return pinned;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto name = params.arguments.const_string(0).value_or("");
        return std::make_unique<TableDrain>(
            read_collection(*params.storage, params.attachment_id, name, params.output_schema), params.output_schema);
    }
};

// `accumulate_clear(name)` — drop a collection, reporting what it held.
class AccumulateClear : public vgi::TableFunction {
public:
    std::string name() const override { return "accumulate_clear"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Remove an accumulated collection by name; returns rows cleared";
        md.categories = {"stateful", "utility"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("name", 0, "varchar",
                                           "Name of the collection to clear")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        validate_name(params.arguments.const_string(0).value_or(""));
        return clear_schema();
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto name = params.arguments.const_string(0).value_or("");
        int64_t rows_cleared = 0;
        if (auto blob = params.storage->kv_get(collection_scope(params.attachment_id, name), kRowsKey)) {
            if (auto table = decode_table(*blob)) rows_cleared = table->num_rows();
        }
        // The rows and the pinned schema share one scope, so both go in a
        // single call — which is what leaves the name free to be accumulated
        // again under a different schema.
        params.storage->clear(collection_scope(params.attachment_id, name));

        arrow::StringBuilder names;
        (void)names.Append(name);
        arrow::Int64Builder counts;
        (void)counts.Append(rows_cleared);
        std::shared_ptr<arrow::Array> name_array;
        std::shared_ptr<arrow::Array> count_array;
        (void)names.Finish(&name_array);
        (void)counts.Finish(&count_array);

        return std::make_unique<TableDrain>(
            arrow::Table::Make(clear_schema(), {name_array, count_array}), params.output_schema);
    }
};

}  // namespace

void register_accumulate(vgi::Worker& worker) {
    // Its own catalog, not `example`: what these probe is per-attachment
    // scoping, and a catalog of their own is what a session attaches twice.
    auto& model = worker.catalog(kCatalog);
    model.data_version_spec = "2.0.0";
    model.implementation_version = "vgi-fixture";
    model.comment =
        "Row accumulation keyed by name, persisted via FunctionStorage and scoped per ATTACH";
    model.schema("main");

    worker.register_buffering_in(kCatalog, "main", std::make_shared<Accumulate>());
    worker.register_table_in(kCatalog, "main", std::make_shared<AccumulateRead>());
    worker.register_table_in(kCatalog, "main", std::make_shared<AccumulateClear>());
}

}  // namespace example
