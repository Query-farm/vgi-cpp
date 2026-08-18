// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// `sum_all_columns(data)` — a column-wise sum across every input batch,
// emitted as a single row.

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/table.h>

#include <vgi/numeric.h>
#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

constexpr const char* kNamespace = "sums";

// One entry per process() call, for the fixture that has to raise on the
// second batch. Separate from the partials so an empty sum log still means
// "nothing was summed".
constexpr const char* kCountNamespace = "count";

std::string encode_batch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(sink, batch->schema()).ValueOrDie();
    (void)writer->WriteRecordBatch(*batch);
    (void)writer->Close();
    return sink->Finish().ValueOrDie()->ToString();
}

std::shared_ptr<arrow::RecordBatch> decode_batch(const std::string& bytes) {
    if (bytes.empty()) return nullptr;
    auto buffer = arrow::Buffer::FromString(bytes);
    auto source = std::make_shared<arrow::io::BufferReader>(buffer);
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!reader.ok()) return nullptr;
    std::shared_ptr<arrow::RecordBatch> batch;
    (void)reader.ValueUnsafe()->ReadNext(&batch);
    return batch;
}

// Integers sum as int64, anything real-valued as double, everything else is
// dropped — a sum of strings has no meaning and silently emitting zero would be
// worse than leaving the column out.
std::shared_ptr<arrow::DataType> sum_type(const arrow::DataType& type) {
    if (vgi::is_integer_type(type)) return arrow::int64();
    if (vgi::is_floating_type(type) || vgi::is_decimal_type(type)) return arrow::float64();
    return nullptr;
}

// Derived from the data on hand rather than from `params.output_schema`.
//
// A process or combine call can land on a pooled worker that never ran the
// bind, so the output schema it is handed may be the raw input schema — whose
// decimal and narrow-integer columns this cannot emit. Deriving keeps the
// function correct whichever worker it runs on.
std::shared_ptr<arrow::Schema> derive_output_schema(const arrow::Schema& input) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int i = 0; i < input.num_fields(); ++i) {
        const auto& field = input.field(i);
        if (auto type = sum_type(*field->type())) {
            fields.push_back(arrow::field(field->name(), type, /*nullable=*/true));
        }
    }
    if (fields.empty()) {
        std::ostringstream summary;
        for (int i = 0; i < input.num_fields(); ++i) {
            if (i) summary << ", ";
            summary << input.field(i)->name() << ": " << input.field(i)->type()->ToString();
        }
        throw std::invalid_argument(
            "sum_all_columns requires at least one numeric (integer, floating-point, or "
            "decimal) input column, got [" +
            summary.str() + "]");
    }
    return arrow::schema(std::move(fields));
}

// One row holding each column's sum over `batch`.
std::shared_ptr<arrow::RecordBatch> sum_batch(const std::shared_ptr<arrow::RecordBatch>& batch,
                                              const std::shared_ptr<arrow::Schema>& schema) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(static_cast<size_t>(schema->num_fields()));
    for (int i = 0; i < schema->num_fields(); ++i) {
        const auto& field = schema->field(i);
        auto source = batch->GetColumnByName(field->name());
        auto casted = source ? cast_to(source, field->type()) : nullptr;

        auto total = casted ? arrow::compute::Sum(casted) : arrow::Result<arrow::Datum>();
        if (!casted || !total.ok()) {
            // No such column, or nothing summable in it: null, which is what
            // SQL's SUM over no rows gives.
            arrow::NullBuilder nulls;
            (void)nulls.AppendNull();
            std::shared_ptr<arrow::Array> array;
            (void)nulls.Finish(&array);
            columns.push_back(cast_to(array, field->type()));
            continue;
        }
        auto scalar = total.MoveValueUnsafe().scalar();
        auto array = arrow::MakeArrayFromScalar(*scalar, 1);
        columns.push_back(cast_to(array.ValueOrDie(), field->type()));
    }
    return arrow::RecordBatch::Make(schema, 1, columns);
}

class Once : public vgi::TableProducer {
public:
    Once(std::shared_ptr<arrow::RecordBatch> batch,
         std::map<std::string, std::string> metadata = {})
        : batch_(std::move(batch)), metadata_(std::move(metadata)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (!batch_) return nullptr;
        auto batch = batch_;
        batch_ = nullptr;
        return batch;
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    std::shared_ptr<arrow::RecordBatch> batch_;
    std::map<std::string, std::string> metadata_;
};

class SumAllColumns : public vgi::TableBufferingFunction {
public:
    enum class Failure { None, Finalize, Process };

    explicit SumAllColumns(std::string name, Failure failure = Failure::None)
        : name_(std::move(name)), failure_(failure) {}

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        if (failure_ != Failure::None) {
            md.description = failure_ == Failure::Finalize
                                 ? "Test function that raises exception during finalize"
                                 : "Test function that raises exception during process";
            md.categories = {"test", "error"};
            return md;
        }
        md.description = name_ == "sum_all_columns_simple_distributed"
                             ? "Distributed sum using the buffered (Sink+Combine+Source) model"
                             : "Computes column-wise sums across all batches";
        md.categories = name_ == "sum_all_columns_simple_distributed"
                            ? std::vector<std::string>{"aggregation", "numeric", "distributed"}
                            : std::vector<std::string>{"aggregation", "numeric"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        // The distributed variant declares only the table input, matching the
        // canonical fixture surface that function_registration.test pins.
        if (name_ == "sum_all_columns_simple_distributed") {
            return {vgi::ArgSpec::table("data", 0, "Input table")};
        }
        return {vgi::ArgSpec::table("data", 0, "Input table"),
                vgi::ArgSpec::named("logging", "boolean", "Emit per-batch INFO logs")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        if (!params.input_schema) {
            throw std::invalid_argument("sum_all_columns requires input schema");
        }
        return derive_output_schema(*params.input_schema);
    }

    std::string process(const vgi::ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>& batch) override {
        if (failure_ == Failure::Process) {
            // Counted through the log rather than a member: several worker
            // processes sink one query in parallel, and "the second batch" is
            // only meaningful across all of them. Nothing is summed on this
            // path, so finalize sees an empty log and emits the zero row.
            params.storage->append(params.execution_id, kCountNamespace, "", "");
            const auto seen =
                params.storage->scan(params.execution_id, kCountNamespace, "", 0, SIZE_MAX).size();
            if (seen % 2 == 0) {
                throw std::invalid_argument("Intentional exception on batch " +
                                            std::to_string(seen));
            }
            return params.execution_id;
        }
        if (params.arguments.named_bool("logging").value_or(false)) {
            params.client_log(
                vgi::LogLevel::Info,
                "Processing batch with " + std::to_string(batch ? batch->num_rows() : 0) + " rows");
        }
        if (batch && batch->num_rows() > 0) {
            auto schema = derive_output_schema(*batch->schema());
            params.storage->append(params.execution_id, kNamespace, "",
                                   encode_batch(sum_batch(batch, schema)));
        }
        return params.execution_id;
    }

    std::vector<std::string> combine(const vgi::ProcessParams& params,
                                     const std::vector<std::string>& state_ids) override {
        // Logged from here too, and deliberately: combine is a unary RPC with
        // no streaming collector, so it is the call that proves the in-band
        // channel is not the producer's alone.
        if (params.arguments.named_bool("logging").value_or(false)) {
            params.client_log(vgi::LogLevel::Info,
                              "Combining " + std::to_string(state_ids.size()) + " state_ids");
        }
        return {params.execution_id};
    }

    std::unique_ptr<vgi::TableProducer> finalize_producer(
        const vgi::ProcessParams& params, const std::string& finalize_state_id) override {
        if (failure_ == Failure::Finalize) {
            throw std::invalid_argument("Intentional exception during finalize()");
        }
        const auto scope = finalize_state_id.empty() ? params.execution_id : finalize_state_id;
        // Every partial is one row, so the whole log fits in memory here even
        // though the input relation did not.
        auto partials = params.storage->scan(scope, kNamespace, "", 0, SIZE_MAX);

        auto schema = params.output_schema;
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        for (const auto& [id, bytes] : partials) {
            (void)id;
            if (auto batch = decode_batch(bytes)) {
                if (!schema) schema = batch->schema();
                batches.push_back(vgi::project_batch(batch, schema));
            }
        }
        if (!schema) return std::make_unique<Once>(nullptr, cache_metadata());

        if (batches.empty()) {
            // No rows sunk at all: one row of zeros. NULL would be the SQL
            // answer for SUM over nothing, but the canonical fixture emits
            // zeros and the tests read the shape as well as the value.
            std::vector<std::shared_ptr<arrow::Array>> columns;
            for (int i = 0; i < schema->num_fields(); ++i) {
                arrow::Int64Builder zero;
                (void)zero.Append(0);
                std::shared_ptr<arrow::Array> array;
                (void)zero.Finish(&array);
                columns.push_back(cast_to(array, schema->field(i)->type()));
            }
            return std::make_unique<Once>(arrow::RecordBatch::Make(schema, 1, columns),
                                          cache_metadata());
        }

        auto table = arrow::Table::FromRecordBatches(schema, batches).ValueOrDie();
        auto combined = table->CombineChunks().ValueOrDie();
        arrow::TableBatchReader reader(*combined);
        std::shared_ptr<arrow::RecordBatch> all;
        (void)reader.ReadNext(&all);
        return std::make_unique<Once>(all ? sum_batch(all, schema) : nullptr, cache_metadata());
    }

private:
    // The cacheable variant advertises a TTL on its finalize output, which is
    // what opts a buffered function into the exchange-mode result cache — the
    // hit then skips the combine RPC and the finalize drain entirely.
    std::map<std::string, std::string> cache_metadata() const {
        if (name_ != "cached_sum_all") return {};
        vgi::CacheControl control;
        control.ttl_seconds = 300;
        return control.to_metadata();
    }

    std::string name_;
    Failure failure_;
};

}  // namespace

void register_sum_all_columns(vgi::Worker& worker) {
    worker.register_buffering(std::make_shared<SumAllColumns>("sum_all_columns"));
    worker.register_buffering(
        std::make_shared<SumAllColumns>("sum_all_columns_simple_distributed"));
    worker.register_buffering(std::make_shared<SumAllColumns>("cached_sum_all"));
    worker.register_buffering(
        std::make_shared<SumAllColumns>("exception_finalize", SumAllColumns::Failure::Finalize));
    worker.register_buffering(
        std::make_shared<SumAllColumns>("exception_process", SumAllColumns::Failure::Process));
}

}  // namespace example
