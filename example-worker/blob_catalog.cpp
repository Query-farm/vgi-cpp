// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The `blob_test` catalog: the blob-vendor fixture contract (vgi
// docs/testing/blob-test-catalog.md), the reference being vgi-python's
// `vgi/_test_fixtures/blob_catalog.py`.
//
// Served here: the stateless half -- `blob_repeat`, `blob_len`, `blob_rows`,
// `blob_param_echo` -- and the four protocol 3.4 argument-reference fixtures
// `blob_param_batches`, `blob_param_streams`, `blob_param_scalar` and the
// `blob_param_table` catalog table. Not served: `blob_buffer_echo`,
// `blob_tio` and `blob_store_values`, which need the vendor-backed function
// storage envelope (`VGI_STORAGE_EXTERNALIZE`) this SDK does not implement.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>

#include <vgi_rpc/crypto.h>

#include <vgi/worker.h>

#include "registry.h"
#include "scalar/util.h"

namespace example {
namespace {

constexpr const char* kCatalog = "blob_test";
constexpr int64_t kRowsPerBatch = 2048;
// `blob_param_table`'s fixed argument: repeat('t', 65536).
const std::string& table_argument() {
    static const std::string value(65536, 't');
    return value;
}

std::string sha256_hex(const std::string& input) {
    const auto digest =
        vgi_rpc::crypto::sha256(reinterpret_cast<const uint8_t*>(input.data()), input.size());
    return vgi_rpc::crypto::hex_encode(digest.data(), digest.size());
}

std::shared_ptr<arrow::Array> finish(arrow::ArrayBuilder& builder) {
    std::shared_ptr<arrow::Array> array;
    if (!builder.Finish(&array).ok()) throw std::runtime_error("blob_test: array build failed");
    return array;
}

std::shared_ptr<arrow::Array> int64s(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    for (auto v : values) (void)builder.Append(v);
    return finish(builder);
}

std::shared_ptr<arrow::Array> strings(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& v : values) (void)builder.Append(v);
    return finish(builder);
}

std::shared_ptr<arrow::Schema> echo_schema() {
    return arrow::schema({arrow::field("payload_len", arrow::int64()),
                          arrow::field("payload_sha256", arrow::utf8())});
}

vgi::FunctionMetadata described(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = {"test"};
    return md;
}

// Emits the batches it was given, one per tick.
class Batches : public vgi::TableProducer {
public:
    explicit Batches(std::vector<std::shared_ptr<arrow::RecordBatch>> batches)
        : batches_(std::move(batches)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (next_ >= batches_.size()) return nullptr;
        return batches_[next_++];
    }

private:
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
    size_t next_ = 0;
};

// The one row `(strlen(payload), sha256(payload))`.
std::shared_ptr<arrow::RecordBatch> echo_row(const std::shared_ptr<arrow::Schema>& schema,
                                             const std::string& payload) {
    return arrow::RecordBatch::Make(
        schema, 1,
        {int64s({static_cast<int64_t>(payload.size())}), strings({sha256_hex(payload)})});
}

// --- scalars ---------------------------------------------------------------

// `blob_repeat(s, n)`: `s` repeated `n` times (large scalar outputs).
class BlobRepeat : public vgi::ScalarFunction {
public:
    std::string name() const override { return "blob_repeat"; }
    vgi::FunctionMetadata metadata() const override {
        auto md = described("Repeat a string n times (externalization test: large scalar output)");
        md.return_type = arrow::utf8();
        return md;
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("s", 0, "varchar", "String to repeat"),
                vgi::ArgSpec::column("n", 1, "int64", "Number of copies")};
    }
    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto s =
            std::static_pointer_cast<arrow::StringArray>(cast_to(batch->column(0), arrow::utf8()));
        const auto n =
            std::static_pointer_cast<arrow::Int64Array>(cast_to(batch->column(1), arrow::int64()));
        arrow::StringBuilder out;
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            if (s->IsNull(i) || n->IsNull(i)) {
                (void)out.AppendNull();
                continue;
            }
            const auto piece = s->GetView(i);
            std::string value;
            const int64_t copies = std::max<int64_t>(n->Value(i), 0);
            value.reserve(piece.size() * static_cast<size_t>(copies));
            for (int64_t c = 0; c < copies; ++c) value.append(piece);
            (void)out.Append(value);
        }
        return result(params, finish(out));
    }
};

// `blob_len(s)`: strlen(s) in UTF-8 bytes (large scalar inputs).
class BlobLen : public vgi::ScalarFunction {
public:
    std::string name() const override { return "blob_len"; }
    vgi::FunctionMetadata metadata() const override {
        auto md = described("UTF-8 byte length of a string (externalization test: large input)");
        md.return_type = arrow::int64();
        return md;
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("s", 0, "varchar", "String to measure")};
    }
    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto s =
            std::static_pointer_cast<arrow::StringArray>(cast_to(batch->column(0), arrow::utf8()));
        arrow::Int64Builder out;
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            if (s->IsNull(i)) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(static_cast<int64_t>(s->GetView(i).size()));
            }
        }
        return result(params, finish(out));
    }
};

// `blob_param_scalar(i, payload)`: sha256(payload) for every non-NULL `i`.
// `payload` is a constant parameter, so it rides the bind (and with it an
// argument reference), not the input batches.
class BlobParamScalar : public vgi::ScalarFunction {
public:
    std::string name() const override { return "blob_param_scalar"; }
    vgi::FunctionMetadata metadata() const override {
        auto md = described(
            "SHA-256 of a constant parameter for every row (argument references for scalars)");
        md.return_type = arrow::utf8();
        return md;
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("i", 0, "int64", "Row value; NULL gives NULL"),
                vgi::ArgSpec::constant_arg("payload", 1, "varchar", "Constant parameter to hash")};
    }
    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto payload = params.arguments.const_string(1);
        const std::optional<std::string> digest =
            payload ? std::optional<std::string>(sha256_hex(*payload)) : std::nullopt;
        const auto column = batch->column(0);
        arrow::StringBuilder out;
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            if (column->IsNull(i) || !digest) {
                (void)out.AppendNull();
            } else {
                (void)out.Append(*digest);
            }
        }
        return result(params, finish(out));
    }
};

// --- table functions -------------------------------------------------------

// `blob_rows(rows, width)`: row i is repeat(chr(97 + i % 26), width).
class BlobRows : public vgi::TableFunction {
public:
    std::string name() const override { return "blob_rows"; }
    vgi::FunctionMetadata metadata() const override {
        return described("Rows of a given width (externalization test: large table output)");
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto rows = vgi::ArgSpec::constant_arg("rows", 0, "int64", "Number of rows");
        rows.with_range(0, std::nullopt);
        auto width = vgi::ArgSpec::constant_arg("width", 1, "int64", "Characters per row");
        width.with_range(0, std::nullopt);
        return {rows, width};
    }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema(
            {arrow::field("i", arrow::int64()), arrow::field("payload", arrow::utf8())});
    }
    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        const auto rows = params.arguments.const_int64(0).value_or(0);
        return {rows, rows};
    }
    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const int64_t rows = std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        const int64_t width = std::max<int64_t>(0, params.arguments.const_int64(1).value_or(0));
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        for (int64_t start = 0; start < rows; start += kRowsPerBatch) {
            const int64_t stop = std::min(start + kRowsPerBatch, rows);
            std::vector<int64_t> ids;
            std::vector<std::string> payloads;
            for (int64_t i = start; i < stop; ++i) {
                ids.push_back(i);
                payloads.emplace_back(static_cast<size_t>(width), static_cast<char>('a' + i % 26));
            }
            batches.push_back(arrow::RecordBatch::Make(params.output_schema, stop - start,
                                                       {int64s(ids), strings(payloads)}));
        }
        return std::make_unique<Batches>(std::move(batches));
    }
};

// `blob_param_echo(payload)`, and `blob_param_table_scan` -- the same echo of
// a fixed argument, reporting STABLE_WITHIN_QUERY -- behind `blob_param_table`.
class BlobParamEcho : public vgi::TableFunction {
public:
    explicit BlobParamEcho(bool table_scan) : table_scan_(table_scan) {}

    std::string name() const override {
        return table_scan_ ? "blob_param_table_scan" : "blob_param_echo";
    }
    vgi::FunctionMetadata metadata() const override {
        return described(table_scan_ ? "Length and SHA-256 of blob_param_table's fixed parameter "
                                       "(STABLE_WITHIN_QUERY)"
                                     : "Length and SHA-256 of a parameter (externalization "
                                       "test: large bind parameter)");
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("payload", 0, "varchar", "Parameter to measure")};
    }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return echo_schema();
    }
    vgi::BindRepeatability repeatability(const vgi::BindParams&) const override {
        return table_scan_ ? vgi::BindRepeatability::StableWithinQuery
                           : vgi::BindRepeatability::Unknown;
    }
    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override { return {1, 1}; }
    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto payload = params.arguments.const_string(0).value_or("");
        return std::make_unique<Batches>(std::vector<std::shared_ptr<arrow::RecordBatch>>{
            echo_row(params.output_schema, payload)});
    }

private:
    bool table_scan_;
};

// `blob_param_batches(payload, batches)`: `batches` one-row batches, one per
// stream turn, row i = (i, strlen(payload), sha256(payload)).
class BlobParamBatches : public vgi::TableFunction {
public:
    std::string name() const override { return "blob_param_batches"; }
    vgi::FunctionMetadata metadata() const override {
        return described(
            "Length and SHA-256 of a parameter, once per batch (argument references over many "
            "turns)");
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto batches =
            vgi::ArgSpec::constant_arg("batches", 1, "int64", "Number of one-row batches");
        batches.with_range(0, std::nullopt);
        return {vgi::ArgSpec::constant_arg("payload", 0, "varchar", "Parameter to measure"),
                batches};
    }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("i", arrow::int64()),
                              arrow::field("payload_len", arrow::int64()),
                              arrow::field("payload_sha256", arrow::utf8())});
    }
    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        const auto n = params.arguments.const_int64(1).value_or(0);
        return {n, n};
    }
    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto payload = params.arguments.const_string(0).value_or("");
        const int64_t count = std::max<int64_t>(0, params.arguments.const_int64(1).value_or(0));
        const auto length = static_cast<int64_t>(payload.size());
        const auto digest = sha256_hex(payload);
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        for (int64_t i = 0; i < count; ++i) {
            batches.push_back(arrow::RecordBatch::Make(
                params.output_schema, 1, {int64s({i}), int64s({length}), strings({digest})}));
        }
        return std::make_unique<Batches>(std::move(batches));
    }
};

std::string encode_index(int64_t index) {
    std::string out(8, '\0');
    for (int i = 0; i < 8; ++i)
        out[static_cast<size_t>(i)] =
            static_cast<char>((static_cast<uint64_t>(index) >> (8 * i)) & 0xFF);
    return out;
}

int64_t decode_index(const std::string& payload) {
    if (payload.size() != 8) {
        throw std::runtime_error("blob_param_streams was handed a split payload it did not mint");
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(static_cast<unsigned char>(payload[static_cast<size_t>(i)]))
             << (8 * i);
    return static_cast<int64_t>(v);
}

// `blob_param_streams(payload, streams)`: one bind planned as `streams`
// splits, each emitting (stream index, strlen(payload), sha256(payload)).
// Every split's init embeds the same bind call, so with an argument reference
// a worker process fetches the argument once (the vgi-rpc object cache).
class BlobParamStreams : public vgi::TableFunction {
public:
    std::string name() const override { return "blob_param_streams"; }
    vgi::FunctionMetadata metadata() const override {
        return described(
            "Length and SHA-256 of a parameter from parallel streams (argument references)");
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto streams = vgi::ArgSpec::constant_arg("streams", 1, "int64",
                                                  "Number of parallel streams (one split each)");
        streams.with_range(1, std::nullopt);
        return {vgi::ArgSpec::constant_arg("payload", 0, "varchar", "Parameter to measure"),
                streams};
    }
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("stream", arrow::int64()),
                              arrow::field("payload_len", arrow::int64()),
                              arrow::field("payload_sha256", arrow::utf8())});
    }
    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        const auto n = params.arguments.const_int64(1).value_or(1);
        return {n, n};
    }
    bool supports_splits() const override { return true; }
    vgi::PlanResult plan(const vgi::BindParams& params, const vgi::PlanParams&) const override {
        const int64_t count = std::max<int64_t>(1, params.arguments.const_int64(1).value_or(1));
        vgi::PlanResult result;
        result.estimated_total_splits = count;
        result.estimated_total_rows = count;
        for (int64_t i = 0; i < count; ++i) {
            vgi::ScanSplit split;
            split.payload = encode_index(i);
            split.estimated_rows = 1;
            split.rows_exact = true;
            result.splits.push_back(std::move(split));
        }
        return result;
    }
    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        if (!params.split_payloads) {
            throw std::runtime_error(
                "blob_param_streams is split-only; vgi_split_scans is probably off");
        }
        const auto payload = params.arguments.const_string(0).value_or("");
        const auto length = static_cast<int64_t>(payload.size());
        const auto digest = sha256_hex(payload);
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        for (const auto& split : *params.split_payloads) {
            batches.push_back(arrow::RecordBatch::Make(
                params.output_schema, 1,
                {int64s({decode_index(split)}), int64s({length}), strings({digest})}));
        }
        return std::make_unique<Batches>(std::move(batches));
    }
};

std::shared_ptr<arrow::Array> string_value(const std::string& value) {
    return strings({value});
}

}  // namespace

void register_blob_catalog(vgi::Worker& worker) {
    auto& model = worker.catalog(kCatalog);
    model.comment = "Blob-vendor externalization fixtures (vgi docs/testing/blob-test-catalog.md)";
    worker.register_scalar_in(kCatalog, "main", std::make_shared<BlobRepeat>());
    worker.register_scalar_in(kCatalog, "main", std::make_shared<BlobLen>());
    worker.register_scalar_in(kCatalog, "main", std::make_shared<BlobParamScalar>());
    worker.register_table_in(kCatalog, "main", std::make_shared<BlobRows>());
    worker.register_table_in(kCatalog, "main", std::make_shared<BlobParamEcho>(false));
    worker.register_table_in(kCatalog, "main", std::make_shared<BlobParamEcho>(true));
    worker.register_table_in(kCatalog, "main", std::make_shared<BlobParamBatches>());
    worker.register_table_in(kCatalog, "main", std::make_shared<BlobParamStreams>());

    // A table whose scan binds the echo with a fixed 64 KiB argument and
    // reports STABLE_WITHIN_QUERY. Its bind is never inlined, so every scan
    // binds over RPC and a serialized-plan rebind has to reproduce the bind
    // bytes (the extension's per-query argument-reference memo).
    vgi::CatalogTable table;
    table.name = "blob_param_table";
    table.scan_function = "blob_param_table_scan";
    table.scan_arguments = vgi::serialize_scan_arguments({string_value(table_argument())});
    table.columns = echo_schema();
    table.cardinality = 1;
    model.schema("main").tables.push_back(std::move(table));
}

}  // namespace example
