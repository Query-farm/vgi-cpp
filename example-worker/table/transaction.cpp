// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The transaction-scoped storage fixture.
//
// Everything else that remembers something keys on the execution id, which
// lives for one scan. This one keys on the transaction instead, so its answer
// survives across statements inside BEGIN/COMMIT and is gone the moment the
// transaction ends.

#include <memory>
#include <string>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

std::shared_ptr<arrow::Schema> value_schema() {
    static const auto schema =
        arrow::schema({arrow::field("v", arrow::int64(), /*nullable=*/false)});
    return schema;
}

// The value this call should report: whatever the transaction already holds
// for `key`, or `seed` if it holds nothing yet.
//
// Resolved identically in bind and in init rather than shipped between them:
// the pool hands a different worker process to each RPC, so the two only agree
// because they both read the same cross-process store.
int64_t resolve(const vgi::Arguments& arguments, const std::optional<std::string>& transaction,
                vgi::FunctionStorage& storage) {
    const int64_t seed = arguments.const_int64(1).value_or(0);
    // Autocommit: the engine opens no catalog transaction, so there is nowhere
    // to remember anything and every call is its own seed.
    if (!transaction || transaction->empty()) return seed;

    const std::string scope = vgi::transaction_scope(*transaction);
    const std::string key = "txcache:" + arguments.const_string(0).value_or("");
    if (auto stored = storage.kv_get(scope, key)) {
        return std::stoll(*stored);
    }
    storage.kv_put(scope, key, std::to_string(seed));
    return seed;
}

class OneValue : public vgi::TableProducer {
public:
    explicit OneValue(int64_t value) : value_(value) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (emitted_) return nullptr;
        emitted_ = true;
        arrow::Int64Builder builder;
        (void)builder.Append(value_);
        std::shared_ptr<arrow::Array> array;
        (void)builder.Finish(&array);
        return arrow::RecordBatch::Make(value_schema(), 1, {array});
    }

private:
    int64_t value_;
    bool emitted_ = false;
};

// `tx_cached_value(key, seed)` — one row, cached per (transaction, key).
class TxCachedValue : public vgi::TableFunction {
public:
    std::string name() const override { return "tx_cached_value"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Return a value cached per (transaction_opaque_data, key)";
        md.categories = {"test", "transaction-storage"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {
            vgi::ArgSpec::constant_arg("key", 0, "varchar", "Cache key, scoped to the transaction"),
            vgi::ArgSpec::constant_arg("seed", 1, "int64", "Value to cache on first call")};
    }

    // Seeding here rather than in init is the point of the fixture: bind is
    // the only phase the engine runs once per call site, so a second call site
    // in the same transaction sees what the first one left.
    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        (void)resolve(params.arguments, params.transaction_opaque_data, *vgi::default_storage());
        return value_schema();
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override { return {1, 1}; }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<OneValue>(
            resolve(params.arguments, params.transaction_opaque_data, *params.storage));
    }
};

}  // namespace

void register_transaction_storage(vgi::Worker& worker) {
    worker.register_table(std::make_shared<TxCachedValue>());
}

}  // namespace example
