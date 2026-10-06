// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Cacheable fixtures whose results depend on a secret, mirroring vgi-python's
// _test_fixtures/secret_cache.py (and vgi-go's examples/table/secret_cache.go).
//
// The DuckDB result cache keys a secret-dependent result on a fingerprint of
// the secrets its bind resolved (never their values): reused while the secret
// is unchanged, recomputed the moment it is rotated, re-fielded or dropped.
// Each fixture reads the `vgi_example` secret's `secret_string` and advertises
// cacheability, one per cache path cache/secret_scope.test drives:
//
//   secret_cache_nonce()       producer; also the data.secret_cache_nonce table
//   secret_cached_scalar(x)    scalar, per-value memoized
//   secret_cached_lateral(x)   blended map, per-value memoized, under LATERAL
//
// Every output carries a nonce minted only when the worker really runs. It is
// random rather than a counter because a pooled worker may run several
// processes, and a per-process counter can repeat across them: equal nonces
// prove a cache HIT, different ones a MISS, on any pool size.
//
// The secret is declared (`required_secrets`) on all three. vgi-python's
// lateral requests it from a two-phase bind instead; what the test observes --
// the value reaching the worker and the fingerprint keying the cache -- is the
// same either way.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "registry.h"
#include "scalar/util.h"

namespace example {
namespace {

constexpr const char* kSecretType = "vgi_example";
// Long enough that a TTL never lapses mid-test.
constexpr int64_t kTtlSeconds = 300;

// A value unique to this invocation across every process in a worker pool:
// 56 random bits, like the reference's os.urandom(7).
int64_t nonce() {
    static thread_local std::mt19937_64 engine{std::random_device{}()};
    return static_cast<int64_t>(engine() >> 8);
}

std::optional<std::string> secret_string(const vgi::Secrets& secrets) {
    return secrets.typed_field(kSecretType, "secret_string");
}

vgi::CacheControl ttl_cache(bool per_value) {
    vgi::CacheControl control;
    control.ttl_seconds = kTtlSeconds;
    control.per_value = per_value;
    return control;
}

std::vector<vgi::SecretLookup> declared_secret() {
    return {{kSecretType, std::nullopt, std::nullopt}};
}

std::shared_ptr<arrow::Schema> secret_nonce_schema() {
    return arrow::schema({arrow::field("secret_string", arrow::utf8(), /*nullable=*/true),
                          arrow::field("nonce", arrow::int64(), /*nullable=*/true)});
}

// `rows` copies of (secret_string, nonce), against `schema` and by column name.
std::shared_ptr<arrow::RecordBatch> secret_nonce_batch(const std::shared_ptr<arrow::Schema>& schema,
                                                       const std::optional<std::string>& value,
                                                       int64_t nonce_value, int64_t rows) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (const auto& field : schema->fields()) {
        if (field->name() == "secret_string") {
            arrow::StringBuilder b;
            for (int64_t i = 0; i < rows; ++i) {
                (void)(value ? b.Append(*value) : b.AppendNull());
            }
            columns.push_back(b.Finish().ValueOrDie());
        } else {
            arrow::Int64Builder b;
            for (int64_t i = 0; i < rows; ++i) (void)b.Append(nonce_value);
            columns.push_back(b.Finish().ValueOrDie());
        }
    }
    return arrow::RecordBatch::Make(schema, rows, std::move(columns));
}

// One row, emitted once, advertising a TTL. Constructed in init, which the
// engine reaches only on a cache MISS, so the nonce is stable across HITs.
class SecretNonceRow : public vgi::TableProducer {
public:
    SecretNonceRow(std::shared_ptr<arrow::Schema> schema, std::optional<std::string> value)
        : schema_(std::move(schema)), value_(std::move(value)), nonce_(nonce()) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (done_) return nullptr;
        done_ = true;
        metadata_ = ttl_cache(false).to_metadata();
        return secret_nonce_batch(schema_, value_, nonce_, 1);
    }

    std::map<std::string, std::string> last_metadata() const override { return metadata_; }

private:
    std::shared_ptr<arrow::Schema> schema_;
    std::optional<std::string> value_;
    int64_t nonce_;
    bool done_ = false;
    std::map<std::string, std::string> metadata_;
};

// `secret_cache_nonce()` — one row: the secret's value and a per-invocation nonce.
class SecretCacheNonce : public vgi::TableFunction {
public:
    std::string name() const override { return "secret_cache_nonce"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "One row with a secret's value and a per-invocation nonce; cacheable per secret";
        md.categories = {"generator", "cache", "secret", "testing"};
        md.required_secrets = declared_secret();
        md.examples = {{"SELECT * FROM secret_cache_nonce()",
                        "The nonce is stable while the vgi_example secret is unchanged",
                        std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return secret_nonce_schema();
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<SecretNonceRow>(params.output_schema,
                                                secret_string(params.secrets));
    }
};

// `secret_cached_scalar(x)` — '<secret_string>|<nonce>', memoized per value per
// secret; '|<nonce>' when no secret resolved. One nonce per call, shared by
// the batch, so a served value keeps the nonce of the call that produced it.
class SecretCachedScalar : public vgi::ScalarFunction {
public:
    std::string name() const override { return "secret_cached_scalar"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Returns '<secret_string>|<nonce>' per value; memoized per value per secret";
        md.return_type = arrow::utf8();
        md.required_secrets = declared_secret();
        md.examples = {{"SELECT secret_cached_scalar(1)",
                        "Stable while the vgi_example secret is unchanged", std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Any value; the output ignores it")};
    }

    std::optional<vgi::CacheControl> cache_control() const override { return ttl_cache(true); }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const auto label =
            secret_string(params.secrets).value_or("") + "|" + std::to_string(nonce());
        arrow::StringBuilder out;
        for (int64_t i = 0; i < batch->num_rows(); ++i) (void)out.Append(label);
        return result(params, out.Finish().ValueOrDie());
    }
};

// `secret_cached_lateral(x)` — a 1->1 blended map emitting the secret's value
// and this call's nonce, memoized per input value per secret under LATERAL.
class SecretCachedLateral : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "secret_cached_lateral"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Blended map emitting a secret's value and a per-call nonce; memoized per secret";
        md.categories = {"blended", "cache", "secret", "test"};
        md.input_from_args = true;
        md.required_secrets = declared_secret();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("x", 0, "int64", "Input column")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return secret_nonce_schema();
    }

    std::optional<vgi::CacheControl> cache_control() const override { return ttl_cache(true); }

    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        return {secret_nonce_batch(params.output_schema, secret_string(params.secrets), nonce(),
                                   batch->num_rows())};
    }
};

}  // namespace

void register_secret_cache(vgi::Worker& worker) {
    worker.register_table(std::make_shared<SecretCacheNonce>());
    worker.register_scalar(std::make_shared<SecretCachedScalar>());
    worker.register_table_in_out(std::make_shared<SecretCachedLateral>());
}

}  // namespace example
