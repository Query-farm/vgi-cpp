// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Fixtures that resolve secrets.

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// The secret type these fixtures ask for. Declared in the catalog's
// `secret_types`, which is what lets a user write `CREATE SECRET … (TYPE
// vgi_example, …)` at all.
constexpr const char* kSecretType = "vgi_example";

// The parameter types, echoed back so a test can confirm the secret's schema
// survived the round trip rather than being flattened to strings.
const char* arrow_type_of(const std::string& field) {
    if (field == "port") return "int32";
    if (field == "use_ssl") return "bool";
    if (field == "timeout") return "double";
    return "string";
}

std::shared_ptr<arrow::Array> strings(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    (void)builder.Reserve(static_cast<int64_t>(values.size()));
    for (const auto& value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// A batch whose columns are asked for one at a time, by the name the *bound*
// schema gives them.
//
// Driven by the schema rather than built to the declared shape: the engine
// narrows a scan's columns to those the query reads, and a fixed-shape batch
// then disagrees with the schema it is shipped under — which Arrow does not
// check and the consumer discovers as a crash.
std::shared_ptr<arrow::RecordBatch> batch_from(
    const std::shared_ptr<arrow::Schema>& schema, int64_t rows,
    const std::function<std::shared_ptr<arrow::Array>(const std::string&)>& column) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(static_cast<size_t>(schema->num_fields()));
    for (const auto& field : schema->fields()) columns.push_back(column(field->name()));
    return arrow::RecordBatch::Make(schema, rows, columns);
}

// Emits one prepared batch, then stops. Every fixture here knows its whole
// result at init, so the producer carries no scan position.
class OneShot : public vgi::TableProducer {
public:
    explicit OneShot(std::shared_ptr<arrow::RecordBatch> batch)
        : batch_(std::move(batch)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        return std::exchange(batch_, nullptr);
    }

private:
    std::shared_ptr<arrow::RecordBatch> batch_;
};

// `secret_demo()` — one row per field of the resolved `vgi_example` secret.
class SecretDemo : public vgi::TableFunction {
public:
    std::string name() const override { return "secret_demo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Outputs secret contents as key-value rows";
        md.categories = {"generator", "utility"};
        // Asking here is what triggers the engine's two-phase resolution: the
        // first bind returns this list, the engine resolves it, and binds
        // again with the values in `params.secrets`.
        md.required_secrets = {{kSecretType, std::nullopt, std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("key", arrow::utf8(), true),
                              arrow::field("value", arrow::utf8(), true),
                              arrow::field("arrow_type", arrow::utf8(), true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        std::vector<std::string> keys;
        std::vector<std::string> values;
        std::vector<std::string> types;
        // By *type*, not by name: the user chose the secret's name in
        // `CREATE SECRET`, so a fixture cannot know it. Looking it up by name
        // silently found nothing and emitted zero rows.
        if (const auto* secret = params.secrets.of_type(kSecretType)) {
            for (const auto& [field, value] : *secret) {
                keys.push_back(field);
                values.push_back(value);
                types.emplace_back(arrow_type_of(field));
            }
        }

        auto batch = batch_from(
            params.output_schema, static_cast<int64_t>(keys.size()),
            [&](const std::string& name) {
                if (name == "key") return strings(keys);
                if (name == "value") return strings(values);
                return strings(types);
            });
        return std::make_unique<OneShot>(std::move(batch));
    }
};

// `scoped_secret_demo(path)` — resolve the secret whose scope covers `path`.
//
// What it probes that `multi_secret_demo` does not: the lookup's *scope* is
// the argument, so which secret is wanted cannot be known until the call site
// is. A scan of a user-supplied URL has to ask this way; a fixed set of
// scopes, as below, is the easier half of the problem.
class ScopedSecretDemo : public vgi::TableFunction {
public:
    std::string name() const override { return "scoped_secret_demo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Demo: resolves scoped secret based on argument";
        md.categories = {"generator", "utility"};
        return md;
    }

    // Asked per call site rather than declared in metadata(): the scope is an
    // argument, and metadata() is answered before any call site exists.
    std::vector<vgi::SecretLookup> secret_lookups(
        const vgi::BindParams& params) const override {
        return {{kSecretType, params.arguments.const_string(0), std::nullopt}};
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("path", 0, "varchar",
                                           "Scope path for secret lookup")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("scope", arrow::utf8(), true),
                              arrow::field("found", arrow::boolean(), true),
                              arrow::field("secret_keys", arrow::utf8(), true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto path = params.arguments.const_string(0).value_or("");
        // Selected by path rather than taken as "the first secret of this
        // type": a secret resolved for some earlier call's scope would
        // otherwise answer for a path it does not cover.
        const auto* secret = params.secrets.for_scope_of_type(path, kSecretType);

        std::string keys;
        if (secret) {
            for (const auto& [field, value] : *secret) {
                (void)value;
                if (!keys.empty()) keys.push_back(',');
                keys += field;
            }
        }

        auto batch = batch_from(
            params.output_schema, 1, [&](const std::string& name) {
                if (name == "found") {
                    arrow::BooleanBuilder builder;
                    (void)builder.Append(secret != nullptr);
                    std::shared_ptr<arrow::Array> array;
                    (void)builder.Finish(&array);
                    return array;
                }
                return strings({name == "scope" ? path : keys});
            });
        return std::make_unique<OneShot>(std::move(batch));
    }

private:
};

// `multi_secret_demo(path)` — two same-type secrets resolved in one bind.
//
// What it probes that `scoped_secret_demo` does not: resolved secrets are
// keyed by the user's secret *name*, so asking for one type at two scopes
// yields two secrets rather than one overwriting the other. Choosing between
// them per path is then the worker's job.
class MultiSecretDemo : public vgi::TableFunction {
public:
    std::string name() const override { return "multi_secret_demo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Demo: two same-type scoped secrets resolved in one bind";
        md.categories = {"generator", "utility"};
        md.required_secrets = {{kSecretType, "s3://bucket-a/", std::nullopt},
                               {kSecretType, "s3://bucket-b/", std::nullopt}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("path", 0, "varchar",
                                           "Path for scoped secret lookup")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("api_key", arrow::utf8(), true)});
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto path = params.arguments.const_string(0).value_or("");
        std::string api_key;
        if (const auto* secret = params.secrets.for_scope_of_type(path, kSecretType)) {
            auto value = secret->find("api_key");
            if (value != secret->end()) api_key = value->second;
        }

        auto batch = batch_from(params.output_schema, 1,
                                [&](const std::string&) { return strings({api_key}); });
        return std::make_unique<OneShot>(std::move(batch));
    }
};

}  // namespace

void register_secret_fixtures(vgi::Worker& worker) {
    worker.register_table(std::make_shared<SecretDemo>());
    worker.register_table(std::make_shared<ScopedSecretDemo>());
    worker.register_table(std::make_shared<MultiSecretDemo>());
}

}  // namespace example
