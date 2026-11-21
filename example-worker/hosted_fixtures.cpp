// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The Orchard hosted-mode fixture: `pass_claims(x)`, registered only when
// `VGI_ORCHARD_HOSTED` is on, so the integration suite's function counts are
// unaffected. The C++ twin of vgi-python's `_test_fixtures/orchard_hosted.py`.

#include <memory>
#include <string>
#include <vector>

#include <arrow/builder.h>
#include <nlohmann/json.hpp>

#include <vgi/hosted.h>
#include <vgi/worker.h>

#include "registry.h"
#include "scalar/util.h"

namespace example {

namespace {

// `pass_claims(x)` — the fixture function of the SDK-neutral hosted-mode
// conformance harness (the extension repo's `test/hosted/`): one row per input
// row, each the JSON object {"principal", "domain", "claims"}, where `claims`
// is the verified claim set as the auth context carries it. Claims keep their JSON types:
// `email_verified` is a boolean and `exp` a number, so a test can tell a real boolean from the
// string "true".
class PassClaims : public vgi::ScalarFunction {
public:
    std::string name() const override { return "pass_claims"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Return the caller's verified Pass claims as JSON";
        md.return_type = arrow::utf8();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("x", 0, "int64", "dummy input")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        const nlohmann::json payload = {
            {"principal", params.auth.principal ? nlohmann::json(*params.auth.principal)
                                                : nlohmann::json(nullptr)},
            {"domain", params.auth.domain},
            {"claims", params.auth.claims}};
        // nlohmann orders object keys, as the reference's sort_keys does.
        const std::string text = payload.dump();
        arrow::StringBuilder out;
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            if (auto st = out.Append(text); !st.ok()) throw std::runtime_error(st.ToString());
        }
        std::shared_ptr<arrow::Array> array;
        if (auto st = out.Finish(&array); !st.ok()) throw std::runtime_error(st.ToString());
        return result(params, array);
    }
};

}  // namespace

void register_hosted_fixtures(vgi::Worker& worker) {
    bool hosted = false;
    try {
        hosted = vgi::hosted::hosted_mode_enabled();
    } catch (const vgi::hosted::ConfigError&) {
        return;  // Worker::run refuses to start with the reason
    }
    if (hosted) worker.register_scalar(std::make_shared<PassClaims>());
}

}  // namespace example
