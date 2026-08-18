// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// Same-name-in-two-schemas fixtures.
//
// Two implementations register under the *same* function name in different
// schemas of the `example` catalog. They exist to prove that a
// schema-qualified call reaches the implementation in that schema rather than
// collapsing into one flat by-name entry and failing as an ambiguous overload.
//
// Each tags its output with its own schema, so a mis-routed call shows up in
// the query result instead of being silently plausible.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Identical for both implementations — the collision is the point.
constexpr const char* kFunctionName = "test_same_name_bind";

class SameName : public vgi::ScalarFunction {
public:
    explicit SameName(std::string schema) : schema_(std::move(schema)) {}

    std::string name() const override { return kFunctionName; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Schema-disambiguation probe; the " + schema_ + "-schema implementation";
        md.return_type = arrow::utf8();
        md.examples = {{"SELECT example." + schema_ + "." + kFunctionName + "(1)",
                        "Returns '" + schema_ + ":1'", schema_ + ":1"}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "int64", "Value to tag")};
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
                (void)out.Append(schema_ + ":" + std::to_string(values->Value(i)));
            }
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }

private:
    std::string schema_;
};

}  // namespace

void register_same_name(vgi::Worker& worker) {
    // The primary catalog, not a hardcoded name: one binary stands in for
    // several fixtures, and these belong to whichever it is serving.
    const auto& primary = worker.catalog().name;
    worker.register_scalar_in(primary, "main", std::make_shared<SameName>("main"));
    worker.register_scalar_in(primary, "data", std::make_shared<SameName>("data"));
}

}  // namespace example
