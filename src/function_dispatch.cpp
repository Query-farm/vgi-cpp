// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The function-invocation half of the protocol: `bind`, which settles a call
// site's output schema, and `init`, which opens the stream that carries the
// data.
//
// These two are the only methods whose shape is not fixed by the generated
// schemas: `bind` decides what `init`'s stream will carry, so `init`'s input
// and output schemas are per-call and cannot be declared at registration time.

#include <atomic>
#include <stdexcept>

#include <arrow/record_batch.h>
#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>
#include <vgi_rpc/stream.h>

#include "dispatcher.h"
#include "enums.h"
#include "methods.h"
#include "wire.h"

namespace vgi {

// `GlobalInitResponse` — the header batch every init stream leads with.
//
// Not in the generated schemas because it is a stream *header* rather than a
// method result, so it is spelled out here against the canonical dataclass in
// `vgi/invocation.py`. Registration in dispatcher.cpp needs it too, which is
// why it is not file-local.
const std::shared_ptr<arrow::Schema>& global_init_response_schema() {
    static const auto schema = arrow::schema({
        arrow::field("execution_id", arrow::binary(), /*nullable=*/false),
        arrow::field("max_workers", arrow::int64(), /*nullable=*/false),
        arrow::field("opaque_data", arrow::binary(), /*nullable=*/true),
    });
    return schema;
}

namespace {

// Distinguishes one function execution from another. The engine echoes it on
// follow-up calls; a worker that keeps per-execution state keys on it.
std::string next_execution_id() {
    static std::atomic<uint64_t> counter{0};
    const uint64_t n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    return std::to_string(n);
}

// Feeds each input batch through a scalar function and emits exactly one
// output batch, which is what an exchange stream promises its consumer.
class ScalarExchange : public vgi_rpc::ExchangeState {
public:
    ScalarExchange(std::shared_ptr<ScalarFunction> fn, ProcessParams params)
        : fn_(std::move(fn)), params_(std::move(params)) {}

    void exchange(const vgi_rpc::AnnotatedBatch& input, vgi_rpc::OutputCollector& out,
                  vgi_rpc::CallContext&) override {
        auto result = fn_->process(params_, input.batch);
        if (!result) {
            throw std::runtime_error("scalar function '" + fn_->name() +
                                     "' returned no batch");
        }
        // The engine matches results to inputs positionally, so a row-count
        // mismatch is a silent misalignment rather than an error it can see.
        if (input.batch && result->num_rows() != input.batch->num_rows()) {
            throw std::runtime_error(
                "scalar function '" + fn_->name() + "' returned " +
                std::to_string(result->num_rows()) + " rows for " +
                std::to_string(input.batch->num_rows()) + " input rows");
        }
        out.emit_batch(result);
    }

private:
    std::shared_ptr<ScalarFunction> fn_;
    ProcessParams params_;
};

}  // namespace

std::vector<std::shared_ptr<ScalarFunction>> Dispatcher::scalars_named(
    const std::string& name) const {
    std::vector<std::shared_ptr<ScalarFunction>> found;
    auto it = scalar_by_name_.find(name);
    if (it == scalar_by_name_.end()) return found;
    found.reserve(it->second.size());
    for (size_t index : it->second) found.push_back(scalars_[index]);
    return found;
}

std::shared_ptr<ScalarFunction> Dispatcher::resolve_scalar(const std::string& name,
                                                           const BindParams& params) const {
    auto candidates = scalars_named(name);
    if (candidates.empty()) {
        throw std::invalid_argument("no scalar function named '" + name + "'");
    }
    if (candidates.size() == 1) return candidates.front();

    // Score each overload by how many declared argument types the engine's
    // resolved types match exactly. The engine has already narrowed to
    // something callable, so this only has to break the remaining tie — and a
    // polymorphic parameter matches anything, which is what makes `any_mixed`
    // lose to a concrete overload rather than shadow it.
    std::shared_ptr<ScalarFunction> best;
    int best_score = -1;
    for (const auto& candidate : candidates) {
        const auto specs = candidate->argument_specs();
        int score = 0;
        bool viable = true;
        for (size_t i = 0; i < specs.size(); ++i) {
            const auto declared = specs[i].arrow_type;
            auto actual = params.input_type(i);
            if (!declared || declared->id() == arrow::Type::NA) continue;  // polymorphic
            if (!actual) continue;
            if (declared->Equals(*actual)) {
                ++score;
            } else {
                viable = false;
                break;
            }
        }
        if (viable && score > best_score) {
            best_score = score;
            best = candidate;
        }
    }
    // Nothing matched exactly: fall back to the first registration rather than
    // failing, since the engine would not have called us if the call site were
    // uncallable, and its own resolution is authoritative.
    return best ? best : candidates.front();
}

// Enforce each argument's declared type bound against the type the engine
// resolved.
//
// The check exists so the error names the function and argument. Without it a
// bound violation surfaces from inside process() as an Arrow cast failure,
// which mentions neither, and the user is left guessing which of several
// arguments was wrong.
void Dispatcher::check_type_bounds(const ScalarFunction& fn, const BindParams& params) {
    const auto specs = fn.argument_specs();
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& spec = specs[i];
        if (!spec.type_bound || !spec.type_bound->accepts) continue;
        auto type = params.input_type(i);
        // No resolved type means the engine did not supply one — nothing to
        // check against, and refusing here would reject a legitimate call.
        if (!type) continue;
        if (!spec.type_bound->accepts(*type)) {
            throw std::invalid_argument("function '" + fn.name() + "' argument '" + spec.name +
                                        "' requires " + spec.type_bound->name + ", got " +
                                        type->ToString());
        }
    }
}

BindParams Dispatcher::read_bind_request(
    const std::shared_ptr<arrow::RecordBatch>& bind_call) const {
    BindParams params;
    params.input_schema = wire::get_schema(bind_call, "input_schema");
    params.arguments =
        Arguments::parse(wire::get_optional_binary(bind_call, "arguments").value_or(""));
    params.schema_name = wire::get_optional_string(bind_call, "schema_name").value_or("main");
    params.catalog_name = catalog_.name;
    return params;
}

vgi_rpc::Result Dispatcher::bind(const vgi_rpc::Request& request) {
    auto bind_call = wire::get_ipc(request.batch(), "request");
    if (!bind_call) throw std::runtime_error("bind: empty request");

    const auto function_name = wire::get_string(bind_call, "function_name");
    const auto params = read_bind_request(bind_call);
    auto fn = resolve_scalar(function_name, params);
    check_type_bounds(*fn, params);
    auto output_schema = fn->bind(params);
    if (!output_schema) {
        throw std::runtime_error("function '" + function_name + "' bound to no schema");
    }

    auto payload = wire::ResultBuilder(payload_schema_of("bind"))
                       .set_binary("output_schema", wire::encode_schema(output_schema))
                       // Nothing to carry from bind to init yet. When a
                       // function needs bind-time state, this is where it
                       // travels — the engine treats it as opaque.
                       .set_binary("opaque_data", "")
                       .fill_defaults()
                       .finish();
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
                                      .set_binary("result", wire::encode_ipc(payload))
                                      .finish());
}

vgi_rpc::Stream Dispatcher::init(const vgi_rpc::Request& request) {
    auto init_request = wire::get_ipc(request.batch(), "request");
    if (!init_request) throw std::runtime_error("init: empty request");

    auto bind_call = wire::get_ipc(init_request, "bind_call");
    if (!bind_call) throw std::runtime_error("init: request carries no bind_call");

    const auto function_name = wire::get_string(bind_call, "function_name");
    auto fn = resolve_scalar(function_name, read_bind_request(bind_call));

    // The engine sends back the schema bind settled on rather than making the
    // worker re-derive it, so a function whose bind is expensive pays once.
    auto output_schema = wire::get_schema(init_request, "output_schema");
    if (!output_schema) throw std::runtime_error("init: request carries no output_schema");

    ProcessParams params;
    params.output_schema = output_schema;
    // Constant arguments were evaluated at bind and ride along on every call,
    // which is why a const parameter never appears in the input batch.
    params.arguments =
        Arguments::parse(wire::get_optional_binary(bind_call, "arguments").value_or(""));
    params.catalog_name = catalog_.name;
    params.schema_name = wire::get_optional_string(bind_call, "schema_name").value_or("main");

    auto header = wire::ResultBuilder(global_init_response_schema())
                      .set_binary("execution_id", next_execution_id())
                      .set_int64("max_workers", 1)
                      .set_null("opaque_data")
                      .finish();

    auto input_schema = wire::get_schema(bind_call, "input_schema");
    if (!input_schema) input_schema = arrow::schema({});

    vgi_rpc::Stream stream;
    stream.output_schema = std::move(output_schema);
    stream.input_schema = std::move(input_schema);
    stream.state = std::make_shared<ScalarExchange>(std::move(fn), std::move(params));
    stream.header = std::move(header);
    return stream;
}

}  // namespace vgi
