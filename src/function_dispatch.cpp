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
#include <chrono>
#include <functional>
#include <sstream>

#include <unistd.h>
#include <stdexcept>

#include <arrow/record_batch.h>
#include <arrow/util/key_value_metadata.h>
#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>
#include <vgi_rpc/stream.h>

#include "dispatcher.h"
#include "arg_schema.h"
#include "enums.h"
#include "methods.h"
#include "vgi/storage.h"

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

}  // namespace

// Distinguishes one function execution from another. The engine echoes it on
// follow-up calls; a worker that keeps per-execution state keys on it.
//
// Unique across *processes*, not just within one. The cross-process store in
// storage.cpp is rooted at a path shared by every worker of this uid and
// scopes entries by execution id, so a per-process counter starting at zero
// makes two concurrent queries alias the same directory — each then finalizes
// over the other's partials and both return wrong numbers, with no error. A
// killed worker's leftover state would likewise be replayed by the next query
// to reach the same id.
//
// pid and start time together survive pid reuse; the counter separates
// executions within one process. Both references do the same (Python mints a
// uuid4, Rust pid+time+counter).
std::string next_execution_id() {
    static const std::string prefix = [] {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto wall = std::chrono::system_clock::now().time_since_epoch();
        std::ostringstream out;
        out << std::hex << static_cast<long>(::getpid()) << '-'
            << std::chrono::duration_cast<std::chrono::nanoseconds>(wall).count() << '-'
            << std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() << '-';
        return out.str();
    }();
    static std::atomic<uint64_t> counter{0};
    return prefix + std::to_string(counter.fetch_add(1, std::memory_order_relaxed) + 1);
}

namespace {

// Feeds each input batch through a scalar function and emits exactly one
// output batch, which is what an exchange stream promises its consumer.
// Renders a cache advertisement onto a batch, if the function makes one.
std::shared_ptr<arrow::KeyValueMetadata> cache_metadata(
    const std::optional<CacheControl>& control) {
    if (!control) return nullptr;
    const auto rendered = control->to_metadata();
    std::vector<std::string> keys;
    std::vector<std::string> values;
    keys.reserve(rendered.size());
    values.reserve(rendered.size());
    for (const auto& [key, value] : rendered) {
        keys.push_back(key);
        values.push_back(value);
    }
    return arrow::key_value_metadata(keys, values);
}

class ScalarExchange : public vgi_rpc::ExchangeState {
public:
    ScalarExchange(std::shared_ptr<ScalarFunction> fn, ProcessParams params)
        : fn_(std::move(fn)),
          params_(std::move(params)),
          cache_(cache_metadata(fn_->cache_control())) {}

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
        if (cache_) {
            out.emit_batch(result, cache_);
        } else {
            out.emit_batch(result);
        }
    }

private:
    std::shared_ptr<ScalarFunction> fn_;
    ProcessParams params_;
    std::shared_ptr<arrow::KeyValueMetadata> cache_;
};

// Drives a TableProducer: one output batch per tick until the scan is
// exhausted, then finish().
//
// The engine ticks a producer stream with an empty batch and reads whatever
// comes back, so "no more rows" has to be signalled explicitly — a producer
// that simply stops emitting would hang the scan.
// Score one candidate's declared argument types against the types the engine
// resolved. Returns -1 when the candidate cannot serve this call.
//
// The engine has already narrowed to something callable, so this only breaks
// the remaining tie. A polymorphic parameter matches anything and scores
// nothing, which is what makes a concrete overload win over an `any` one
// rather than the other way round.
int score_overload(const std::vector<ArgSpec>& specs,
                   const std::function<std::shared_ptr<arrow::DataType>(size_t)>& actual_type) {
    int score = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        // An exact Arrow type wins, but a VGI type *name* is just as much a
        // declaration — `constant_arg(..., "varchar", ...)` names utf8 and has
        // to take part in resolution, or two overloads that differ only by
        // their named types are indistinguishable and the first always wins.
        auto declared = specs[i].arrow_type;
        if (!declared) declared = arg_type_to_arrow(specs[i].type);
        if (!declared || declared->id() == arrow::Type::NA) continue;
        auto actual = actual_type(i);
        if (!actual) continue;
        if (declared->Equals(*actual)) {
            ++score;
        } else {
            return -1;
        }
    }
    return score;
}

vgi_rpc::LogLevel to_rpc_level(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return vgi_rpc::LogLevel::DEBUG;
        case LogLevel::Warning: return vgi_rpc::LogLevel::WARN;
        case LogLevel::Error: return vgi_rpc::LogLevel::ERROR;
        case LogLevel::Info: break;
    }
    return vgi_rpc::LogLevel::INFO;
}

class TableProduce : public vgi_rpc::ProducerState {
public:
    TableProduce(std::unique_ptr<TableProducer> producer, PushdownFilters filters = {})
        : producer_(std::move(producer)), filters_(std::move(filters)) {}

    void produce(vgi_rpc::OutputCollector& out, vgi_rpc::CallContext&) override {
        // A null producer is the buffering sink phase, which emits nothing:
        // its batches arrive through table_buffering_process instead.
        if (!producer_) {
            out.finish();
            return;
        }
        // Bound per tick, since the collector is created per tick: a producer
        // holding one from an earlier tick would write into a dead sink.
        producer_->set_log([&out](LogLevel level, const std::string& message) {
            out.client_log(to_rpc_level(level), message);
        });
        auto batch = producer_->next_batch();
        if (!batch) {
            out.finish();
            return;
        }
        // Applied here rather than in the producer so a function that only
        // advertises the capability gets it for free, and one that uses the
        // filters itself is not filtered twice.
        if (!filters_.empty()) batch = filters_.apply(batch);

        // Validated before it leaves.
        //
        // A batch whose arrays disagree with their type — a StructArray with
        // two children for a one-field type, say, which is what projection
        // pushdown into a struct produces if a producer builds a fixed shape —
        // is accepted by Arrow's constructors and corrupts the *consumer*.
        // That surfaces as a crash in an unrelated query, arbitrarily later,
        // and cost a full bisect to attribute. Checking here turns it into an
        // error naming the function.
        if (auto status = batch->ValidateFull(); !status.ok()) {
            throw std::runtime_error("producer emitted an invalid batch: " + status.ToString());
        }
        const auto metadata = producer_->last_metadata();
        if (metadata.empty()) {
            out.emit_batch(batch);
            return;
        }
        std::vector<std::string> keys;
        std::vector<std::string> values;
        keys.reserve(metadata.size());
        values.reserve(metadata.size());
        for (const auto& [key, value] : metadata) {
            keys.push_back(key);
            values.push_back(value);
        }
        out.emit_batch(batch, arrow::key_value_metadata(keys, values));
    }

private:
    std::unique_ptr<TableProducer> producer_;
    PushdownFilters filters_;
};

// Drives a table-in-out function: one input batch in, zero or more out.
//
// Unlike a scalar exchange there is no row-count relationship to enforce —
// that a batch may fan out or collapse is the whole point of the shape.
class TableInOutExchange : public vgi_rpc::ExchangeState {
public:
    TableInOutExchange(std::shared_ptr<TableInOutFunction> fn, ProcessParams params)
        : fn_(std::move(fn)),
          params_(std::move(params)),
          cache_(cache_metadata(fn_->cache_control())) {}

    void exchange(const vgi_rpc::AnnotatedBatch& input, vgi_rpc::OutputCollector& out,
                  vgi_rpc::CallContext&) override {
        for (const auto& batch : fn_->process(params_, input.batch)) {
            if (!batch) continue;
            if (cache_) {
                out.emit_batch(batch, cache_);
            } else {
                out.emit_batch(batch);
            }
        }
    }

private:
    std::shared_ptr<TableInOutFunction> fn_;
    ProcessParams params_;
    std::shared_ptr<arrow::KeyValueMetadata> cache_;
};

}  // namespace

std::vector<std::shared_ptr<TableBufferingFunction>> Dispatcher::bufferings_in_schema(
    const std::string& schema) const {
    std::vector<std::shared_ptr<TableBufferingFunction>> found;
    for (size_t i = 0; i < bufferings_.size(); ++i) {
        if (buffering_scopes_[i].schema == schema) found.push_back(bufferings_[i]);
    }
    return found;
}

std::shared_ptr<TableBufferingFunction> Dispatcher::find_buffering(
    const std::string& name, const std::string& schema) const {
    auto it = buffering_by_name_.find(name);
    if (it == buffering_by_name_.end()) return nullptr;
    for (size_t index : it->second) {
        if (schema.empty() || buffering_scopes_[index].schema == schema) {
            return bufferings_[index];
        }
    }
    // No cross-schema fallback — see find_table.
    return nullptr;
}

std::shared_ptr<TableBufferingFunction> Dispatcher::require_buffering(
    const std::string& name, const std::string& schema) const {
    if (auto fn = find_buffering(name, schema)) return fn;
    throw std::invalid_argument("no buffering function named '" + name + "'");
}

std::vector<std::shared_ptr<AggregateFunction>> Dispatcher::aggregates_in_schema(
    const std::string& schema) const {
    std::vector<std::shared_ptr<AggregateFunction>> found;
    for (size_t i = 0; i < aggregates_.size(); ++i) {
        if (aggregate_scopes_[i].schema == schema) found.push_back(aggregates_[i]);
    }
    return found;
}

std::shared_ptr<AggregateFunction> Dispatcher::require_aggregate(
    const std::string& name, const std::string& schema) const {
    auto it = aggregate_by_name_.find(name);
    if (it != aggregate_by_name_.end()) {
        for (size_t index : it->second) {
            if (schema.empty() || aggregate_scopes_[index].schema == schema) {
                return aggregates_[index];
            }
        }
    }
    // Naming the schema, because "no aggregate named x" when x exists in
    // another schema sends the reader looking in the wrong place.
    throw std::invalid_argument("no aggregate function named '" + name + "' in schema '" +
                                schema + "'");
}

std::vector<std::shared_ptr<TableInOutFunction>> Dispatcher::table_in_outs_in_schema(
    const std::string& schema) const {
    std::vector<std::shared_ptr<TableInOutFunction>> found;
    for (size_t i = 0; i < table_in_outs_.size(); ++i) {
        if (table_in_out_scopes_[i].schema == schema) found.push_back(table_in_outs_[i]);
    }
    return found;
}

std::shared_ptr<TableInOutFunction> Dispatcher::find_table_in_out(
    const std::string& name, const std::string& schema) const {
    auto it = table_in_out_by_name_.find(name);
    if (it == table_in_out_by_name_.end()) return nullptr;
    for (size_t index : it->second) {
        if (schema.empty() || table_in_out_scopes_[index].schema == schema) {
            return table_in_outs_[index];
        }
    }
    // No cross-schema fallback — see find_table.
    return nullptr;
}

std::vector<std::shared_ptr<TableFunction>> Dispatcher::tables_in_schema(
    const std::string& schema) const {
    std::vector<std::shared_ptr<TableFunction>> found;
    for (size_t i = 0; i < tables_.size(); ++i) {
        if (table_scopes_[i].schema == schema) found.push_back(tables_[i]);
    }
    return found;
}

std::shared_ptr<TableFunction> Dispatcher::find_table(const std::string& name,
                                                      const std::string& schema) const {
    return find_table(name, schema, nullptr);
}

std::shared_ptr<TableFunction> Dispatcher::find_table(const std::string& name,
                                                      const std::string& schema,
                                                      const BindParams* params) const {
    auto it = table_by_name_.find(name);
    if (it == table_by_name_.end()) return nullptr;

    // The named schema, and only it.
    //
    // Falling back to the first registration when no scope matches makes a
    // table function declared in schema `a` answer a call in schema `main`,
    // shadowing a scalar of the same name that should have served it. Since
    // binds always carry a schema (defaulted to `main`), the fallback fired on
    // every genuine mismatch rather than on the unqualified call it was
    // written for.
    std::vector<std::shared_ptr<TableFunction>> candidates;
    for (size_t index : it->second) {
        if (schema.empty() || table_scopes_[index].schema == schema) {
            candidates.push_back(tables_[index]);
        }
    }
    if (candidates.empty()) return nullptr;
    if (candidates.size() == 1 || !params) return candidates.front();

    // Table functions overload exactly as scalars do — `repeat_value` is
    // registered once per element type — so the same scoring applies. Without
    // it the first registration wins and a varchar call reaches the int64
    // overload, failing inside the cast rather than at resolution.
    std::shared_ptr<TableFunction> best;
    int best_score = -1;
    for (const auto& candidate : candidates) {
        const int score = score_overload(candidate->argument_specs(),
                                         [&](size_t i) { return params->input_type(i); });
        if (score > best_score) {
            best_score = score;
            best = candidate;
        }
    }
    return best ? best : candidates.front();
}

std::vector<std::shared_ptr<ScalarFunction>> Dispatcher::scalars_in_schema(
    const std::string& schema) const {
    std::vector<std::shared_ptr<ScalarFunction>> found;
    for (size_t i = 0; i < scalars_.size(); ++i) {
        if (scalar_scopes_[i].schema == schema) found.push_back(scalars_[i]);
    }
    return found;
}

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

    // Narrow to the schema the call named before considering types. Two
    // schemas may declare the same name with different implementations, and
    // routing a schema-qualified call to the wrong one is silently plausible —
    // the fixtures tag their output with their schema precisely so a
    // mis-routed call is visible in the result.
    if (!params.schema_name.empty()) {
        std::vector<std::shared_ptr<ScalarFunction>> in_schema;
        auto it = scalar_by_name_.find(name);
        for (size_t index : it->second) {
            if (scalar_scopes_[index].schema == params.schema_name) {
                in_schema.push_back(scalars_[index]);
            }
        }
        if (!in_schema.empty()) candidates = std::move(in_schema);
    }

    if (candidates.size() == 1) return candidates.front();

    std::shared_ptr<ScalarFunction> best;
    int best_score = -1;
    for (const auto& candidate : candidates) {
        const int score = score_overload(
            candidate->argument_specs(),
            [&](size_t i) { return params.input_type(i); });
        if (score > best_score) {
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

void Dispatcher::check_arg_constraints(const std::string& function_name,
                                       const std::vector<ArgSpec>& specs,
                                       const Arguments& arguments) {
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& spec = specs[i];
        if (!spec.ge && !spec.le && !spec.gt && !spec.lt) continue;
        // Only a constant can be checked here: a column's values are not known
        // until process(), and the engine re-checks nothing on our behalf.
        if (!spec.constant) continue;
        const auto value = arguments.const_double(i);
        if (!value) continue;

        const auto refuse = [&](const std::string& bound) {
            throw std::invalid_argument("function '" + function_name + "' argument '" +
                                        spec.name + "' must be " + bound);
        };
        if (spec.ge && *value < *spec.ge) refuse(">= " + std::to_string(*spec.ge));
        if (spec.gt && *value <= *spec.gt) refuse("> " + std::to_string(*spec.gt));
        if (spec.le && *value > *spec.le) refuse("<= " + std::to_string(*spec.le));
        if (spec.lt && *value >= *spec.lt) refuse("< " + std::to_string(*spec.lt));
    }
}

// The secrets `name` declares, whichever registry it lives in.
//
// Answered only on the *first* bind: once the engine has resolved them it sets
// `resolved_secrets_provided`, and asking again would loop.
std::vector<SecretLookup> Dispatcher::required_secrets_of(const std::string& name,
                                                          const BindParams& params) const {
    if (params.secrets_resolved) return {};
    if (auto fn = find_buffering(name, params.schema_name)) {
        // A COPY writer scopes its lookup to the destination path, which only
        // the bind params carry — so ask it rather than reading a static list.
        for (const auto& writer : copy_to_) {
            if (writer->handler_name() == name) return writer->secret_lookups(params);
        }
        return fn->metadata().required_secrets;
    }
    if (auto fn = find_table_in_out(name, params.schema_name)) {
        return fn->secret_lookups(params);
    }
    if (auto fn = find_table(name, params.schema_name, &params)) {
        return fn->secret_lookups(params);
    }
    auto candidates = scalars_named(name);
    if (!candidates.empty()) return candidates.front()->secret_lookups(params);
    return {};
}

BindParams Dispatcher::read_bind_request(
    const std::shared_ptr<arrow::RecordBatch>& bind_call) const {
    BindParams params;
    params.input_schema = wire::get_schema(bind_call, "input_schema");
    params.arguments =
        Arguments::parse(wire::get_optional_binary(bind_call, "arguments").value_or(""));
    params.settings =
        Settings::parse(wire::get_optional_binary(bind_call, "settings").value_or(""));
    params.secrets =
        Secrets::parse(wire::get_optional_binary(bind_call, "secrets").value_or(""));
    params.schema_name = wire::get_optional_string(bind_call, "schema_name").value_or("main");
    params.catalog_name = catalog_.name;
    params.secrets_resolved =
        wire::get_optional_bool(bind_call, "resolved_secrets_provided").value_or(false);
    // `copy_to` rides as a nested struct column rather than IPC bytes, unlike
    // most of this request's compound fields.
    if (auto copy_to = wire::get_struct_fields(bind_call, "copy_to")) {
        auto format = copy_to->find("format");
        auto path = copy_to->find("file_path");
        if (format != copy_to->end()) params.copy_to_format = format->second;
        if (path != copy_to->end()) params.copy_to_path = path->second;
    }
    if (auto copy_from = wire::get_struct_fields(bind_call, "copy_from")) {
        auto format = copy_from->find("format");
        auto path = copy_from->find("file_path");
        if (format != copy_from->end()) params.copy_from_format = format->second;
        if (path != copy_from->end()) params.copy_from_path = path->second;
        // The target's columns ride the same struct as IPC bytes, so they are
        // read separately rather than through the stringified fields above.
        params.copy_from_schema = wire::decode_schema(
            wire::get_struct_binary(bind_call, "copy_from", "expected_schema").value_or(""));
    }
    return params;
}

vgi_rpc::Result Dispatcher::bind(const vgi_rpc::Request& request) {
    auto bind_call = wire::get_ipc(request.batch(), "request");
    if (!bind_call) throw std::runtime_error("bind: empty request");

    const auto function_name = wire::get_string(bind_call, "function_name");
    const auto params = read_bind_request(bind_call);

    // The request says which registry to use; consult it before guessing.
    //
    // Probing by name alone resolves a name registered in two kinds to
    // whichever probe runs first, which is arbitrary. A COPY format's reader
    // and writer deliberately share a name, so this is not hypothetical.
    const auto declared_kind = wire::get_optional_enum(bind_call, "function_type");
    std::shared_ptr<arrow::Schema> output_schema;
    if (auto sink = find_buffering(function_name, params.schema_name)) {
        output_schema = sink->bind(params);
    } else if (auto transform = find_table_in_out(function_name, params.schema_name)) {
        check_arg_constraints(function_name, transform->argument_specs(), params.arguments);
        output_schema = transform->bind(params);
    } else if (auto table = find_table(function_name, params.schema_name, &params)) {
        check_arg_constraints(function_name, table->argument_specs(), params.arguments);
        output_schema = table->bind(params);
    } else {
        auto fn = resolve_scalar(function_name, params);
        check_type_bounds(*fn, params);
        check_arg_constraints(function_name, fn->argument_specs(), params.arguments);
        output_schema = fn->bind(params);
    }
    if (!output_schema) {
        throw std::runtime_error("function '" + function_name + "' bound to no schema");
    }

    // A function that needs secrets says so here; the engine resolves them and
    // binds again with the values in place.
    std::vector<std::string> secret_types;
    std::vector<std::string> secret_scopes;
    std::vector<std::string> secret_names;
    for (const auto& lookup : required_secrets_of(function_name, params)) {
        secret_types.push_back(lookup.secret_type);
        // Absent scope and name travel as empty strings, since the columns are
        // parallel lists and must stay the same length.
        secret_scopes.push_back(lookup.scope.value_or(""));
        secret_names.push_back(lookup.secret_name.value_or(""));
    }

    auto payload = wire::ResultBuilder(payload_schema_of("bind"))
                       .set_string_list("lookup_secret_types", secret_types)
                       .set_string_list("lookup_scopes", secret_scopes)
                       .set_string_list("lookup_names", secret_names)
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

vgi_rpc::Result Dispatcher::table_function_cardinality(const vgi_rpc::Request& request) {
    auto cardinality_request = wire::get_ipc(request.batch(), "request");
    if (!cardinality_request) throw std::runtime_error("cardinality: empty request");

    auto bind_call = wire::get_ipc(cardinality_request, "bind_call");
    if (!bind_call) throw std::runtime_error("cardinality: request carries no bind_call");

    const auto function_name = wire::get_string(bind_call, "function_name");
    const auto bind_params = read_bind_request(bind_call);

    // Asked before init, so there is no execution and no output schema yet —
    // only the arguments the estimate can be derived from. `estimate` and
    // `max` stay null when the function does not answer, which the planner
    // reads as "unknown" rather than as zero.
    TableCardinality cardinality;
    if (auto table = find_table(function_name, bind_params.schema_name, &bind_params)) {
        ProcessParams params;
        params.arguments = bind_params.arguments;
        params.settings = bind_params.settings;
        params.secrets = bind_params.secrets;
        params.catalog_name = catalog_.name;
        params.schema_name = bind_params.schema_name;
        params.storage = default_storage();
        cardinality = table->cardinality(params);
    }

    wire::ResultBuilder payload(payload_schema_of("table_function_cardinality"));
    if (cardinality.estimate) {
        payload.set_int64("estimate", *cardinality.estimate);
    } else {
        payload.set_null("estimate");
    }
    if (cardinality.max) {
        payload.set_int64("max", *cardinality.max);
    } else {
        payload.set_null("max");
    }
    return vgi_rpc::Result::value(
        wire::ResultBuilder(envelope_schema())
            .set_binary("result", wire::encode_ipc(payload.finish()))
            .finish());
}

vgi_rpc::Result Dispatcher::table_function_statistics(const vgi_rpc::Request& request) {
    auto statistics_request = wire::get_ipc(request.batch(), "request");
    if (!statistics_request) throw std::runtime_error("statistics: empty request");

    auto bind_call = wire::get_ipc(statistics_request, "bind_call");
    if (!bind_call) throw std::runtime_error("statistics: request carries no bind_call");

    const auto function_name = wire::get_string(bind_call, "function_name");
    const auto bind_params = read_bind_request(bind_call);

    std::optional<std::vector<ColumnStatistics>> statistics;
    if (auto table = find_table(function_name, bind_params.schema_name, &bind_params)) {
        ProcessParams params;
        params.arguments = bind_params.arguments;
        params.settings = bind_params.settings;
        params.secrets = bind_params.secrets;
        params.catalog_name = catalog_.name;
        params.schema_name = bind_params.schema_name;
        params.storage = default_storage();
        statistics = table->statistics(params);
    }

    // A Binary method: the bytes travel in the envelope verbatim, and a null
    // there is "no statistics" — distinct from an empty set of them.
    wire::ResultBuilder result(envelope_schema());
    if (statistics) {
        result.set_binary("result", serialize_column_statistics(*statistics));
    } else {
        result.set_null("result");
    }
    return vgi_rpc::Result::value(result.finish());
}

vgi_rpc::Result Dispatcher::table_function_dynamic_to_string(const vgi_rpc::Request& request) {
    auto profile_request = wire::get_ipc(request.batch(), "request");
    if (!profile_request) throw std::runtime_error("dynamic_to_string: empty request");

    auto bind_call = wire::get_ipc(profile_request, "bind_call");
    if (!bind_call) throw std::runtime_error("dynamic_to_string: request carries no bind_call");

    const auto function_name = wire::get_string(bind_call, "function_name");
    const auto bind_params = read_bind_request(bind_call);
    const auto execution_id =
        wire::get_optional_binary(profile_request, "global_execution_id").value_or(std::string{});

    std::vector<std::string> keys;
    std::vector<std::string> values;
    if (auto table = find_table(function_name, bind_params.schema_name, &bind_params)) {
        for (const auto& [key, value] : table->dynamic_to_string(execution_id, *default_storage())) {
            keys.push_back(key);
            values.push_back(value);
        }
    }

    auto payload = wire::ResultBuilder(payload_schema_of("table_function_dynamic_to_string"))
                       .set_string_list("keys", keys)
                       .set_string_list("values", values)
                       .fill_defaults()
                       .finish();
    return vgi_rpc::Result::value(
        wire::ResultBuilder(envelope_schema())
            .set_binary("result", wire::encode_ipc(payload))
            .finish());
}

vgi_rpc::Stream Dispatcher::init(const vgi_rpc::Request& request) {
    auto init_request = wire::get_ipc(request.batch(), "request");
    if (!init_request) throw std::runtime_error("init: empty request");

    auto bind_call = wire::get_ipc(init_request, "bind_call");
    if (!bind_call) throw std::runtime_error("init: request carries no bind_call");

    const auto function_name = wire::get_string(bind_call, "function_name");
    const auto bind_params = read_bind_request(bind_call);

    // The engine sends back the schema bind settled on rather than making the
    // worker re-derive it, so a function whose bind is expensive pays once.
    auto output_schema = wire::get_schema(init_request, "output_schema");
    if (!output_schema) throw std::runtime_error("init: request carries no output_schema");

    // Projection pushdown arrives as indices into the bound schema rather than
    // as a narrowed schema, so the narrowing has to happen here: a function
    // that declared `projection_pushdown` and then emitted every column would
    // have its columns read positionally against the engine's shorter list.
    if (const auto projection_ids = wire::get_int64_list(init_request, "projection_ids");
        !projection_ids.empty()) {
        arrow::FieldVector fields;
        fields.reserve(projection_ids.size());
        for (const auto id : projection_ids) {
            if (id < 0 || id >= output_schema->num_fields()) {
                throw std::runtime_error("init: projection id " + std::to_string(id) +
                                         " is outside the bound output schema");
            }
            fields.push_back(output_schema->field(static_cast<int>(id)));
        }
        output_schema = arrow::schema(std::move(fields));
    }

    ProcessParams params;
    params.output_schema = output_schema;
    // Constant arguments were evaluated at bind and ride along on every call,
    // which is why a const parameter never appears in the input batch.
    params.arguments = bind_params.arguments;
    params.copy_to_format = bind_params.copy_to_format;
    params.copy_to_path = bind_params.copy_to_path;
    params.copy_from_format = bind_params.copy_from_format;
    params.copy_from_path = bind_params.copy_from_path;
    params.settings = bind_params.settings;
    params.secrets = bind_params.secrets;
    params.catalog_name = catalog_.name;
    params.schema_name = bind_params.schema_name;
    params.storage = default_storage();

    // The engine may supply the execution id (it does for a follow-on worker,
    // and for the buffering finalize phase); otherwise this is the primary
    // init and we mint one.
    auto execution_id =
        wire::get_optional_binary(init_request, "execution_id").value_or(std::string{});
    const bool primary = execution_id.empty();
    if (primary) execution_id = next_execution_id();
    params.execution_id = execution_id;

    // Optimizer hints, all absent unless the plan shape let DuckDB fold the
    // clause into the scan. The two enum fields are dictionary-encoded, so
    // reading them as plain strings would fail the whole init.
    params.scan_hints.order_by_column =
        wire::get_optional_string(init_request, "order_by_column_name");
    params.scan_hints.order_by_direction =
        wire::get_optional_enum(init_request, "order_by_direction");
    params.scan_hints.order_by_null_order =
        wire::get_optional_enum(init_request, "order_by_null_order");
    params.scan_hints.order_by_limit = wire::get_optional_int64(init_request, "order_by_limit");
    params.scan_hints.tablesample_percentage =
        wire::get_optional_double(init_request, "tablesample_percentage");
    params.scan_hints.tablesample_seed =
        wire::get_optional_int64(init_request, "tablesample_seed");

    // Parsed once for the whole scan; the engine sends it on init.
    params.pushdown_filters = PushdownFilters::parse(
        wire::get_optional_binary(init_request, "pushdown_filters").value_or(std::string{}),
        wire::get_binary_list(init_request, "join_keys"));

    int64_t max_workers = 1;
    if (auto table = find_table(function_name, params.schema_name, &bind_params)) {
        max_workers = std::max<int64_t>(1, table->max_workers(params));
        // Only the primary init divides the work, and only once. A follow-on
        // worker calling this too would push the same chunks again.
        if (primary) table->on_init(params);
    }

    auto header = wire::ResultBuilder(global_init_response_schema())
                      .set_binary("execution_id", execution_id)
                      .set_int64("max_workers", max_workers)
                      .set_null("opaque_data")
                      .finish();

    vgi_rpc::Stream stream;
    stream.output_schema = output_schema;
    stream.header = std::move(header);

    // A table function is a *producer* — it generates rows and reads none — so
    // its stream has no input schema. A scalar function is an *exchange*: one
    // output batch per input batch.
    if (auto sink = find_buffering(function_name, params.schema_name)) {
        // Stash the bind-time context for the buffering RPCs.
        //
        // `table_buffering_process` and `_combine` carry no arguments,
        // settings or COPY destination of their own — those were settled at
        // bind — and they may land on a worker that never ran it. Persisting
        // them here, scoped by execution id, is the only channel between the
        // two. Without it a COPY writer sees none of its options.
        if (primary) {
            const auto stash = [&](const char* key, const std::string& value) {
                if (!value.empty()) params.storage->kv_put(execution_id, key, value);
            };
            stash("bind.arguments",
                  wire::get_optional_binary(bind_call, "arguments").value_or(""));
            stash("bind.settings",
                  wire::get_optional_binary(bind_call, "settings").value_or(""));
            stash("bind.secrets",
                  wire::get_optional_binary(bind_call, "secrets").value_or(""));
            stash("bind.copy_to_format", params.copy_to_format.value_or(""));
            stash("bind.copy_to_path", params.copy_to_path.value_or(""));
            stash("bind.schema", wire::encode_schema(output_schema));
        }
        // Two shapes behind one name. The sink phase is header-only — the
        // engine ships batches through table_buffering_process, not through
        // this stream — while the finalize phase drains one state id as an
        // ordinary producer. The phase says which, and defaults to the sink.
        // `phase` is a dictionary-encoded enum, not a plain string — reading
        // it as one fails the whole call rather than defaulting.
        const auto phase = wire::get_optional_enum(init_request, "phase").value_or("");
        if (phase == enums::phase::kTableBufferingFinalize) {
            const auto state_id =
                wire::get_optional_binary(init_request, "finalize_state_id").value_or("");
            stream.input_schema = arrow::schema({});
            stream.state = std::make_shared<TableProduce>(
                sink->finalize_producer(params, state_id));
            return stream;
        }
        // Header-only: an empty producer that finishes on its first tick.
        stream.input_schema = arrow::schema({});
        stream.state = std::make_shared<TableProduce>(nullptr);
        return stream;
    }

    if (auto table = find_table(function_name, params.schema_name, &bind_params)) {
        stream.input_schema = arrow::schema({});
        auto auto_apply = table->metadata().auto_apply_filters ? params.pushdown_filters
                                                              : PushdownFilters{};
        stream.state =
            std::make_shared<TableProduce>(table->init(params), std::move(auto_apply));
        return stream;
    }

    auto input_schema_or_empty = [&] {
        auto schema = wire::get_schema(bind_call, "input_schema");
        return schema ? schema : arrow::schema({});
    };

    if (auto transform = find_table_in_out(function_name, params.schema_name)) {
        stream.input_schema = input_schema_or_empty();
        stream.state = std::make_shared<TableInOutExchange>(std::move(transform),
                                                            std::move(params));
        return stream;
    }

    auto fn = resolve_scalar(function_name, bind_params);
    stream.input_schema = input_schema_or_empty();
    stream.state = std::make_shared<ScalarExchange>(std::move(fn), std::move(params));
    return stream;
}

}  // namespace vgi
