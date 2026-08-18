// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
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
#include <regex>
#include <sstream>

#include <unistd.h>
#include <stdexcept>

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>
#include <nlohmann/json.hpp>
#include <arrow/array/util.h>
#include <arrow/table.h>
#include <arrow/util/base64.h>
#include <arrow/util/key_value_metadata.h>
#include <vgi_rpc/request.h>
#include <vgi_rpc/result.h>
#include <vgi_rpc/stream.h>

#include "dispatcher.h"
#include "arg_schema.h"
#include "enums.h"
#include "methods.h"
#include "vgi/storage.h"

#include "split_token.h"
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

namespace {}  // namespace

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

// The per-batch and per-tick metadata keys this dispatcher reads and writes.
//
// Spelled here rather than at each use because a typo in one of them is
// silent: an unrecognised key is simply ignored by the other side.
namespace keys {
// Per-output-row provenance: base64 of a raw little-endian int32 array with
// one entry per emitted row, naming the input row that produced it. Raw bytes
// rather than Arrow IPC — the engine reinterprets them directly.
constexpr const char* kParentRow = "vgi_rpc.parent_row#b64";
// Conditional-request validators the engine sends when it holds a cached
// answer it would like revalidated instead of recomputed.
constexpr const char* kIfNoneMatch = "vgi.cache.if_none_match";
constexpr const char* kIfModifiedSince = "vgi.cache.if_modified_since";
// Filters the engine only learned after the scan began — a join's build side,
// a Top-N threshold — base64 of the same IPC blob the init request carries.
constexpr const char* kDynamicFilters = "vgi_pushdown_filters";
}  // namespace keys

std::optional<std::string> metadata_value(
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, const char* key) {
    if (!metadata) return std::nullopt;
    const auto index = metadata->FindKey(key);
    if (index < 0) return std::nullopt;
    return metadata->value(index);
}

std::optional<std::string> request_metadata(const vgi_rpc::Request& request, const char* key) {
    return metadata_value(request.metadata(), key);
}

std::string encode_parent_rows(const std::vector<int32_t>& parent_rows) {
    std::string raw(parent_rows.size() * sizeof(int32_t), '\0');
    for (size_t i = 0; i < parent_rows.size(); ++i) {
        // Little-endian by hand: the wire fixes the byte order, and the host's
        // is only incidentally the same.
        const auto value = static_cast<uint32_t>(parent_rows[i]);
        raw[i * 4 + 0] = static_cast<char>(value & 0xFF);
        raw[i * 4 + 1] = static_cast<char>((value >> 8) & 0xFF);
        raw[i * 4 + 2] = static_cast<char>((value >> 16) & 0xFF);
        raw[i * 4 + 3] = static_cast<char>((value >> 24) & 0xFF);
    }
    return arrow::util::base64_encode(raw);
}

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
        auto params = params_;
        params.client_log = [&out](LogLevel level, const std::string& message) {
            out.client_log(to_rpc_level(level), message);
        };
        auto result = fn_->process(params, input.batch);
        if (!result) {
            throw std::runtime_error("scalar function '" + fn_->name() + "' returned no batch");
        }
        // The engine matches results to inputs positionally, so a row-count
        // mismatch is a silent misalignment rather than an error it can see.
        if (input.batch && result->num_rows() != input.batch->num_rows()) {
            throw std::runtime_error("scalar function '" + fn_->name() + "' returned " +
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
// Whether `value` appears in `choices`, which is declared as a JSON array.
//
// A `choices` that is not a JSON array is the declaration's bug; accepting
// everything is the safer reading of it than refusing everything.
bool json_array_contains(const std::string& choices, const std::string& value) {
    try {
        const auto parsed = nlohmann::json::parse(choices);
        if (!parsed.is_array()) return true;
        for (const auto& choice : parsed) {
            if (choice.is_string() ? choice.get<std::string>() == value : choice.dump() == value) {
                return true;
            }
        }
        return false;
    } catch (const nlohmann::json::exception&) {
        return true;
    }
}

int score_overload(const std::vector<ArgSpec>& specs,
                   const std::function<std::shared_ptr<arrow::DataType>(size_t)>& actual_type) {
    int score = 0;
    for (const auto& spec : specs) {
        // `spec.index` is the call position, which is not the position in this
        // vector: specs may be listed in any order, and a named-only one has
        // no call position at all.
        if (!spec.index || *spec.index < 0) continue;
        // An exact Arrow type wins, but a VGI type *name* is just as much a
        // declaration — `constant_arg(..., "varchar", ...)` names utf8 and has
        // to take part in resolution, or two overloads that differ only by
        // their named types are indistinguishable and the first always wins.
        auto declared = spec.arrow_type;
        if (!declared) declared = arg_type_to_arrow(spec.type);
        if (!declared || declared->id() == arrow::Type::NA) continue;
        auto actual = actual_type(static_cast<size_t>(*spec.index));
        if (!actual) continue;
        if (declared->Equals(*actual)) {
            ++score;
        } else {
            return -1;
        }
    }
    return score;
}

// Drains a prepared list of batches, for the phases that compute their whole
// answer before the stream starts.
class BatchesProducer : public TableProducer {
public:
    explicit BatchesProducer(std::vector<std::shared_ptr<arrow::RecordBatch>> batches)
        : batches_(std::move(batches)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (next_ >= batches_.size()) return nullptr;
        return batches_[next_++];
    }

private:
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
    size_t next_ = 0;
};

// Narrow `batch` to `schema` by name, for the producer that ignored the
// projection and emitted every column.
//
// Reading the wide batch positionally against the engine's shorter list is
// exactly what makes an all-NULL column come back non-NULL, so the columns are
// matched by name; anything that does not line up is left alone, and the
// emit-time schema check reports it.
std::shared_ptr<arrow::RecordBatch> narrow_to(const std::shared_ptr<arrow::RecordBatch>& batch,
                                              const std::shared_ptr<arrow::Schema>& schema) {
    if (!schema || batch->schema()->Equals(*schema)) return batch;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(static_cast<size_t>(schema->num_fields()));
    for (const auto& field : schema->fields()) {
        auto column = batch->GetColumnByName(field->name());
        if (!column || !column->type()->Equals(*field->type())) return batch;
        columns.push_back(std::move(column));
    }
    return arrow::RecordBatch::Make(schema, batch->num_rows(), columns);
}

class TableProduce : public vgi_rpc::ProducerState {
public:
    TableProduce(std::unique_ptr<TableProducer> producer, PushdownFilters filters = {},
                 std::shared_ptr<arrow::Schema> output_schema = nullptr)
        : producer_(std::move(producer)),
          filters_(std::move(filters)),
          output_schema_(std::move(output_schema)) {}

    // Conditional-request validators from the init request, which is where
    // they arrive for a producer: over HTTP the first tick is folded into the
    // init, so waiting for a tick would miss them entirely.
    void set_validators(std::optional<std::string> if_none_match,
                        std::optional<std::string> if_modified_since) {
        if_none_match_ = std::move(if_none_match);
        if_modified_since_ = std::move(if_modified_since);
    }

    // Overriding `process` rather than `produce`: the validators ride the
    // first tick's metadata, and `produce` never sees the tick. Over HTTP the
    // first tick is folded into the init, which is why the init's metadata is
    // the fallback.
    void process(const vgi_rpc::AnnotatedBatch& input, vgi_rpc::OutputCollector& out,
                 vgi_rpc::CallContext& context) override {
        if (!asked_) {
            if (auto etag = metadata_value(input.custom_metadata, keys::kIfNoneMatch)) {
                if_none_match_ = std::move(etag);
            }
            if (auto since = metadata_value(input.custom_metadata, keys::kIfModifiedSince)) {
                if_modified_since_ = std::move(since);
            }
        }
        // Bound before anything that can log, not only in `produce`: the hook
        // below runs first, and a producer holding a collector from an earlier
        // tick would write into one that has already been destroyed.
        bind_log(out);
        // Re-read every tick, not only the first: the engine re-sends them as
        // the join's build side fills in, and the last one is the tightest.
        //
        // Kept per-tick rather than folded into `filters_`: they describe this
        // tick's build side, and a later tick that carries none must fall back
        // to the static predicates rather than keep stale dynamic ones.
        dynamic_filters_.reset();
        if (auto encoded = metadata_value(input.custom_metadata, keys::kDynamicFilters)) {
            auto decoded = arrow::util::base64_decode(*encoded);
            dynamic_filters_ = PushdownFilters::parse(decoded);
            if (producer_) producer_->on_dynamic_filters(*dynamic_filters_);
        }
        produce(out, context);
    }

    void produce(vgi_rpc::OutputCollector& out, vgi_rpc::CallContext&) override {
        // A null producer is the buffering sink phase, which emits nothing:
        // its batches arrive through table_buffering_process instead.
        if (!producer_) {
            out.finish();
            return;
        }
        bind_log(out);
        if (!asked_) {
            asked_ = true;
            producer_->on_conditional_request(if_none_match_, if_modified_since_);
        }
        auto batch = producer_->next_batch();
        if (!batch) {
            out.finish();
            return;
        }
        // Applied here rather than in the producer so a function that only
        // advertises the capability gets it for free, and one that uses the
        // filters itself is not filtered twice. This tick's dynamic filters
        // supersede the static ones when it carried any — they are the
        // tighter predicate, derived from the same scan.
        const PushdownFilters& active = dynamic_filters_ ? *dynamic_filters_ : filters_;
        if (!active.empty()) batch = active.apply(batch);

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
        batch = narrow_to(batch, output_schema_);
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
    // Bound per tick, since the collector is created per tick: a producer
    // holding one from an earlier tick would write into a dead sink.
    void bind_log(vgi_rpc::OutputCollector& out) {
        if (!producer_) return;
        producer_->set_log([&out](LogLevel level, const std::string& message) {
            out.client_log(to_rpc_level(level), message);
        });
    }

    std::unique_ptr<TableProducer> producer_;
    PushdownFilters filters_;
    // This tick's join-side predicate, if it carried one.
    std::optional<PushdownFilters> dynamic_filters_;
    // What the engine asked for, which a producer that ignores the projection
    // does not emit.
    std::shared_ptr<arrow::Schema> output_schema_;
    std::optional<std::string> if_none_match_;
    std::optional<std::string> if_modified_since_;
    // Asked once, before the first batch: a producer that answered
    // `not_modified` has nothing more to decide.
    bool asked_ = false;
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
        // Conditional-request validators ride the tick, not the init: an
        // exchange is asked once per input chunk and the engine may hold a
        // cached answer for some of them and not others.
        auto params = params_;
        // Bound per tick, since the collector is created per tick: a function
        // holding one from an earlier tick would write into a dead sink.
        params.client_log = [&out](LogLevel level, const std::string& message) {
            out.client_log(to_rpc_level(level), message);
        };
        params.if_none_match = tick_metadata(input, keys::kIfNoneMatch);
        params.if_modified_since = tick_metadata(input, keys::kIfModifiedSince);

        emit_merged(fn_->process(params, input.batch), out);
    }

private:
    // One data batch per exchange tick is all the wire carries, so several
    // emitted batches are concatenated — and their provenance with them, since
    // an output row's parent index is relative to the one input batch either
    // way.
    void emit_merged(const std::vector<EmittedBatch>& emitted,
                     vgi_rpc::OutputCollector& out) const {
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        std::vector<int32_t> parent_rows;
        bool have_parents = false;
        std::map<std::string, std::string> metadata;

        for (const auto& item : emitted) {
            if (!item.batch) continue;
            if (item.parent_rows) {
                if (item.parent_rows->size() != static_cast<size_t>(item.batch->num_rows())) {
                    throw std::runtime_error(
                        "table-in-out '" + fn_->name() + "' emitted " +
                        std::to_string(item.parent_rows->size()) + " parent rows for " +
                        std::to_string(item.batch->num_rows()) + " output rows");
                }
                have_parents = true;
                parent_rows.insert(parent_rows.end(), item.parent_rows->begin(),
                                   item.parent_rows->end());
            } else {
                // An unannotated batch in an annotated set would leave a hole
                // in the map, so it takes the identity indices it implies.
                for (int64_t i = 0; i < item.batch->num_rows(); ++i) {
                    parent_rows.push_back(static_cast<int32_t>(i));
                }
            }
            if (item.cache_control) {
                for (const auto& [key, value] : item.cache_control->to_metadata()) {
                    metadata[key] = value;
                }
            }
            for (const auto& [key, value] : item.metadata) metadata[key] = value;
            batches.push_back(item.batch);
        }
        // Exactly one data batch per turn, always: an exchange tick that
        // answered with nothing would leave the caller waiting on a reply that
        // never comes. A function that emits nothing this tick — one that
        // accumulates and flushes at finalize — answers with no rows.
        if (batches.empty()) {
            out.emit_batch(make_empty(params_.output_schema));
            return;
        }

        auto batch = batches.size() == 1 ? batches.front() : concatenate(batches);
        if (have_parents) metadata[keys::kParentRow] = encode_parent_rows(parent_rows);

        if (metadata.empty()) {
            if (cache_) {
                out.emit_batch(batch, cache_);
            } else {
                out.emit_batch(batch);
            }
            return;
        }
        // The function-wide advertisement first, so a per-batch one overrides it.
        std::vector<std::string> merged_keys;
        std::vector<std::string> merged_values;
        if (cache_) {
            for (int i = 0; i < cache_->size(); ++i) {
                if (!metadata.count(cache_->key(i))) {
                    merged_keys.push_back(cache_->key(i));
                    merged_values.push_back(cache_->value(i));
                }
            }
        }
        for (const auto& [key, value] : metadata) {
            merged_keys.push_back(key);
            merged_values.push_back(value);
        }
        out.emit_batch(batch, arrow::key_value_metadata(merged_keys, merged_values));
    }

    static std::shared_ptr<arrow::RecordBatch> make_empty(
        const std::shared_ptr<arrow::Schema>& schema) {
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(schema->num_fields()));
        for (const auto& field : schema->fields()) {
            auto column = arrow::MakeArrayOfNull(field->type(), 0);
            if (!column.ok())
                throw std::runtime_error("table-in-out: " + column.status().ToString());
            columns.push_back(column.MoveValueUnsafe());
        }
        return arrow::RecordBatch::Make(schema, 0, columns);
    }

    static std::shared_ptr<arrow::RecordBatch> concatenate(
        const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches) {
        auto table = arrow::Table::FromRecordBatches(batches);
        if (!table.ok()) {
            throw std::runtime_error("table-in-out: " + table.status().ToString());
        }
        auto combined = (*table)->CombineChunksToBatch();
        if (!combined.ok()) {
            throw std::runtime_error("table-in-out: " + combined.status().ToString());
        }
        return combined.MoveValueUnsafe();
    }

    static std::optional<std::string> tick_metadata(const vgi_rpc::AnnotatedBatch& input,
                                                    const char* key) {
        if (!input.custom_metadata) return std::nullopt;
        const auto index = input.custom_metadata->FindKey(key);
        if (index < 0) return std::nullopt;
        return input.custom_metadata->value(index);
    }

    std::shared_ptr<TableInOutFunction> fn_;
    ProcessParams params_;
    std::shared_ptr<arrow::KeyValueMetadata> cache_;
};

}  // namespace

// A registration belongs to a scope when both halves match. An empty schema
// means "any", which is how a call that names no schema still resolves; the
// catalog is never optional, because two catalogs may declare the same name in
// the same schema and only the attachment distinguishes them.
namespace {

bool in_scope(const Dispatcher::Scope& declared, const Dispatcher::Scope& wanted) {
    if (declared.catalog != wanted.catalog) return false;
    return wanted.schema.empty() || declared.schema == wanted.schema;
}

}  // namespace

std::vector<std::shared_ptr<TableBufferingFunction>> Dispatcher::bufferings_in_schema(
    const Scope& scope) const {
    std::vector<std::shared_ptr<TableBufferingFunction>> found;
    for (size_t i = 0; i < bufferings_.size(); ++i) {
        if (in_scope(buffering_scopes_[i], scope)) found.push_back(bufferings_[i]);
    }
    return found;
}

std::shared_ptr<TableBufferingFunction> Dispatcher::find_buffering(const std::string& name,
                                                                   const Scope& scope) const {
    auto it = buffering_by_name_.find(name);
    if (it == buffering_by_name_.end()) return nullptr;
    for (size_t index : it->second) {
        if (in_scope(buffering_scopes_[index], scope)) return bufferings_[index];
    }
    // No cross-schema fallback — see find_table.
    return nullptr;
}

std::shared_ptr<TableBufferingFunction> Dispatcher::require_buffering(const std::string& name,
                                                                      const Scope& scope) const {
    if (auto fn = find_buffering(name, scope)) return fn;
    throw std::invalid_argument("no buffering function named '" + name + "'");
}

std::vector<std::shared_ptr<AggregateFunction>> Dispatcher::aggregates_in_schema(
    const Scope& scope) const {
    std::vector<std::shared_ptr<AggregateFunction>> found;
    for (size_t i = 0; i < aggregates_.size(); ++i) {
        if (in_scope(aggregate_scopes_[i], scope)) found.push_back(aggregates_[i]);
    }
    return found;
}

std::shared_ptr<AggregateFunction> Dispatcher::require_aggregate(const std::string& name,
                                                                 const Scope& scope) const {
    auto it = aggregate_by_name_.find(name);
    if (it != aggregate_by_name_.end()) {
        for (size_t index : it->second) {
            if (in_scope(aggregate_scopes_[index], scope)) return aggregates_[index];
        }
    }
    // Naming the schema, because "no aggregate named x" when x exists in
    // another schema sends the reader looking in the wrong place.
    throw std::invalid_argument("no aggregate function named '" + name + "' in schema '" +
                                scope.schema + "'");
}

std::vector<std::shared_ptr<TableInOutFunction>> Dispatcher::table_in_outs_in_schema(
    const Scope& scope) const {
    std::vector<std::shared_ptr<TableInOutFunction>> found;
    for (size_t i = 0; i < table_in_outs_.size(); ++i) {
        if (in_scope(table_in_out_scopes_[i], scope)) found.push_back(table_in_outs_[i]);
    }
    return found;
}

std::shared_ptr<TableInOutFunction> Dispatcher::find_table_in_out(const std::string& name,
                                                                  const Scope& scope) const {
    auto it = table_in_out_by_name_.find(name);
    if (it == table_in_out_by_name_.end()) return nullptr;
    for (size_t index : it->second) {
        if (in_scope(table_in_out_scopes_[index], scope)) return table_in_outs_[index];
    }
    // No cross-schema fallback — see find_table.
    return nullptr;
}

std::vector<std::shared_ptr<TableFunction>> Dispatcher::tables_in_schema(const Scope& scope) const {
    std::vector<std::shared_ptr<TableFunction>> found;
    for (size_t i = 0; i < tables_.size(); ++i) {
        if (in_scope(table_scopes_[i], scope)) found.push_back(tables_[i]);
    }
    return found;
}

std::shared_ptr<TableFunction> Dispatcher::find_table(const std::string& name,
                                                      const Scope& scope) const {
    return find_table(name, scope, nullptr);
}

std::shared_ptr<TableFunction> Dispatcher::find_table(const std::string& name, const Scope& scope,
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
        if (in_scope(table_scopes_[index], scope)) candidates.push_back(tables_[index]);
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
    const Scope& scope) const {
    std::vector<std::shared_ptr<ScalarFunction>> found;
    for (size_t i = 0; i < scalars_.size(); ++i) {
        if (in_scope(scalar_scopes_[i], scope)) found.push_back(scalars_[i]);
    }
    return found;
}

std::vector<std::shared_ptr<ScalarFunction>> Dispatcher::scalars_named(const std::string& name,
                                                                       const Scope& scope) const {
    std::vector<std::shared_ptr<ScalarFunction>> found;
    auto it = scalar_by_name_.find(name);
    if (it == scalar_by_name_.end()) return found;
    found.reserve(it->second.size());
    for (size_t index : it->second) {
        if (in_scope(scalar_scopes_[index], scope)) found.push_back(scalars_[index]);
    }
    return found;
}

std::shared_ptr<ScalarFunction> Dispatcher::resolve_scalar(const std::string& name,
                                                           const BindParams& params) const {
    auto candidates = scalars_named(name, scope_of(params));
    if (candidates.empty()) {
        throw std::invalid_argument("no scalar function named '" + name + "'");
    }

    if (candidates.size() == 1) return candidates.front();

    std::shared_ptr<ScalarFunction> best;
    int best_score = -1;
    for (const auto& candidate : candidates) {
        const int score = score_overload(candidate->argument_specs(),
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
    for (const auto& spec : fn.argument_specs()) {
        if (!spec.type_bound || !spec.type_bound->accepts) continue;
        // Indexed by the declared call position, not by where the spec happens
        // to sit in the vector.
        if (!spec.index || *spec.index < 0) continue;
        auto type = params.input_type(static_cast<size_t>(*spec.index));
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
    for (const auto& spec : specs) {
        // `spec.index` is the call position. Using the loop position instead
        // checks each constraint against whichever argument happens to sit at
        // the same offset in the spec vector, which is only the same argument
        // when the specs were declared in index order.
        const auto position = spec.index && *spec.index >= 0 ? static_cast<size_t>(*spec.index) : 0;

        // A named parameter the declaration marked required has to be there.
        // Positional ones are the engine's to supply and it always does.
        if (!spec.index && spec.required && !arguments.named(spec.name)) {
            throw std::invalid_argument("function '" + function_name + "' requires argument '" +
                                        spec.name + "'");
        }

        // An explicit SQL NULL for a constant is a caller error, and one worth
        // naming here: left to the function it becomes an absent value, which
        // silently takes the default instead of failing.
        const bool supplied_null = spec.index ? arguments.positional_is_null(position)
                                              : arguments.named_is_null(spec.name);
        // Named-only parameters are read as constants too, even though they
        // carry no `vgi_const` marker; a column argument is exempt, since a
        // column may legitimately be null in some rows. So is a varargs
        // parameter: it stands for a whole tail of arguments, of which a NULL
        // is an ordinary member — `constant_columns(3, NULL::INTEGER)` asks
        // for a NULL column, not for the default.
        if ((spec.constant || !spec.index) && !spec.varargs && supplied_null) {
            throw std::invalid_argument("function '" + function_name + "' argument '" + spec.name +
                                        "' cannot be NULL");
        }

        // Only a constant can be checked here: a column's values are not known
        // until process(), and the engine re-checks nothing on our behalf.
        if (!spec.constant && spec.index) continue;

        const auto refuse = [&](const std::string& bound) {
            throw std::invalid_argument("function '" + function_name + "' argument '" + spec.name +
                                        "' must be " + bound);
        };

        if (spec.ge || spec.le || spec.gt || spec.lt) {
            const auto value =
                spec.index ? arguments.const_double(position) : arguments.named_double(spec.name);
            if (value) {
                if (spec.ge && *value < *spec.ge) refuse(">= " + std::to_string(*spec.ge));
                if (spec.gt && *value <= *spec.gt) refuse("> " + std::to_string(*spec.gt));
                if (spec.le && *value > *spec.le) refuse("<= " + std::to_string(*spec.le));
                if (spec.lt && *value >= *spec.lt) refuse("< " + std::to_string(*spec.lt));
            }
        }

        // `choices` and `pattern` are enforced, not merely advertised — both
        // references validate them, and a worker that only shows them in
        // discovery accepts calls the reference refuses.
        if (spec.choices || spec.pattern) {
            const auto text =
                spec.index ? arguments.const_string(position) : arguments.named_string(spec.name);
            if (text) {
                if (spec.choices && !json_array_contains(*spec.choices, *text)) {
                    refuse("one of " + *spec.choices);
                }
                if (spec.pattern) {
                    try {
                        if (!std::regex_search(*text, std::regex(*spec.pattern))) {
                            refuse("a match for /" + *spec.pattern + "/");
                        }
                    } catch (const std::regex_error&) {
                        // An unparseable pattern is the declaration's bug, not
                        // the caller's; refusing every call would be worse.
                    }
                }
            }
        }
    }
}

// The (catalog, schema) a bind or a call belongs to. Both halves are needed:
// two catalogs may declare the same function in the same schema, and only the
// attachment says which one this call meant.
Dispatcher::Scope Dispatcher::scope_of(const BindParams& params) {
    return {params.catalog_name, params.schema_name};
}

Dispatcher::Scope Dispatcher::scope_of(const ProcessParams& params) {
    return {params.catalog_name, params.schema_name};
}

// The secrets `name` declares, whichever registry it lives in.
//
// Answered only on the *first* bind: once the engine has resolved them it sets
// `resolved_secrets_provided`, and asking again would loop.
std::vector<SecretLookup> Dispatcher::required_secrets_of(const std::string& name,
                                                          const BindParams& params) const {
    if (params.secrets_resolved) return {};
    if (auto fn = find_buffering(name, scope_of(params))) {
        // A COPY writer scopes its lookup to the destination path, which only
        // the bind params carry — so ask it rather than reading a static list.
        for (const auto& writer : copy_to_) {
            if (writer->handler_name() == name) return writer->secret_lookups(params);
        }
        return fn->metadata().required_secrets;
    }
    if (auto fn = find_table_in_out(name, scope_of(params))) {
        return fn->secret_lookups(params);
    }
    if (auto fn = find_table(name, scope_of(params), &params)) {
        return fn->secret_lookups(params);
    }
    auto candidates = scalars_named(name, scope_of(params));
    if (!candidates.empty()) return candidates.front()->secret_lookups(params);
    return {};
}

// The argument specs `name` declares, whichever registry it lives in.
std::vector<ArgSpec> Dispatcher::argument_specs_of(const std::string& name,
                                                   const BindParams& params) const {
    if (auto fn = find_buffering(name, scope_of(params))) return fn->argument_specs();
    if (auto fn = find_table_in_out(name, scope_of(params))) return fn->argument_specs();
    if (auto fn = find_table(name, scope_of(params), &params)) return fn->argument_specs();
    auto candidates = scalars_named(name, scope_of(params));
    if (!candidates.empty()) return candidates.front()->argument_specs();
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
    params.secrets = Secrets::parse(wire::get_optional_binary(bind_call, "secrets").value_or(""));
    params.schema_name = wire::get_optional_string(bind_call, "schema_name").value_or("main");
    // From the attachment's seal, not from the primary: one worker may serve
    // several catalogs, and a bind carries nothing else that says which one.
    const auto attachment = attachment_of(bind_call);
    params.catalog_name = attachment.catalog;
    params.attachment_id = attachment.id;
    params.attach_options = attachment.options;
    params.transaction_opaque_data =
        wire::get_optional_binary(bind_call, "transaction_opaque_data");
    // Empty strings mean "no clause" on this wire, which is not the same as a
    // clause whose value happens to be empty.
    params.at_unit = wire::get_optional_string(bind_call, "at_unit");
    if (params.at_unit && params.at_unit->empty()) params.at_unit.reset();
    params.at_value = wire::get_optional_string(bind_call, "at_value");
    if (params.at_value && params.at_value->empty()) params.at_value.reset();
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
    params.arguments.remap_positional(
        argument_specs_of(wire::get_string(bind_call, "function_name"), params));
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
    // A COPY format's reader is a table function and its writer a buffering
    // one, and they deliberately share a name. Probing buffering first would
    // hand the writer to a COPY FROM; probing tables first would hand the
    // reader to a COPY TO. The declared kind is what breaks the tie.
    const bool prefers_table = declared_kind == enums::function_type::kTable;
    std::shared_ptr<arrow::Schema> output_schema;
    std::shared_ptr<TableBufferingFunction> sink;
    if (!prefers_table) sink = find_buffering(function_name, scope_of(params));
    if (sink) {
        output_schema = sink->bind(params);
        check_arg_constraints(function_name, sink->argument_specs(), params.arguments);
    } else if (auto transform = find_table_in_out(function_name, scope_of(params))) {
        check_arg_constraints(function_name, transform->argument_specs(), params.arguments);
        output_schema = transform->bind(params);
    } else if (auto table = find_table(function_name, scope_of(params), &params)) {
        check_arg_constraints(function_name, table->argument_specs(), params.arguments);
        output_schema = table->bind(params);
    } else if (auto late = find_buffering(function_name, scope_of(params))) {
        // Declared `table` but no table function of that name: the declaration
        // is a hint, not a guarantee.
        output_schema = late->bind(params);
        check_arg_constraints(function_name, late->argument_specs(), params.arguments);
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

// One serialized `ScanSplit` per unit of work.
//
// The engine reads `token` and nothing else — the payload rides *inside* the
// envelope, where it is verifiable, and a split record without a token is
// rejected as a worker that bypassed the framework. The rest of the shape is
// emitted anyway so a reader of these bytes sees the documented record rather
// than a private subset.
namespace {

const std::shared_ptr<arrow::Schema>& scan_split_schema() {
    static const auto schema = arrow::schema({
        arrow::field("payload", arrow::large_binary(), /*nullable=*/false),
        arrow::field("token", arrow::large_binary(), /*nullable=*/false),
        arrow::field("estimated_rows", arrow::int64(), /*nullable=*/true),
        arrow::field("rows_exact", arrow::boolean(), /*nullable=*/false),
        arrow::field("estimated_bytes", arrow::int64(), /*nullable=*/true),
    });
    return schema;
}

std::string encode_scan_split(const ScanSplit& split, const std::string& token) {
    arrow::LargeBinaryBuilder payload;
    arrow::LargeBinaryBuilder stamped;
    arrow::Int64Builder rows;
    arrow::BooleanBuilder exact;
    arrow::Int64Builder bytes;
    // The payload is carried as well as sealed into the token so the record is
    // self-describing to a human reading it; only the token is redeemable.
    (void)payload.Append(split.payload);
    (void)stamped.Append(token);
    if (split.estimated_rows) {
        (void)rows.Append(*split.estimated_rows);
    } else {
        (void)rows.AppendNull();
    }
    (void)exact.Append(split.rows_exact);
    if (split.estimated_bytes) {
        (void)bytes.Append(*split.estimated_bytes);
    } else {
        (void)bytes.AppendNull();
    }
    std::vector<std::shared_ptr<arrow::Array>> columns(5);
    if (!payload.Finish(&columns[0]).ok() || !stamped.Finish(&columns[1]).ok() ||
        !rows.Finish(&columns[2]).ok() || !exact.Finish(&columns[3]).ok() ||
        !bytes.Finish(&columns[4]).ok()) {
        throw std::runtime_error("plan: could not build a ScanSplit record");
    }
    return wire::encode_ipc(arrow::RecordBatch::Make(scan_split_schema(), 1, columns));
}

}  // namespace

vgi_rpc::Result Dispatcher::table_function_plan(const vgi_rpc::Request& request) {
    auto plan_request = wire::get_ipc(request.batch(), "request");
    if (!plan_request) throw std::runtime_error("plan: empty request");

    auto bind_call = wire::get_ipc(plan_request, "bind_call");
    if (!bind_call) throw std::runtime_error("plan: request carries no bind_call");

    const auto function_name = wire::get_string(bind_call, "function_name");
    const auto bind_params = read_bind_request(bind_call);

    PlanResult result;
    auto table = find_table(function_name, scope_of(bind_params), &bind_params);
    if (table && table->supports_splits()) {
        PlanParams plan;
        plan.target_split_bytes = wire::get_optional_int64(plan_request, "target_split_bytes");
        plan.min_splits = wire::get_optional_int64(plan_request, "min_splits");
        plan.max_splits_per_response =
            wire::get_optional_int64(plan_request, "max_splits_per_response");
        plan.row_limit = wire::get_optional_int64(plan_request, "row_limit");
        plan.cursor = wire::get_optional_binary(plan_request, "cursor");
        plan.filters_complete =
            wire::get_optional_bool(plan_request, "filters_complete").value_or(true);
        plan.pushdown_filters = PushdownFilters::parse(
            wire::get_optional_binary(plan_request, "pushdown_filters").value_or(std::string{}),
            wire::get_binary_list(plan_request, "join_keys"));
        result = table->plan(bind_params, plan);
    } else {
        // The whole scan as one unit. This is what a function that has not
        // opted into splits means, and answering it here rather than refusing
        // is what lets every existing function keep working under a protocol
        // that now plans before it scans.
        result.splits.push_back(ScanSplit{});
    }

    // The envelope is stamped here, never by the function: an author cannot
    // then forget the anchor or mis-bind the fingerprint, and the format stays
    // a framework detail that can change without touching a worker.
    const auto fingerprint = split_token::bind_fingerprint(
        bind_params.schema_name, function_name,
        wire::get_optional_binary(bind_call, "arguments").value_or(std::string{}),
        wire::get_optional_binary(bind_call, "settings").value_or(std::string{}));
    const auto anchor = split_token::anchor_for(result.catalog_version);

    std::vector<std::string> splits;
    splits.reserve(result.splits.size());
    for (const auto& split : result.splits) {
        splits.push_back(
            encode_scan_split(split, split_token::build(split.payload, fingerprint, anchor)));
    }

    auto payload = wire::ResultBuilder(payload_schema_of("table_function_plan"));
    payload.set_binary_list("splits", splits)
        .set_binary_list("next_cursors", result.next_cursor
                                             ? std::vector<std::string>{*result.next_cursor}
                                             : std::vector<std::string>{})
        .set_string("scope", "");
    const auto optional_int = [&](const char* field, std::optional<int64_t> value) {
        if (value) {
            payload.set_int64(field, *value);
        } else {
            payload.set_null(field);
        }
    };
    optional_int("max_workers", result.max_workers);
    optional_int("estimated_total_splits", result.estimated_total_splits);
    optional_int("estimated_total_rows", result.estimated_total_rows);
    optional_int("estimated_total_bytes", result.estimated_total_bytes);
    optional_int("catalog_version", result.catalog_version);
    return vgi_rpc::Result::value(
        wire::ResultBuilder(envelope_schema())
            .set_binary("result", wire::encode_ipc(payload.fill_defaults().finish()))
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
    if (auto table = find_table(function_name, scope_of(bind_params), &bind_params)) {
        ProcessParams params;
        params.client_log = client_log_sink(nullptr);
        params.arguments = bind_params.arguments;
        params.settings = bind_params.settings;
        params.secrets = bind_params.secrets;
        params.catalog_name = bind_params.catalog_name;
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
    return vgi_rpc::Result::value(wire::ResultBuilder(envelope_schema())
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
    if (auto table = find_table(function_name, scope_of(bind_params), &bind_params)) {
        ProcessParams params;
        params.client_log = client_log_sink(nullptr);
        params.arguments = bind_params.arguments;
        params.settings = bind_params.settings;
        params.secrets = bind_params.secrets;
        params.catalog_name = bind_params.catalog_name;
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
    if (auto table = find_table(function_name, scope_of(bind_params), &bind_params)) {
        for (const auto& [key, value] :
             table->dynamic_to_string(execution_id, *default_storage())) {
            keys.push_back(key);
            values.push_back(value);
        }
    }

    auto payload = wire::ResultBuilder(payload_schema_of("table_function_dynamic_to_string"))
                       .set_string_list("keys", keys)
                       .set_string_list("values", values)
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
    params.catalog_name = bind_params.catalog_name;
    params.attachment_id = bind_params.attachment_id;
    params.attach_options = bind_params.attach_options;
    params.schema_name = bind_params.schema_name;
    params.at_unit = bind_params.at_unit;
    params.at_value = bind_params.at_value;
    params.transaction_opaque_data = bind_params.transaction_opaque_data;
    params.storage = default_storage();
    // Replaced per tick by the exchange and producer drivers, which are the
    // only calls with a channel; a no-op until then.
    params.client_log = client_log_sink(nullptr);

    // The engine may supply the execution id (it does for a follow-on worker,
    // and for the buffering finalize phase); otherwise this is the primary
    // init and we mint one.
    auto execution_id =
        wire::get_optional_binary(init_request, "execution_id").value_or(std::string{});
    const bool primary = execution_id.empty();
    if (primary) execution_id = next_execution_id();
    params.execution_id = execution_id;
    // Client-minted and stable across a substream's init, ticks and finalize,
    // which is what a per-substream finalize keys its accumulated rows on.
    params.substream_id =
        wire::get_optional_binary(init_request, "substream_id").value_or(std::string{});

    // Split redemption. Verified and stripped here rather than in the function,
    // so an unverified token can never be acted on and the envelope stays a
    // framework detail. An empty list is not a split init — that is the
    // ordinary scan path, and it must stay untouched.
    if (auto tokens = wire::get_binary_list(init_request, "split_tokens"); !tokens.empty()) {
        const auto fingerprint = split_token::bind_fingerprint(
            bind_params.schema_name, function_name,
            wire::get_optional_binary(bind_call, "arguments").value_or(std::string{}),
            wire::get_optional_binary(bind_call, "settings").value_or(std::string{}));
        // The anchor a plan sealed in. Nothing here time-travels, so the
        // current anchor is the one a plan would mint now.
        const auto anchor = split_token::anchor_for(std::nullopt);

        std::vector<std::string> payloads;
        payloads.reserve(tokens.size());
        for (const auto& token : tokens) {
            auto opened = split_token::open(token, fingerprint, anchor);
            if (!opened) {
                throw std::runtime_error(
                    "init: split token for '" + function_name +
                    "' is not redeemable here — it was minted for a different bind, or "
                    "against a snapshot this worker no longer serves");
            }
            payloads.push_back(std::move(*opened));
        }
        params.split_payloads = std::move(payloads);
    }

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
    params.scan_hints.tablesample_seed = wire::get_optional_int64(init_request, "tablesample_seed");

    // Parsed once for the whole scan; the engine sends it on init.
    params.pushdown_filters = PushdownFilters::parse(
        wire::get_optional_binary(init_request, "pushdown_filters").value_or(std::string{}),
        wire::get_binary_list(init_request, "join_keys"));

    int64_t max_workers = 1;
    // Same tie-break as `bind`: a COPY TO whose writer shares a name with a
    // reader must not divide the reader's work into the shared queue.
    const bool buffering = wire::get_optional_enum(bind_call, "function_type") ==
                               enums::function_type::kTableBuffering &&
                           find_buffering(function_name, scope_of(params)) != nullptr;
    if (auto table =
            buffering ? nullptr : find_table(function_name, scope_of(params), &bind_params)) {
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
    if (auto sink = find_buffering(function_name, scope_of(params))) {
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
            stash("bind.arguments", wire::get_optional_binary(bind_call, "arguments").value_or(""));
            stash("bind.settings", wire::get_optional_binary(bind_call, "settings").value_or(""));
            stash("bind.secrets", wire::get_optional_binary(bind_call, "secrets").value_or(""));
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
            stream.state =
                std::make_shared<TableProduce>(sink->finalize_producer(params, state_id));
            return stream;
        }
        // Header-only: an empty producer that finishes on its first tick.
        stream.input_schema = arrow::schema({});
        stream.state = std::make_shared<TableProduce>(nullptr);
        return stream;
    }

    if (auto table = find_table(function_name, scope_of(params), &bind_params)) {
        stream.input_schema = arrow::schema({});
        auto auto_apply =
            table->metadata().auto_apply_filters ? params.pushdown_filters : PushdownFilters{};
        auto produce = std::make_shared<TableProduce>(table->init(params), std::move(auto_apply),
                                                      output_schema);
        produce->set_validators(request_metadata(request, keys::kIfNoneMatch),
                                request_metadata(request, keys::kIfModifiedSince));
        stream.state = std::move(produce);
        return stream;
    }

    auto input_schema_or_empty = [&] {
        auto schema = wire::get_schema(bind_call, "input_schema");
        return schema ? schema : arrow::schema({});
    };

    if (auto transform = find_table_in_out(function_name, scope_of(params))) {
        // Two shapes behind one name, as for buffering: the FINALIZE phase is
        // a producer that drains what this substream's ticks accumulated, and
        // every other phase is the ordinary exchange.
        const auto phase = wire::get_optional_enum(init_request, "phase").value_or("");
        if (phase == enums::phase::kFinalize) {
            stream.input_schema = arrow::schema({});
            auto batches = transform->finish(params);
            std::vector<std::shared_ptr<arrow::RecordBatch>> rows;
            rows.reserve(batches.size());
            for (auto& emitted : batches) {
                if (emitted.batch) rows.push_back(std::move(emitted.batch));
            }
            stream.state =
                std::make_shared<TableProduce>(std::make_unique<BatchesProducer>(std::move(rows)));
            return stream;
        }
        stream.input_schema = input_schema_or_empty();
        stream.state =
            std::make_shared<TableInOutExchange>(std::move(transform), std::move(params));
        return stream;
    }

    auto fn = resolve_scalar(function_name, bind_params);
    stream.input_schema = input_schema_or_empty();
    stream.state = std::make_shared<ScalarExchange>(std::move(fn), std::move(params));
    return stream;
}

}  // namespace vgi
