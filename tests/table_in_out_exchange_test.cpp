// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// The turn rules of a table-in-out exchange, driven directly: one
// OutputCollector per turn, as the stream loop creates them, so each test sees
// exactly the reply one turn wrote. Continued output (`vgi.status` CONTINUE
// ticks and HAVE_MORE_OUTPUT replies) is strict, with no capability
// negotiation, so every refusal here is a client or function bug the
// integration suite would only see as a hung or garbled query.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <vgi_rpc/call_context.h>
#include <vgi_rpc/log_sink.h>
#include <vgi_rpc/metadata.h>
#include <vgi_rpc/output_collector.h>
#include <vgi_rpc/server.h>
#include <vgi_rpc/wire.h>

#include "dispatcher.h"
#include "vgi/table_in_out.h"

namespace {

using Catch::Matchers::ContainsSubstring;

constexpr const char* kStatus = "vgi.status";
constexpr const char* kContinue = "CONTINUE";
constexpr const char* kHaveMoreOutput = "HAVE_MORE_OUTPUT";
constexpr const char* kCacheTtl = "vgi.cache.ttl";

std::shared_ptr<arrow::Schema> schema() {
    return arrow::schema({arrow::field("n", arrow::int64())});
}

std::shared_ptr<arrow::RecordBatch> rows(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    REQUIRE(builder.AppendValues(values).ok());
    return arrow::RecordBatch::Make(schema(), static_cast<int64_t>(values.size()),
                                    {builder.Finish().ValueOrDie()});
}

vgi_rpc::AnnotatedBatch input(const std::vector<int64_t>& values) {
    return vgi_rpc::AnnotatedBatch::data(rows(values));
}

vgi_rpc::AnnotatedBatch tick(const std::vector<int64_t>& values = {}) {
    return vgi_rpc::AnnotatedBatch::with_metadata(
        rows(values), arrow::key_value_metadata({kStatus}, {kContinue}));
}

// What a fixture does on each call; set per test.
struct Script {
    // Pages per real input: process() answers the first, resume() the rest.
    int pages = 1;
    // Each page's rows are the input's, or none.
    bool empty_pages = false;
    // A per-batch cache advertisement on resume()'s reply.
    bool cache_on_resume = false;
    // A raw `vgi.cache.*` key on resume()'s reply.
    bool raw_cache_key_on_resume = false;
    // Log from initial_state().
    bool log_initial_state = false;
};

struct Paging {
    std::shared_ptr<arrow::RecordBatch> batch;
    int remaining = 0;
};

// A resumable function that pages each input `script.pages` times.
class Paged final : public vgi::ResumableTableInOutFunction<Paging> {
public:
    explicit Paged(Script script, bool advertise_cache = false)
        : script_(script), advertise_cache_(advertise_cache) {}

    std::string name() const override { return "paged"; }
    vgi::FunctionMetadata metadata() const override { return {}; }
    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::optional<vgi::CacheControl> cache_control() const override {
        if (!advertise_cache_) return std::nullopt;
        vgi::CacheControl cache;
        cache.ttl_seconds = 60;
        return cache;
    }

    Paging initial_state(const vgi::ProcessParams& params) const override {
        if (script_.log_initial_state) params.client_log(vgi::LogLevel::Info, "opening");
        return {};
    }

    std::vector<vgi::EmittedBatch> process(const vgi::ProcessParams&,
                                           const std::shared_ptr<arrow::RecordBatch>& batch,
                                           Paging& state) const override {
        state.batch = batch;
        state.remaining = script_.pages - 1;
        return {page(state)};
    }

    std::vector<vgi::EmittedBatch> resume(const vgi::ProcessParams&, Paging& state) const override {
        --state.remaining;
        auto reply = page(state);
        if (script_.cache_on_resume) {
            vgi::CacheControl cache;
            cache.ttl_seconds = 5;
            reply.cache_control = cache;
        }
        if (script_.raw_cache_key_on_resume) reply.metadata[kCacheTtl] = "5";
        return {reply};
    }

private:
    vgi::EmittedBatch page(const Paging& state) const {
        vgi::EmittedBatch reply;
        if (!script_.empty_pages) reply.batch = state.batch;
        reply.more = state.remaining > 0;
        return reply;
    }

    Script script_;
    bool advertise_cache_;
};

// A plain table-in-out that claims more output it has no way to resume.
class Unresumable final : public vgi::TableInOutFunction {
public:
    std::string name() const override { return "unresumable"; }
    vgi::FunctionMetadata metadata() const override { return {}; }
    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }
    std::vector<vgi::EmittedBatch> process(
        const vgi::ProcessParams&,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        vgi::EmittedBatch reply(batch);
        reply.more = true;
        return {reply};
    }
};

// One exchange stream and the context its turns run under.
class Exchange {
public:
    explicit Exchange(std::shared_ptr<vgi::TableInOutFunction> fn)
        : init_sink_(std::make_shared<vgi_rpc::LogSink>("server", "init")),
          context_(init_sink_, "server", "init") {
        vgi::ProcessParams params;
        params.output_schema = schema();
        state_ = vgi::make_table_in_out_exchange(std::move(fn), std::move(params), &context_);
    }

    // One turn's reply: the batches its collector holds (logs included).
    std::vector<vgi_rpc::AnnotatedBatch> turn(const vgi_rpc::AnnotatedBatch& in) {
        vgi_rpc::OutputCollector out(schema(), /*producer_mode=*/false);
        state_->exchange(in, out, context_);
        return out.batches();
    }

    // The one data batch a turn replied with.
    vgi_rpc::AnnotatedBatch reply(const vgi_rpc::AnnotatedBatch& in) {
        const auto batches = turn(in);
        std::vector<vgi_rpc::AnnotatedBatch> data;
        for (const auto& batch : batches) {
            if (batch.type() != vgi_rpc::BatchType::LOG) data.push_back(batch);
        }
        REQUIRE(data.size() == 1);
        return data.front();
    }

    // The messages logged on the init call's channel.
    std::vector<std::string> init_logs() {
        std::vector<std::string> messages;
        for (const auto& batch : init_sink_->flush(schema())) {
            const auto index = batch.custom_metadata->FindKey(vgi_rpc::keys::LOG_MESSAGE);
            if (index >= 0) messages.push_back(batch.custom_metadata->value(index));
        }
        return messages;
    }

private:
    std::shared_ptr<vgi_rpc::LogSink> init_sink_;
    vgi_rpc::CallContext context_;
    std::shared_ptr<vgi_rpc::ExchangeState> state_;
};

std::optional<std::string> meta(const vgi_rpc::AnnotatedBatch& batch, const std::string& key) {
    if (!batch.custom_metadata) return std::nullopt;
    const auto index = batch.custom_metadata->FindKey(key);
    if (index < 0) return std::nullopt;
    return batch.custom_metadata->value(index);
}

// Whether a reply carries any cache-control key at all.
bool has_cache_keys(const vgi_rpc::AnnotatedBatch& batch) {
    if (!batch.custom_metadata) return false;
    for (const auto& key : batch.custom_metadata->keys()) {
        if (key.rfind("vgi.cache.", 0) == 0) return true;
    }
    return false;
}

bool says_more(const vgi_rpc::AnnotatedBatch& batch) {
    return meta(batch, kStatus) == std::string(kHaveMoreOutput);
}

std::shared_ptr<Paged> paged(int pages) {
    Script script;
    script.pages = pages;
    return std::make_shared<Paged>(script);
}

}  // namespace

TEST_CASE("an input answered over several replies is resumed by ticks",
          "[table_in_out][continue]") {
    Exchange exchange(paged(3));
    const auto first = exchange.reply(input({1, 2}));
    CHECK(first.batch->num_rows() == 2);
    CHECK(says_more(first));
    CHECK(says_more(exchange.reply(tick())));
    const auto last = exchange.reply(tick());
    CHECK(last.batch->num_rows() == 2);
    CHECK_FALSE(says_more(last));
    // Done: the next real input starts over.
    CHECK(says_more(exchange.reply(input({3}))));
}

TEST_CASE("a tick with nothing pending is refused", "[table_in_out][continue]") {
    Exchange exchange(paged(1));
    CHECK_THROWS_WITH(exchange.turn(tick()), ContainsSubstring("no output is pending"));

    Exchange done(paged(2));
    (void)done.reply(input({1}));
    (void)done.reply(tick());
    CHECK_THROWS_WITH(done.turn(tick()), ContainsSubstring("no output is pending"));
}

TEST_CASE("a tick with rows is refused", "[table_in_out][continue]") {
    Exchange exchange(paged(2));
    (void)exchange.reply(input({1}));
    CHECK_THROWS_WITH(exchange.turn(tick({7})), ContainsSubstring("must have no rows"));
}

TEST_CASE("real input while output is pending is refused", "[table_in_out][continue]") {
    Exchange exchange(paged(2));
    (void)exchange.reply(input({1}));
    CHECK_THROWS_WITH(exchange.turn(input({2})), ContainsSubstring("expected a continuation tick"));
}

TEST_CASE("a tick's reply may carry no cache control", "[table_in_out][continue][cache]") {
    for (const bool raw : {false, true}) {
        Script script;
        script.pages = 2;
        script.cache_on_resume = !raw;
        script.raw_cache_key_on_resume = raw;
        Exchange exchange(std::make_shared<Paged>(script));
        (void)exchange.reply(input({1}));
        CHECK_THROWS_WITH(exchange.turn(tick()),
                          ContainsSubstring("cache control is only valid on an input's first"));
    }
}

TEST_CASE("the function-wide cache advertisement rides only an input's first reply",
          "[table_in_out][continue][cache]") {
    Script script;
    script.pages = 3;
    Exchange exchange(std::make_shared<Paged>(script, /*advertise_cache=*/true));
    const auto first = exchange.reply(input({1}));
    CHECK(meta(first, kCacheTtl) == std::string("60"));
    const auto second = exchange.reply(tick());
    CHECK(says_more(second));
    CHECK_FALSE(has_cache_keys(second));
    const auto last = exchange.reply(tick());
    CHECK_FALSE(says_more(last));
    CHECK_FALSE(has_cache_keys(last));
}

TEST_CASE("more is refused from a function that cannot resume", "[table_in_out][continue]") {
    Exchange exchange(std::make_shared<Unresumable>());
    CHECK_THROWS_WITH(exchange.turn(input({1})), ContainsSubstring("cannot resume"));
}

TEST_CASE("more is refused from finish()", "[table_in_out][continue]") {
    const Unresumable fn;
    vgi::EmittedBatch flushed(rows({1}));
    CHECK(vgi::finalize_batches(fn, {flushed}).size() == 1);
    flushed.more = true;
    CHECK_THROWS_WITH(vgi::finalize_batches(fn, {flushed}), ContainsSubstring("not finish()"));
}

TEST_CASE("an empty reply still says more", "[table_in_out][continue]") {
    Script script;
    script.pages = 3;
    script.empty_pages = true;
    Exchange exchange(std::make_shared<Paged>(script));
    const auto first = exchange.reply(input({1, 2}));
    CHECK(first.batch->num_rows() == 0);
    CHECK(first.batch->schema()->Equals(*schema()));
    CHECK(says_more(first));
    const auto second = exchange.reply(tick());
    CHECK(second.batch->num_rows() == 0);
    CHECK(says_more(second));
    const auto last = exchange.reply(tick());
    CHECK(last.batch->num_rows() == 0);
    CHECK_FALSE(says_more(last));
}

TEST_CASE("every turn replies with exactly one data batch", "[table_in_out]") {
    // Including one whose function emitted nothing at all: the client waits
    // for a reply per turn.
    class Silent final : public vgi::TableInOutFunction {
    public:
        std::string name() const override { return "silent"; }
        vgi::FunctionMetadata metadata() const override { return {}; }
        std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }
        std::vector<vgi::EmittedBatch> process(
            const vgi::ProcessParams&, const std::shared_ptr<arrow::RecordBatch>&) const override {
            return {vgi::EmittedBatch{}, vgi::EmittedBatch{}};
        }
    };
    Exchange exchange(std::make_shared<Silent>());
    const auto reply = exchange.reply(input({1}));
    CHECK(reply.batch->num_rows() == 0);
    CHECK_FALSE(says_more(reply));
}

TEST_CASE("initial_state may log, on the init call's channel", "[table_in_out][continue]") {
    Script script;
    script.pages = 2;
    script.log_initial_state = true;
    // Opening the continuation would throw std::bad_function_call with no
    // logger bound yet.
    Exchange exchange(std::make_shared<Paged>(script));
    CHECK(exchange.init_logs() == std::vector<std::string>{"opening"});
    CHECK(says_more(exchange.reply(input({1}))));
}
