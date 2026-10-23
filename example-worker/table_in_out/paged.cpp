// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Continued output: table-in-out functions that answer one input batch with
// several replies, `page_size` output rows each, the way a function backed by
// a paged upstream API would.
//
// `blended_paged`, `blended_sliced` and `paged_explode` are cross-SDK fixtures:
// their contracts are the headers of the extension's
// test/sql/integration/table_in_out/{lateral_continuation,input_rows_per_call,
// continuation_classic,continuation_resume_error_revalidate}.test, and the
// canonical implementation is vgi-python's _test_fixtures/table_in_out.py.
//
// The state each keeps between replies is the input's fan-out counts and a
// position in them, never output rows: it is what `resume` continues from.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Where the next reply resumes.
struct PagedState {
    // The current input's fan-out counts, NULL and negative already zeroed.
    std::vector<int64_t> counts;
    // The input row, and the offset within its 0..n-1, the next page starts at.
    size_t row = 0;
    int64_t offset = 0;
    // blended_sliced: the input rows of the current call.
    int64_t call_rows = 0;
    // Replies sent for the current input, for fail_on_page.
    int64_t page = 0;
    // not_modified_more: the first reply is empty and says more.
    bool empty_first = false;
    // Added to every output value (revalidate: the data changed); negative
    // sends the whole output in one reply.
    int64_t bump = 0;
};

// One page of output: the values, the input row each came from, and whether
// more pages follow.
struct Page {
    std::vector<int64_t> values;
    std::vector<int32_t> parents;
    bool more = false;
};

std::vector<int64_t> fan_out_counts(const std::shared_ptr<arrow::Array>& column) {
    auto counts = std::static_pointer_cast<arrow::Int64Array>(cast_to(column, arrow::int64()));
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(counts->length()));
    for (int64_t row = 0; row < counts->length(); ++row) {
        const int64_t n = counts->IsNull(row) ? 0 : counts->Value(row);
        out.push_back(n < 0 ? 0 : n);
    }
    return out;
}

// The next `page_size` rows, which may span input rows. Input rows with no
// output after the last non-empty page are skipped, so the last page is never
// followed by an empty one.
Page next_page(PagedState& state, int64_t page_size) {
    Page page;
    const auto& counts = state.counts;
    while (state.row < counts.size() && static_cast<int64_t>(page.values.size()) < page_size) {
        const int64_t take = std::min(counts[state.row] - state.offset,
                                      page_size - static_cast<int64_t>(page.values.size()));
        for (int64_t i = 0; i < take; ++i) {
            page.values.push_back(state.offset + i);
            page.parents.push_back(static_cast<int32_t>(state.row));
        }
        state.offset += take;
        if (state.offset == counts[state.row]) {
            ++state.row;
            state.offset = 0;
        }
    }
    while (state.row < counts.size() && counts[state.row] == 0) ++state.row;
    page.more = state.row < counts.size();
    return page;
}

// The bind-time named arguments blended_paged and paged_explode share.
struct PagedArgs {
    int64_t page_size;
    int64_t cache_ttl;
    int64_t fail_on_page;
    bool revalidate;
    bool not_modified_more;

    static PagedArgs of(const vgi::Arguments& arguments, int64_t default_page_size) {
        return {arguments.named_int64("page_size").value_or(default_page_size),
                arguments.named_int64("cache_ttl").value_or(0),
                arguments.named_int64("fail_on_page").value_or(0),
                arguments.named_bool("revalidate").value_or(false),
                arguments.named_bool("not_modified_more").value_or(false)};
    }
};

void check_paging_args(int64_t page_size, int64_t cache_ttl) {
    if (page_size < 1) {
        throw std::invalid_argument("page_size must be at least 1, got " +
                                    std::to_string(page_size));
    }
    if (cache_ttl < 0) {
        throw std::invalid_argument("cache_ttl must not be negative, got " +
                                    std::to_string(cache_ttl));
    }
}

std::vector<vgi::ArgSpec> paging_named_args(int64_t default_page_size, bool with_revalidation) {
    std::vector<vgi::ArgSpec> specs{
        vgi::ArgSpec::named(
            "page_size", "int64",
            "Output rows per reply (default " + std::to_string(default_page_size) + ")"),
        vgi::ArgSpec::named("cache_ttl", "int64",
                            "When > 0, advertise vgi.cache.ttl on the first reply")};
    if (with_revalidation) {
        specs.push_back(vgi::ArgSpec::named(
            "fail_on_page", "int64", "When > 0, fail instead of sending that (1-based) reply"));
        specs.push_back(vgi::ArgSpec::named(
            "revalidate", "boolean", "Answer conditional requests with changed, paged output"));
        specs.push_back(vgi::ArgSpec::named("not_modified_more", "boolean",
                                            "With revalidate, answer a conditional request with "
                                            "an empty not_modified reply that says more"));
    }
    return specs;
}

vgi::CacheControl ttl_only(int64_t ttl) {
    vgi::CacheControl control;
    control.ttl_seconds = ttl;
    return control;
}

vgi::CacheControl revalidatable(const char* etag, bool not_modified = false) {
    vgi::CacheControl control;
    control.ttl_seconds = 0;
    control.etag = etag;
    control.revalidatable = true;
    control.not_modified = not_modified;
    return control;
}

// Start a new input; returns its first reply's cache control.
//
// `revalidate` makes the output depend on the request alone. A plain request
// gets the whole output in one reply, cacheable but immediately stale (ttl 0,
// etag "v1", revalidatable), so the client stores it and revalidates every
// repeat. A conditional request is told the data changed: every value is 1000
// larger, the etag is "v2", and the output is paged as usual — which the
// client must return without storing, since it continued.
//
// `not_modified_more` breaks the protocol on purpose: a conditional request is
// answered with an empty not_modified reply that also says more, then the
// changed pages. A 304 stands for an input's whole stored output, so the
// client must refuse it rather than serve its entry.
std::optional<vgi::CacheControl> paged_begin(const vgi::ProcessParams& params,
                                             const PagedArgs& args, PagedState& state,
                                             std::vector<int64_t> counts) {
    state = PagedState{};
    state.counts = std::move(counts);
    if (!args.revalidate) {
        if (args.cache_ttl > 0) return ttl_only(args.cache_ttl);
        return std::nullopt;
    }
    if (!params.if_none_match) {
        state.bump = -1;
        return revalidatable("v1");
    }
    state.bump = 1000;
    if (args.not_modified_more) {
        state.empty_first = true;
        return revalidatable("v1", /*not_modified=*/true);
    }
    return revalidatable("v2");
}

// The next reply of blended_paged / paged_explode, or the failure
// `fail_on_page` asks for.
Page paged_next(const PagedArgs& args, PagedState& state) {
    ++state.page;
    if (state.page == args.fail_on_page) {
        throw std::runtime_error("failing on page " + std::to_string(state.page) + " as asked");
    }
    if (state.page == 1 && state.empty_first) return Page{{}, {}, /*more=*/true};
    if (state.bump < 0) return next_page(state, std::numeric_limits<int64_t>::max());
    auto page = next_page(state, args.page_size);
    for (auto& value : page.values) value += state.bump;
    return page;
}

std::shared_ptr<arrow::Array> int64_array(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    (void)builder.AppendValues(values);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// One reply carrying `columns`, in output-schema order, and the page's
// provenance, cache control and continuation.
vgi::EmittedBatch page_reply(const vgi::ProcessParams& params,
                             std::vector<std::shared_ptr<arrow::Array>> columns, Page page,
                             std::optional<vgi::CacheControl> cache_control, bool with_parents) {
    const auto rows = static_cast<int64_t>(page.values.size());
    vgi::EmittedBatch emitted{
        arrow::RecordBatch::Make(params.output_schema, rows, std::move(columns))};
    if (with_parents) emitted.parent_rows = std::move(page.parents);
    emitted.cache_control = std::move(cache_control);
    emitted.more = page.more;
    return emitted;
}

vgi::FunctionMetadata paged_metadata(std::string description, bool blended) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = {blended ? "blended" : "table_in_out", "test"};
    md.input_from_args = blended;
    return md;
}

// `blended_paged(n, page_size := 2, ...)` — blended_explode's output (rows
// 0..n-1 per input row, with provenance), `page_size` rows per reply.
class BlendedPaged : public vgi::ResumableTableInOutFunction<PagedState> {
public:
    std::string name() const override { return "blended_paged"; }

    vgi::FunctionMetadata metadata() const override {
        return paged_metadata(
            "Blended 1->N fan-out (emit 0..n-1 per input row), page_size rows per reply",
            /*blended=*/true);
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto specs = paging_named_args(2, /*with_revalidation=*/true);
        specs.insert(specs.begin(),
                     vgi::ArgSpec::column("n", 0, "int64", "Fan-out count: emit rows 0..n-1"));
        return specs;
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        const auto args = PagedArgs::of(params.arguments, 2);
        check_paging_args(args.page_size, args.cache_ttl);
        return arrow::schema({arrow::field("i", arrow::int64(), /*nullable=*/true)});
    }

    std::vector<vgi::EmittedBatch> process(const vgi::ProcessParams& params,
                                           const std::shared_ptr<arrow::RecordBatch>& batch,
                                           PagedState& state) const override {
        const auto args = PagedArgs::of(params.arguments, 2);
        auto column = batch->GetColumnByName("n");
        if (!column) column = batch->column(0);
        auto cache = paged_begin(params, args, state, fan_out_counts(column));
        return {reply(params, args, state, std::move(cache))};
    }

    std::vector<vgi::EmittedBatch> resume(const vgi::ProcessParams& params,
                                          PagedState& state) const override {
        return {reply(params, PagedArgs::of(params.arguments, 2), state, std::nullopt)};
    }

private:
    static vgi::EmittedBatch reply(const vgi::ProcessParams& params, const PagedArgs& args,
                                   PagedState& state, std::optional<vgi::CacheControl> cache) {
        auto page = paged_next(args, state);
        auto values = int64_array(page.values);
        return page_reply(params, {values}, std::move(page), std::move(cache),
                          /*with_parents=*/true);
    }
};

// `blended_sliced(n, page_size := 1000000, cache_ttl := 0)` — blended_paged
// taking at most 3 input rows per call, and saying in `call_rows` how many it
// got, so a test can see the engine's split.
class BlendedSliced : public vgi::ResumableTableInOutFunction<PagedState> {
public:
    static constexpr int64_t kRowsPerCall = 3;
    static constexpr int64_t kDefaultPageSize = 1000000;

    std::string name() const override { return "blended_sliced"; }

    vgi::FunctionMetadata metadata() const override {
        auto md = paged_metadata(
            "Blended 1->N fan-out taking at most 3 input rows per call, reporting each call's "
            "size",
            /*blended=*/true);
        md.input_rows_per_call = kRowsPerCall;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto specs = paging_named_args(kDefaultPageSize, /*with_revalidation=*/false);
        specs.insert(specs.begin(),
                     vgi::ArgSpec::column("n", 0, "int64", "Fan-out count: emit rows 0..n-1"));
        return specs;
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        check_paging_args(params.arguments.named_int64("page_size").value_or(kDefaultPageSize),
                          params.arguments.named_int64("cache_ttl").value_or(0));
        return arrow::schema({arrow::field("i", arrow::int64(), /*nullable=*/true),
                              arrow::field("call_rows", arrow::int64(), /*nullable=*/true)});
    }

    std::vector<vgi::EmittedBatch> process(const vgi::ProcessParams& params,
                                           const std::shared_ptr<arrow::RecordBatch>& batch,
                                           PagedState& state) const override {
        if (batch->num_rows() > kRowsPerCall) {
            throw std::runtime_error("blended_sliced declares input_rows_per_call = 3 but got " +
                                     std::to_string(batch->num_rows()) + " rows");
        }
        auto column = batch->GetColumnByName("n");
        if (!column) column = batch->column(0);
        state = PagedState{};
        state.counts = fan_out_counts(column);
        state.call_rows = batch->num_rows();
        std::optional<vgi::CacheControl> cache;
        if (const auto ttl = params.arguments.named_int64("cache_ttl").value_or(0); ttl > 0) {
            cache = ttl_only(ttl);
            cache->per_value = true;
        }
        return {reply(params, state, std::move(cache))};
    }

    std::vector<vgi::EmittedBatch> resume(const vgi::ProcessParams& params,
                                          PagedState& state) const override {
        return {reply(params, state, std::nullopt)};
    }

private:
    static vgi::EmittedBatch reply(const vgi::ProcessParams& params, PagedState& state,
                                   std::optional<vgi::CacheControl> cache) {
        auto page =
            next_page(state, params.arguments.named_int64("page_size").value_or(kDefaultPageSize));
        auto values = int64_array(page.values);
        auto call_rows = int64_array(std::vector<int64_t>(page.values.size(), state.call_rows));
        return page_reply(params, {values, call_rows}, std::move(page), std::move(cache),
                          /*with_parents=*/true);
    }
};

// `paged_explode(data, page_size := 2, ...)` — the classic (TABLE-input)
// counterpart of blended_paged, run through the engine's stock in-out path on
// a per-substream connection. The input table's first column is the count.
class PagedExplode : public vgi::ResumableTableInOutFunction<PagedState> {
public:
    std::string name() const override { return "paged_explode"; }

    vgi::FunctionMetadata metadata() const override {
        return paged_metadata("Classic table-in-out 1->N fan-out, page_size rows per reply",
                              /*blended=*/false);
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto specs = paging_named_args(2, /*with_revalidation=*/true);
        specs.insert(specs.begin(),
                     vgi::ArgSpec::table("data", 0, "Input table; its first column is the count"));
        return specs;
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        const auto args = PagedArgs::of(params.arguments, 2);
        check_paging_args(args.page_size, args.cache_ttl);
        return arrow::schema({arrow::field("i", arrow::int64(), /*nullable=*/true)});
    }

    std::vector<vgi::EmittedBatch> process(const vgi::ProcessParams& params,
                                           const std::shared_ptr<arrow::RecordBatch>& batch,
                                           PagedState& state) const override {
        const auto args = PagedArgs::of(params.arguments, 2);
        auto counts =
            batch->num_columns() > 0 ? fan_out_counts(batch->column(0)) : std::vector<int64_t>{};
        auto cache = paged_begin(params, args, state, std::move(counts));
        return {reply(params, args, state, std::move(cache))};
    }

    std::vector<vgi::EmittedBatch> resume(const vgi::ProcessParams& params,
                                          PagedState& state) const override {
        return {reply(params, PagedArgs::of(params.arguments, 2), state, std::nullopt)};
    }

private:
    // No provenance: the stock path stamps nothing per row.
    static vgi::EmittedBatch reply(const vgi::ProcessParams& params, const PagedArgs& args,
                                   PagedState& state, std::optional<vgi::CacheControl> cache) {
        auto page = paged_next(args, state);
        auto values = int64_array(page.values);
        return page_reply(params, {values}, std::move(page), std::move(cache),
                          /*with_parents=*/false);
    }
};

}  // namespace

void register_paged(vgi::Worker& worker) {
    worker.register_table_in_out(std::make_shared<BlendedPaged>());
    worker.register_table_in_out(std::make_shared<BlendedSliced>());
    worker.register_table_in_out(std::make_shared<PagedExplode>());
}

}  // namespace example
