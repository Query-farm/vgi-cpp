// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Fixtures that report what was pushed into them.
//
// Most of these do not *use* the filters — they render them into a column, so
// a test can assert on exactly what the engine sent. `value_prune` is the
// exception that proves the rule: it echoes the resolved discrete value set
// *and* prunes by it, which is the partition-pruning idiom the accessor
// exists for.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/compute/api.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

class FilterEcho : public vgi::TableFunction {
public:
    std::string name() const override { return "filter_echo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Echoes pushed-down filter predicates in output";
        md.categories = {"generator", "diagnostic"};
        md.filter_pushdown = true;
        // On, and load-bearing: DuckDB will not install a join filter on a get
        // that does not take projection pushdown, so without this the join-key
        // tests see no filters at all. The report column survives because the
        // projection is whatever the query selected, and a query that asks for
        // the report gets it.
        md.projection_pushdown = true;
        // Reports the filters *and* applies them. The engine trusts a function
        // that declares filter_pushdown to have honoured the predicate, so
        // advertising without applying returns rows the WHERE excluded.
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
                vgi::ArgSpec::named("batch_size", "int64", "Batch size for output")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), true),
                              arrow::field("s", arrow::utf8(), true),
                              arrow::field("pushed_filters", arrow::utf8(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        vgi::TableCardinality estimate;
        if (auto count = params.arguments.const_int64(0)) {
            estimate.estimate = *count;
            estimate.max = *count;
        }
        return estimate;
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(
            params.output_schema, std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)),
            std::max<int64_t>(1, params.arguments.named_int64("batch_size").value_or(2048)),
            params.pushdown_filters.format());
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size,
                 std::string filters)
            : schema_(std::move(schema)),
              remaining_(rows),
              batch_size_(batch_size),
              filters_(std::move(filters)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t n = std::min(remaining_, batch_size_);

            arrow::Int64Builder ns;
            arrow::StringBuilder ss;
            arrow::StringBuilder pushed;
            (void)ns.Reserve(n);
            for (int64_t i = cursor_; i < cursor_ + n; ++i) {
                (void)ns.Append(i);
                (void)ss.Append("row_" + std::to_string(i));
                (void)pushed.Append(filters_);
            }

            // Built in the *bound* schema's column order, since projection
            // pushdown may have narrowed and reordered it.
            std::vector<std::shared_ptr<arrow::Array>> built(3);
            (void)ns.Finish(&built[0]);
            (void)ss.Finish(&built[1]);
            (void)pushed.Finish(&built[2]);

            static const char* kNames[] = {"n", "s", "pushed_filters"};
            std::vector<std::shared_ptr<arrow::Array>> columns;
            columns.reserve(static_cast<size_t>(schema_->num_fields()));
            for (int i = 0; i < schema_->num_fields(); ++i) {
                const auto& wanted = schema_->field(i)->name();
                for (size_t j = 0; j < 3; ++j) {
                    if (wanted == kNames[j]) columns.push_back(built[j]);
                }
            }
            cursor_ += n;
            remaining_ -= n;
            return arrow::RecordBatch::Make(schema_, n, columns);
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        std::string filters_;
        int64_t cursor_ = 0;
    };
};

// Rendered as the reference fixtures render it: values sorted, then joined
// with commas. The tests compare the string, so the ordering is the contract —
// numeric where the values are numbers, lexicographic otherwise.
std::string join_sorted(const std::shared_ptr<arrow::Array>& values) {
    std::vector<std::string> rendered;
    rendered.reserve(static_cast<size_t>(values->length()));

    // Switched on the array's own type rather than on whether a cast happens
    // to succeed: casting strings to int64 would silently reorder a set of
    // numeric-looking tags.
    if (arrow::is_integer(values->type_id())) {
        auto numbers = arrow::compute::Cast(*values, arrow::int64());
        if (!numbers.ok()) return {};
        std::vector<int64_t> sorted;
        const auto& array = static_cast<const arrow::Int64Array&>(**numbers);
        for (int64_t i = 0; i < array.length(); ++i) {
            if (!array.IsNull(i)) sorted.push_back(array.Value(i));
        }
        std::sort(sorted.begin(), sorted.end());
        for (const auto value : sorted) rendered.push_back(std::to_string(value));
    } else {
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) continue;
            auto scalar = values->GetScalar(i);
            if (!scalar.ok()) continue;
            rendered.push_back(scalar.ValueUnsafe()->ToString());
        }
        std::sort(rendered.begin(), rendered.end());
    }

    std::string joined;
    for (const auto& value : rendered) {
        if (!joined.empty()) joined += ',';
        joined += value;
    }
    return joined;
}

int64_t arg_count(const vgi::ProcessParams& params) {
    return std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
}

int64_t arg_batch_size(const vgi::ProcessParams& params) {
    return std::max<int64_t>(1, params.arguments.named_int64("batch_size").value_or(2048));
}

std::vector<vgi::ArgSpec> count_and_batch_size(const char* count_doc) {
    return {vgi::ArgSpec::constant_arg("count", 0, "int64", count_doc),
            vgi::ArgSpec::named("batch_size", "int64", "Batch size for output")};
}

vgi::TableCardinality count_cardinality(const vgi::ProcessParams& params) {
    vgi::TableCardinality estimate;
    if (auto count = params.arguments.const_int64(0)) {
        estimate.estimate = *count;
        estimate.max = *count;
    }
    return estimate;
}

// Assemble `size` rows of `schema`, drawing each column from `build` by name.
//
// By name because all three fixtures below declare projection pushdown: the
// bound schema is whatever subset the query needs, in the engine's order.
std::shared_ptr<arrow::RecordBatch> assemble(
    const std::shared_ptr<arrow::Schema>& schema, int64_t size,
    const std::function<std::shared_ptr<arrow::Array>(const std::string&, int64_t)>& build) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(static_cast<size_t>(schema->num_fields()));
    for (const auto& field : schema->fields()) {
        auto column = build(field->name(), size);
        if (!column) throw std::runtime_error("filters: unexpected column " + field->name());
        columns.push_back(std::move(column));
    }
    return arrow::RecordBatch::Make(schema, size, columns);
}

std::shared_ptr<arrow::Array> repeat_string(const std::string& value, int64_t size) {
    arrow::StringBuilder builder;
    for (int64_t i = 0; i < size; ++i) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> repeat_bool(bool value, int64_t size) {
    arrow::BooleanBuilder builder;
    (void)builder.Reserve(size);
    for (int64_t i = 0; i < size; ++i) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> int64_array(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    (void)builder.Reserve(static_cast<int64_t>(values.size()));
    for (const auto value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// `value_prune(count)` — emits only the keys `column_values("n")` resolved to.
//
// The point is the accessor, not the filtering: `resolved` echoes exactly what
// it returned, so an AND-descent or OR-union regression is visible from SQL
// even though the framework would have filtered the rows anyway.
class ValuePrune : public vgi::TableFunction {
public:
    std::string name() const override { return "value_prune"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Prunes the key set via get_column_values('n'); echoes the resolved discrete values";
        md.categories = {"generator", "diagnostic"};
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return count_and_batch_size("Number of candidate rows (keys 0..count-1)");
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), true),
                              arrow::field("resolved", arrow::utf8(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const int64_t count = arg_count(params);
        auto discrete = params.pushdown_filters.column_values("n");

        std::vector<int64_t> keys;
        std::string resolved;
        if (discrete) {
            resolved = join_sorted(discrete);
            // The resolved set can name keys this scan does not own; a caller
            // pruning partitions would simply open none of them.
            auto casted = arrow::compute::Cast(*discrete, arrow::int64());
            if (casted.ok()) {
                const auto& array = static_cast<const arrow::Int64Array&>(**casted);
                for (int64_t i = 0; i < array.length(); ++i) {
                    if (!array.IsNull(i) && array.Value(i) >= 0 && array.Value(i) < count) {
                        keys.push_back(array.Value(i));
                    }
                }
                std::sort(keys.begin(), keys.end());
            }
        } else {
            // Null is "cannot enumerate", never "no rows" — so the fallback is
            // to scan every candidate.
            resolved = "(scan)";
            keys.reserve(static_cast<size_t>(count));
            for (int64_t i = 0; i < count; ++i) keys.push_back(i);
        }
        return std::make_unique<Producer>(params.output_schema, std::move(keys),
                                          std::move(resolved), arg_batch_size(params));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, std::vector<int64_t> keys,
                 std::string resolved, int64_t batch_size)
            : schema_(std::move(schema)),
              keys_(std::move(keys)),
              resolved_(std::move(resolved)),
              batch_size_(batch_size) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            const auto total = static_cast<int64_t>(keys_.size());
            if (cursor_ >= total) return nullptr;
            const int64_t size = std::min(total - cursor_, batch_size_);
            std::vector<int64_t> chunk(keys_.begin() + cursor_, keys_.begin() + cursor_ + size);
            cursor_ += size;

            return assemble(
                schema_, size,
                [&](const std::string& column, int64_t rows) -> std::shared_ptr<arrow::Array> {
                    if (column == "n") return int64_array(chunk);
                    if (column == "resolved") return repeat_string(resolved_, rows);
                    return nullptr;
                });
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        std::vector<int64_t> keys_;
        std::string resolved_;
        int64_t batch_size_;
        int64_t cursor_ = 0;
    };
};

// `filtered_columns_echo(count)` — the column-introspection accessors, echoed.
class FilteredColumnsEcho : public vgi::TableFunction {
public:
    std::string name() const override { return "filtered_columns_echo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Echoes filtered_columns / has_filter_for_column / get_column_values_array";
        md.categories = {"generator", "diagnostic"};
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return count_and_batch_size("Number of rows to generate");
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), true),
                              arrow::field("tag", arrow::utf8(), true),
                              arrow::field("filtered_cols", arrow::utf8(), true),
                              arrow::field("has_n", arrow::boolean(), true),
                              arrow::field("has_tag", arrow::boolean(), true),
                              arrow::field("tag_values", arrow::utf8(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto& filters = params.pushdown_filters;

        Diagnostics diagnostics;
        for (const auto& column : filters.filtered_columns()) {
            if (!diagnostics.filtered_cols.empty()) diagnostics.filtered_cols += ',';
            diagnostics.filtered_cols += column;
        }
        diagnostics.has_n = filters.has_filter_for_column("n");
        diagnostics.has_tag = filters.has_filter_for_column("tag");
        auto tag_values = filters.column_values("tag");
        diagnostics.tag_values = tag_values ? join_sorted(tag_values) : "(none)";

        return std::make_unique<Producer>(params.output_schema, arg_count(params),
                                          arg_batch_size(params), std::move(diagnostics));
    }

private:
    struct Diagnostics {
        std::string filtered_cols;
        bool has_n = false;
        bool has_tag = false;
        std::string tag_values;
    };

    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size,
                 Diagnostics diagnostics)
            : schema_(std::move(schema)),
              remaining_(rows),
              batch_size_(batch_size),
              diagnostics_(std::move(diagnostics)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t size = std::min(remaining_, batch_size_);
            const int64_t start = cursor_;
            cursor_ += size;
            remaining_ -= size;

            return assemble(
                schema_, size,
                [&](const std::string& column, int64_t rows) -> std::shared_ptr<arrow::Array> {
                    if (column == "n") {
                        std::vector<int64_t> values;
                        values.reserve(static_cast<size_t>(rows));
                        for (int64_t i = 0; i < rows; ++i) values.push_back(start + i);
                        return int64_array(values);
                    }
                    if (column == "tag") {
                        arrow::StringBuilder builder;
                        for (int64_t i = 0; i < rows; ++i) {
                            (void)builder.Append("t" + std::to_string(start + i));
                        }
                        std::shared_ptr<arrow::Array> array;
                        (void)builder.Finish(&array);
                        return array;
                    }
                    if (column == "filtered_cols") {
                        return repeat_string(diagnostics_.filtered_cols, rows);
                    }
                    if (column == "has_n") {
                        return repeat_bool(diagnostics_.has_n, rows);
                    }
                    if (column == "has_tag") {
                        return repeat_bool(diagnostics_.has_tag, rows);
                    }
                    if (column == "tag_values") {
                        return repeat_string(diagnostics_.tag_values, rows);
                    }
                    return nullptr;
                });
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        Diagnostics diagnostics_;
        int64_t cursor_ = 0;
    };
};

// `dict_filter_echo(count)` — a dictionary-encoded column DuckDB types as
// VARCHAR.
//
// The pair is the point: the engine pushes a plain string literal while the
// worker emits `dictionary<int8, utf8>`, so the auto-applied filter has to
// compare the two. Casting the literal up to the dictionary type throws; the
// column has to be decoded down to its value type instead.
class DictFilterEcho : public vgi::TableFunction {
public:
    std::string name() const override { return "dict_filter_echo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Emits a dictionary-encoded VARCHAR column for filter-pushdown testing";
        md.categories = {"generator", "diagnostic", "testing"};
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return count_and_batch_size("Number of rows to generate");
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema(
            {arrow::field("n", arrow::int64(), true),
             arrow::field("s", arrow::dictionary(arrow::int8(), arrow::utf8()), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema, arg_count(params),
                                          arg_batch_size(params));
    }

private:
    // Low cardinality and deterministic, so the row-to-value mapping is easy
    // to assert and the dictionary encoding is actually meaningful.
    static constexpr const char* kValues[] = {"red", "green", "blue"};

    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size)
            : schema_(std::move(schema)), remaining_(rows), batch_size_(batch_size) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t size = std::min(remaining_, batch_size_);
            const int64_t start = cursor_;
            cursor_ += size;
            remaining_ -= size;

            return assemble(
                schema_, size,
                [&](const std::string& column, int64_t rows) -> std::shared_ptr<arrow::Array> {
                    if (column == "n") {
                        std::vector<int64_t> values;
                        values.reserve(static_cast<size_t>(rows));
                        for (int64_t i = 0; i < rows; ++i) values.push_back(start + i);
                        return int64_array(values);
                    }
                    if (column == "s") return colors(start, rows);
                    return nullptr;
                });
        }

    private:
        static std::shared_ptr<arrow::Array> colors(int64_t start, int64_t rows) {
            arrow::StringBuilder dictionary;
            for (const auto* value : kValues) (void)dictionary.Append(value);
            std::shared_ptr<arrow::Array> values;
            (void)dictionary.Finish(&values);

            arrow::Int8Builder indices;
            (void)indices.Reserve(rows);
            constexpr int8_t kCount = static_cast<int8_t>(std::size(kValues));
            for (int64_t i = start; i < start + rows; ++i) {
                (void)indices.Append(static_cast<int8_t>(i % kCount));
            }
            std::shared_ptr<arrow::Array> index_array;
            (void)indices.Finish(&index_array);

            auto encoded = arrow::DictionaryArray::FromArrays(
                arrow::dictionary(arrow::int8(), arrow::utf8()), index_array, values);
            if (!encoded.ok()) {
                throw std::runtime_error("dict_filter_echo: " + encoded.status().message());
            }
            return encoded.MoveValueUnsafe();
        }

        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        int64_t cursor_ = 0;
    };
};

// `dynamic_filter_echo(count, batch_size := 2048)` — descending values that
// report the filter in force when each batch was produced.
//
// What it probes that `filter_echo` does not: the filters that arrive *after*
// the scan began. A Top-N heap tightens its threshold as it fills, and the
// engine re-sends the predicate on every tick, so the values are descending —
// then the boundary moves with each batch and the report changes with it.
class DynamicFilterEcho : public vgi::TableFunction {
public:
    std::string name() const override { return "dynamic_filter_echo"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Echoes the dynamic filters in force when each batch was produced";
        md.categories = {"generator", "diagnostic"};
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return count_and_batch_size("Number of rows to generate");
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), true),
                              arrow::field("s", arrow::utf8(), true),
                              arrow::field("pushed_filters", arrow::utf8(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema, arg_count(params),
                                          arg_batch_size(params));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size)
            : schema_(std::move(schema)), remaining_(rows), batch_size_(batch_size) {}

        void on_dynamic_filters(const vgi::PushdownFilters& filters) override {
            // Rendered as the kinds rather than as SQL: what the test is
            // checking is which *shape* the engine pushed, and that it moved.
            report_ = filters.format_repr();
        }

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t size = std::min(remaining_, batch_size_);
            const int64_t start = cursor_;
            cursor_ += size;
            remaining_ -= size;

            const auto report = report_;
            return assemble(
                schema_, size,
                [&](const std::string& column, int64_t count) -> std::shared_ptr<arrow::Array> {
                    if (column == "n") {
                        // Descending, so a Top-N over `n` ASC has
                        // to keep tightening rather than settling
                        // on the first batch.
                        std::vector<int64_t> values;
                        values.reserve(static_cast<size_t>(count));
                        for (int64_t i = 0; i < count; ++i) {
                            values.push_back(total_ - 1 - (start + i));
                        }
                        return int64_array(values);
                    }
                    if (column == "s") {
                        arrow::StringBuilder builder;
                        for (int64_t i = 0; i < count; ++i) {
                            (void)builder.Append("row_" + std::to_string(total_ - 1 - (start + i)));
                        }
                        std::shared_ptr<arrow::Array> array;
                        (void)builder.Finish(&array);
                        return array;
                    }
                    if (column == "pushed_filters") {
                        return repeat_string(report, count);
                    }
                    return nullptr;
                });
        }

    private:
        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        int64_t total_ = remaining_;
        int64_t cursor_ = 0;
        // "(none)" until the engine sends one, which is what the first batch
        // of an unfiltered scan reports.
        std::string report_ = "(none)";
    };
};

// `expression_filter_test(count, batch_size := 2048)` — rows an expression
// filter can be pushed into, with a list column the simpler fixtures lack.
class ExpressionFilterTest : public vgi::TableFunction {
public:
    std::string name() const override { return "expression_filter_test"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Generates rows for non-spatial expression filter testing";
        md.categories = {"generator", "testing"};
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return count_and_batch_size("Number of rows to generate");
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema(
            {arrow::field("id", arrow::int64(), true), arrow::field("name", arrow::utf8(), true),
             arrow::field("tags", arrow::list(arrow::field("item", arrow::utf8(), true)), true),
             arrow::field("score", arrow::float64(), true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<Producer>(params.output_schema, arg_count(params),
                                          arg_batch_size(params));
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size)
            : schema_(std::move(schema)), remaining_(rows), batch_size_(batch_size) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t size = std::min(remaining_, batch_size_);
            const int64_t start = cursor_;
            cursor_ += size;
            remaining_ -= size;

            return assemble(
                schema_, size,
                [&](const std::string& column, int64_t count) -> std::shared_ptr<arrow::Array> {
                    if (column == "id") {
                        std::vector<int64_t> values;
                        values.reserve(static_cast<size_t>(count));
                        for (int64_t i = 0; i < count; ++i) values.push_back(start + i);
                        return int64_array(values);
                    }
                    if (column == "name") {
                        arrow::StringBuilder builder;
                        for (int64_t i = 0; i < count; ++i) {
                            (void)builder.Append("item_" + std::to_string(start + i));
                        }
                        std::shared_ptr<arrow::Array> array;
                        (void)builder.Finish(&array);
                        return array;
                    }
                    if (column == "score") {
                        arrow::DoubleBuilder builder;
                        (void)builder.Reserve(count);
                        for (int64_t i = 0; i < count; ++i) {
                            (void)builder.Append(static_cast<double>(start + i) * 1.1);
                        }
                        std::shared_ptr<arrow::Array> array;
                        (void)builder.Finish(&array);
                        return array;
                    }
                    if (column == "tags") return tags(start, count);
                    return nullptr;
                });
        }

    private:
        // Two overlapping tags per row, so `list_contains` matches a run of
        // rows rather than exactly one.
        static std::shared_ptr<arrow::Array> tags(int64_t start, int64_t count) {
            auto items = std::make_shared<arrow::StringBuilder>();
            arrow::ListBuilder builder(arrow::default_memory_pool(), items);
            for (int64_t i = 0; i < count; ++i) {
                (void)builder.Append();
                const int64_t id = start + i;
                (void)items->Append("tag_" + std::to_string(id % 5));
                (void)items->Append("tag_" + std::to_string((id + 1) % 5));
            }
            std::shared_ptr<arrow::Array> array;
            (void)builder.Finish(&array);
            return array;
        }

        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        int64_t cursor_ = 0;
    };
};

// `spatial_filter_example(count, batch_size := 2048)` — a grid of points with
// a WKB geometry column, for spatial filter pushdown.
class SpatialFilterExample : public vgi::TableFunction {
public:
    std::string name() const override { return "spatial_filter_example"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Generates points on a grid with geometry for spatial filter testing";
        md.categories = {"generator", "spatial", "testing"};
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return count_and_batch_size("Number of points to generate");
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        // The extension type markers are what make DuckDB read the blob as a
        // GEOMETRY rather than as bytes.
        auto geom =
            arrow::field("geom", arrow::binary(), true)
                ->WithMetadata(arrow::key_value_metadata(
                    {"ARROW:extension:name", "ARROW:extension:metadata"}, {"geoarrow.wkb", "{}"}));
        return arrow::schema({arrow::field("n", arrow::int64(), true),
                              arrow::field("x", arrow::float64(), true),
                              arrow::field("y", arrow::float64(), true), geom});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        return count_cardinality(params);
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const int64_t count = arg_count(params);
        // A square grid, so the points spread over the unit square rather than
        // along one line — a bounding-box filter has to exclude some.
        const auto columns = std::max<int64_t>(
            1, static_cast<int64_t>(std::ceil(std::sqrt(static_cast<double>(count)))));
        return std::make_unique<Producer>(params.output_schema, count, arg_batch_size(params),
                                          columns);
    }

private:
    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size,
                 int64_t columns)
            : schema_(std::move(schema)),
              remaining_(rows),
              batch_size_(batch_size),
              columns_(columns) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t size = std::min(remaining_, batch_size_);
            const int64_t start = cursor_;
            cursor_ += size;
            remaining_ -= size;

            return assemble(
                schema_, size,
                [&](const std::string& column, int64_t count) -> std::shared_ptr<arrow::Array> {
                    if (column == "n") {
                        std::vector<int64_t> values;
                        values.reserve(static_cast<size_t>(count));
                        for (int64_t i = 0; i < count; ++i) values.push_back(start + i);
                        return int64_array(values);
                    }
                    if (column == "x" || column == "y") {
                        arrow::DoubleBuilder builder;
                        (void)builder.Reserve(count);
                        for (int64_t i = 0; i < count; ++i) {
                            (void)builder.Append(coordinate(start + i, column == "x"));
                        }
                        std::shared_ptr<arrow::Array> array;
                        (void)builder.Finish(&array);
                        return array;
                    }
                    if (column == "geom") {
                        arrow::BinaryBuilder builder;
                        for (int64_t i = 0; i < count; ++i) {
                            (void)builder.Append(wkb_point(coordinate(start + i, true),
                                                           coordinate(start + i, false)));
                        }
                        std::shared_ptr<arrow::Array> array;
                        (void)builder.Finish(&array);
                        return array;
                    }
                    return nullptr;
                });
        }

    private:
        double coordinate(int64_t index, bool horizontal) const {
            const auto step = static_cast<double>(columns_);
            return static_cast<double>(horizontal ? index % columns_ : index / columns_) / step;
        }

        // Little-endian WKB for POINT(x y), which is the blob DuckDB reads a
        // GEOMETRY out of.
        static std::string wkb_point(double x, double y) {
            std::string wkb{'\x01', '\x01', '\x00', '\x00', '\x00'};
            for (double value : {x, y}) {
                char bytes[sizeof(double)];
                std::memcpy(bytes, &value, sizeof(value));
                wkb.append(bytes, sizeof(bytes));
            }
            return wkb;
        }

        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        int64_t columns_;
        int64_t cursor_ = 0;
    };
};

}  // namespace

void register_filter_fixtures(vgi::Worker& worker) {
    worker.register_table(std::make_shared<FilterEcho>());
    worker.register_table(std::make_shared<ValuePrune>());
    worker.register_table(std::make_shared<FilteredColumnsEcho>());
    worker.register_table(std::make_shared<DictFilterEcho>());
    worker.register_table(std::make_shared<DynamicFilterEcho>());
    worker.register_table(std::make_shared<ExpressionFilterTest>());
    worker.register_table(std::make_shared<SpatialFilterExample>());
}

}  // namespace example
