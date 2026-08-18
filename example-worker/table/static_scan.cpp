// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The scans behind the catalog's fixed reference tables.
//
// These carry no arguments and no state: a table like `data.departments` is a
// name the catalog binds to one of these, and the rows never change. What the
// suite asks of them is metadata — comments, defaults, constraints, column
// statistics — so the rows exist mainly to prove the binding resolves.
//
// The exceptions earn their place: `rowid_sequence` is the only scan that
// emits a virtual row_id, `late_materialization` is the only one that reports
// what a rowid filter pushed into it, and the two `tt_pushdown` scans are the
// only ones that answer both a version and a filter in the same row.

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

std::shared_ptr<arrow::Array> int64_column(std::vector<int64_t> values) {
    arrow::Int64Builder builder;
    for (int64_t value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> double_column(std::vector<double> values) {
    arrow::DoubleBuilder builder;
    for (double value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> string_column(std::vector<std::string> values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// A fixed table: named columns of equal length.
using Columns = std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>;

// One batch, then exhaustion.
class OneShot : public vgi::TableProducer {
public:
    explicit OneShot(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        auto batch = batch_;
        batch_ = nullptr;
        return batch;
    }

private:
    std::shared_ptr<arrow::RecordBatch> batch_;
};

// `columns` in the order and subset `schema` names.
//
// By name rather than by position because projection pushdown reorders as
// freely as it drops: a scan that answered positionally would return the right
// values under the wrong headings.
std::shared_ptr<arrow::RecordBatch> project(const std::shared_ptr<arrow::Schema>& schema,
                                            const Columns& columns, int64_t rows) {
    std::vector<std::shared_ptr<arrow::Array>> projected;
    projected.reserve(static_cast<size_t>(schema->num_fields()));
    for (const auto& field : schema->fields()) {
        auto found = std::find_if(columns.begin(), columns.end(),
                                  [&](const auto& entry) { return entry.first == field->name(); });
        if (found == columns.end()) {
            throw std::runtime_error("static scan was asked for unknown column \"" +
                                     field->name() + "\"");
        }
        projected.push_back(found->second);
    }
    return arrow::RecordBatch::Make(schema, rows, std::move(projected));
}

// A no-argument scan over a fixed set of rows.
class StaticScan : public vgi::TableFunction {
public:
    StaticScan(std::string name, Columns columns)
        : name_(std::move(name)), columns_(std::move(columns)) {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        fields.reserve(columns_.size());
        for (const auto& [column, array] : columns_) {
            fields.push_back(arrow::field(column, array->type(), /*nullable=*/true));
        }
        schema_ = arrow::schema(std::move(fields));
    }

    std::string name() const override { return name_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Static catalog scan";
        md.categories = {"catalog"};
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override { return schema_; }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override {
        return {rows(), rows()};
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        return std::make_unique<OneShot>(project(params.output_schema, columns_, rows()));
    }

private:
    int64_t rows() const { return columns_.empty() ? 0 : columns_.front().second->length(); }

    std::string name_;
    Columns columns_;
    std::shared_ptr<arrow::Schema> schema_;
};

// ---------------------------------------------------------------------------
// rowid_sequence(count, layout := first|middle|last, row_id_type := int64|string|struct)
// ---------------------------------------------------------------------------

// The marker that turns a declared column into DuckDB's `rowid`. Empty value:
// the key's presence is the whole signal.
std::shared_ptr<arrow::Field> row_id_field(std::shared_ptr<arrow::DataType> type) {
    return arrow::field("row_id", std::move(type), /*nullable=*/true)
        ->WithMetadata(arrow::key_value_metadata({"is_row_id"}, {""}));
}

// Rejects rather than defaults: the fixture used to accept `layout := 'bogus'`
// and quietly emit the `first` variant, which reads as a passing test.
void require_choice(const std::string& argument, const std::string& value,
                    const std::vector<std::string>& choices) {
    if (std::find(choices.begin(), choices.end(), value) != choices.end()) return;
    std::string allowed;
    for (const auto& choice : choices) {
        if (!allowed.empty()) allowed += ", ";
        allowed += choice;
    }
    throw std::invalid_argument("rowid_sequence: " + argument + " \"" + value +
                                "\" must be one of the allowed choices: " + allowed);
}

class RowIdSequence : public vgi::TableFunction {
public:
    std::string name() const override { return "rowid_sequence"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Sequence with row_id column";
        md.categories = {"catalog"};
        md.projection_pushdown = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows"),
                vgi::ArgSpec::named("layout", "varchar", "Row ID column position"),
                vgi::ArgSpec::named("row_id_type", "varchar", "Row ID type")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        const auto layout = params.arguments.named_string("layout").value_or("first");
        const auto type = params.arguments.named_string("row_id_type").value_or("int64");
        require_choice("layout", layout, {"first", "middle", "last"});
        require_choice("row_id_type", type, {"int64", "string", "struct"});

        auto row_id = row_id_field(row_id_type(type));
        auto name = arrow::field("name", arrow::utf8(), /*nullable=*/true);
        auto value = arrow::field("value", arrow::utf8(), /*nullable=*/true);
        if (layout == "middle") return arrow::schema({name, row_id, value});
        if (layout == "last") return arrow::schema({name, value, row_id});
        return arrow::schema({row_id, name, value});
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
        const int64_t rows = std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0));
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(static_cast<size_t>(params.output_schema->num_fields()));
        for (const auto& field : params.output_schema->fields()) {
            columns.push_back(field->name() == "row_id"
                                  ? build_row_id(field->type(), rows)
                                  : string_series(field->name() == "name" ? "item_" : "val_", rows));
        }
        return std::make_unique<OneShot>(
            arrow::RecordBatch::Make(params.output_schema, rows, std::move(columns)));
    }

private:
    static std::shared_ptr<arrow::DataType> row_id_type(const std::string& name) {
        if (name == "string") return arrow::utf8();
        if (name == "struct") {
            return arrow::struct_({arrow::field("a", arrow::int64(), /*nullable=*/true),
                                   arrow::field("b", arrow::utf8(), /*nullable=*/true)});
        }
        return arrow::int64();
    }

    static std::shared_ptr<arrow::Array> string_series(const std::string& prefix, int64_t rows) {
        std::vector<std::string> values;
        values.reserve(static_cast<size_t>(rows));
        for (int64_t i = 0; i < rows; ++i) values.push_back(prefix + std::to_string(i));
        return string_column(std::move(values));
    }

    // Built from the *bound* type, never from the declared shape.
    //
    // DuckDB pushes projection down into a struct, so a row_id declared
    // `struct<a, b>` arrives here as `struct<b>` when only `rowid.b` is
    // selected. A fixed-shape StructArray would disagree with its own type,
    // which Arrow does not check — it crashes whatever reads the batch next.
    static std::shared_ptr<arrow::Array> build_row_id(
        const std::shared_ptr<arrow::DataType>& type, int64_t rows) {
        if (type->id() == arrow::Type::STRING) return string_series("rid_", rows);
        if (type->id() == arrow::Type::STRUCT) {
            std::vector<std::shared_ptr<arrow::Array>> children;
            children.reserve(type->fields().size());
            for (const auto& child : type->fields()) {
                children.push_back(child->name() == "a" ? index_column(rows)
                                                        : string_series("s_", rows));
            }
            return std::make_shared<arrow::StructArray>(type, rows, std::move(children));
        }
        return index_column(rows);
    }

    static std::shared_ptr<arrow::Array> index_column(int64_t rows) {
        std::vector<int64_t> values;
        values.reserve(static_cast<size_t>(rows));
        for (int64_t i = 0; i < rows; ++i) values.push_back(i);
        return int64_column(std::move(values));
    }
};

// ---------------------------------------------------------------------------
// filter_echo_table_scan()
// ---------------------------------------------------------------------------

// The catalog-table twin of `filter_echo`: no arguments, so it can back a
// table, and a fixed 100 rows. It exists because a filter pushed *through a
// view* has to be observed at the table, where `filter_echo` can only be
// called directly.
class FilterEchoTableScan : public vgi::TableFunction {
public:
    std::string name() const override { return "filter_echo_table_scan"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Catalog table that echoes pushed-down filters";
        md.categories = {"catalog", "filter"};
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        // Advertising pushdown without honouring it returns rows the WHERE
        // excluded; this fixture only reports, so the framework applies.
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true),
                              arrow::field("s", arrow::utf8(), /*nullable=*/true),
                              arrow::field("pushed_filters", arrow::utf8(), /*nullable=*/true)});
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override {
        return {kRows, kRows};
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto pushed = params.pushdown_filters.format();
        std::vector<int64_t> ns;
        std::vector<std::string> ss;
        std::vector<std::string> echoed;
        for (int64_t i = 0; i < kRows; ++i) {
            ns.push_back(i);
            ss.push_back("row_" + std::to_string(i));
            echoed.push_back(pushed);
        }
        const Columns columns = {{"n", int64_column(std::move(ns))},
                                 {"s", string_column(std::move(ss))},
                                 {"pushed_filters", string_column(std::move(echoed))}};
        return std::make_unique<OneShot>(project(params.output_schema, columns, kRows));
    }

private:
    static constexpr int64_t kRows = 100;
};

// ---------------------------------------------------------------------------
// versioned_constraints_scan(version)
// ---------------------------------------------------------------------------

// Both schema and rows change per version, which is what makes
// `SELECT email FROM … AT (VERSION => 1)` a binder error rather than a NULL.
Columns constraints_version(int64_t version) {
    if (version == 1) {
        return {{"id", int64_column({1, 2})}, {"name", string_column({"Alice", "Bob"})}};
    }
    Columns columns = {{"id", int64_column({1, 2, 3})},
                       {"name", string_column({"Alice", "Bob", "Carol"})},
                       {"email", string_column({"a@co", "b@co", "c@co"})}};
    // Version 3 is the fallback: an unknown version has already been rejected
    // by the catalog, so reaching here means "current".
    if (version != 2) columns.push_back({"department_id", int64_column({1, 2, 1})});
    return columns;
}

std::shared_ptr<arrow::Schema> columns_schema(const Columns& columns) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(columns.size());
    for (const auto& [column, array] : columns) {
        fields.push_back(arrow::field(column, array->type(), /*nullable=*/true));
    }
    return arrow::schema(std::move(fields));
}

class VersionedConstraintsScan : public vgi::TableFunction {
public:
    std::string name() const override { return "versioned_constraints_scan"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Versioned constraints scan (time travel)";
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("version", 0, "int64", "Data version to return")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams& params) const override {
        return columns_schema(constraints_version(params.arguments.const_int64(0).value_or(3)));
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams& params) const override {
        const auto columns = constraints_version(params.arguments.const_int64(0).value_or(3));
        const int64_t rows = columns.front().second->length();
        return {rows, rows};
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto columns = constraints_version(params.arguments.const_int64(0).value_or(3));
        return std::make_unique<OneShot>(
            project(params.output_schema, columns, columns.front().second->length()));
    }
};

// ---------------------------------------------------------------------------
// tt_pushdown_scan() / tt_pushdown_cols_scan(version)
// ---------------------------------------------------------------------------

// Version 2 is current; version 1 is a strict prefix of it (ids 1..5 of 1..10).
constexpr int64_t kCurrentTtVersion = 2;

// The two signals in one row: which version was scanned, and what the engine
// pushed into it. Asserting them together is what rules out "right version,
// filter lost" and its mirror image.
Columns tt_rows(int64_t version, const std::string& pushed) {
    const int64_t count = version == 1 ? 5 : 10;
    std::vector<int64_t> ids;
    std::vector<int64_t> vals;
    std::vector<int64_t> seen;
    std::vector<std::string> filters;
    for (int64_t id = 1; id <= count; ++id) {
        ids.push_back(id);
        vals.push_back(id * 10);
        seen.push_back(version);
        filters.push_back(pushed);
    }
    return {{"id", int64_column(std::move(ids))},
            {"val", int64_column(std::move(vals))},
            {"seen_version", int64_column(std::move(seen))},
            {"pushed_filters", string_column(std::move(filters))}};
}

std::shared_ptr<arrow::Schema> tt_schema() {
    return arrow::schema({arrow::field("id", arrow::int64(), /*nullable=*/true),
                          arrow::field("val", arrow::int64(), /*nullable=*/true),
                          arrow::field("seen_version", arrow::int64(), /*nullable=*/true),
                          arrow::field("pushed_filters", arrow::utf8(), /*nullable=*/true)});
}

vgi::FunctionMetadata tt_metadata(std::string description) {
    vgi::FunctionMetadata md;
    md.description = std::move(description);
    md.categories = {"generator", "diagnostic", "testing"};
    md.projection_pushdown = true;
    md.filter_pushdown = true;
    md.auto_apply_filters = true;
    return md;
}

// Function-backed: the catalog declares no versions, so the AT clause reaches
// the scan itself and this is the only arm that has to read it.
class TtPushdownScan : public vgi::TableFunction {
public:
    std::string name() const override { return "tt_pushdown_scan"; }

    vgi::FunctionMetadata metadata() const override {
        return tt_metadata("Function-backed time-travel + filter-pushdown scan (reads AT at init).");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return tt_schema();
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto rows = tt_rows(requested_version(params), params.pushdown_filters.format());
        return std::make_unique<OneShot>(
            project(params.output_schema, rows, rows.front().second->length()));
    }

private:
    // The version an AT clause asks for, or the current one when there is no
    // clause. A timestamp resolves by year, matching the years the
    // catalog-declared arm stamps on its versions — `2020` is v1's.
    static int64_t requested_version(const vgi::ProcessParams& params) {
        if (!params.at_unit || !params.at_value) return kCurrentTtVersion;
        std::string unit = *params.at_unit;
        for (auto& c : unit) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        try {
            if (unit == "VERSION") return std::stoll(*params.at_value);
            if (unit == "TIMESTAMP" && params.at_value->size() >= 4) {
                return std::stoi(params.at_value->substr(0, 4)) <= 2020 ? 1 : kCurrentTtVersion;
            }
        } catch (const std::exception&) {
        }
        return kCurrentTtVersion;
    }
};

// Columns-based: the catalog resolves AT to a version and passes it here, so
// this arm works through the same mechanism `versioned_data` uses.
class TtPushdownColsScan : public vgi::TableFunction {
public:
    std::string name() const override { return "tt_pushdown_cols_scan"; }

    vgi::FunctionMetadata metadata() const override {
        return tt_metadata("Columns-based time-travel + filter-pushdown scan (version via arg).");
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("version", 0, "int64", "Resolved data version")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return tt_schema();
    }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        const auto version = params.arguments.const_int64(0).value_or(kCurrentTtVersion);
        const auto rows = tt_rows(version, params.pushdown_filters.format());
        return std::make_unique<OneShot>(
            project(params.output_schema, rows, rows.front().second->length()));
    }
};

// ---------------------------------------------------------------------------
// late_materialization(count, batch_size, dup_row_id, null_ord_stride)
// ---------------------------------------------------------------------------

// Odd, and coprime with any plausible row count, so the ordering key scatters:
// a Top-N on `ord` then picks survivors spread across the whole rowid range,
// which is what drives the exact IN-list pushdown rather than a bare min/max.
constexpr int64_t kScramble = 2654435761;

// What the worker was told about `row_id`, as a string a test can match.
//
// The witness is the only way to see the rowid filter from SQL: the rewrite's
// output comes from the wide scan, so selecting it reports what reached *that*
// scan, over pipe and HTTP alike.
std::string rowid_witness(const vgi::PushdownFilters& filters) {
    int64_t in_count = 0;
    if (auto values = filters.column_values("row_id")) in_count = values->length();
    const auto bounds = filters.column_bounds("row_id");
    std::string range = "none";
    if (bounds.min || bounds.max) {
        range = (bounds.min ? std::to_string(*bounds.min) : std::string("none")) + ".." +
                (bounds.max ? std::to_string(*bounds.max) : std::string("none"));
    }
    return "rid:in=" + std::to_string(in_count) + ";rng=" + range;
}

// `Arguments` has no boolean accessor, so the one-row array is read directly.
bool named_bool(const vgi::Arguments& arguments, const std::string& name) {
    auto array = arguments.named(name);
    if (!array || array->length() == 0 || array->IsNull(0)) return false;
    const auto* values = dynamic_cast<const arrow::BooleanArray*>(array.get());
    return values != nullptr && values->Value(0);
}

class LateMaterialization : public vgi::TableFunction {
public:
    std::string name() const override { return "late_materialization"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Rowid generator that participates in late materialization";
        md.categories = {"generator", "diagnostic"};
        md.projection_pushdown = true;
        md.filter_pushdown = true;
        md.auto_apply_filters = true;
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::constant_arg("count", 0, "int64", "Number of rows to generate"),
                vgi::ArgSpec::named("batch_size", "int64", "Batch size for output"),
                vgi::ArgSpec::named("dup_row_id", "boolean",
                                    "Emit a deliberately non-unique row_id (index // 2)"),
                vgi::ArgSpec::named("null_ord_stride", "int64",
                                    "Emit NULL ord every Nth row (0 = never)")};
    }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return arrow::schema({row_id_field(arrow::int64()),
                              arrow::field("ord", arrow::int64(), /*nullable=*/true),
                              arrow::field("payload", arrow::utf8(), /*nullable=*/true),
                              arrow::field("pushed", arrow::utf8(), /*nullable=*/true)});
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
        Shape shape;
        shape.duplicate_row_ids = named_bool(params.arguments, "dup_row_id");
        shape.null_ord_stride =
            std::max<int64_t>(0, params.arguments.named_int64("null_ord_stride").value_or(0));
        return std::make_unique<Producer>(
            params.output_schema,
            std::max<int64_t>(0, params.arguments.const_int64(0).value_or(0)),
            std::max<int64_t>(1, params.arguments.named_int64("batch_size").value_or(2048)),
            shape, rowid_witness(params.pushdown_filters));
    }

private:
    struct Shape {
        bool duplicate_row_ids = false;
        int64_t null_ord_stride = 0;
    };

    class Producer : public vgi::TableProducer {
    public:
        Producer(std::shared_ptr<arrow::Schema> schema, int64_t rows, int64_t batch_size,
                 Shape shape, std::string witness)
            : schema_(std::move(schema)),
              remaining_(rows),
              batch_size_(batch_size),
              shape_(shape),
              witness_(std::move(witness)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            if (remaining_ <= 0) return nullptr;
            const int64_t n = std::min(remaining_, batch_size_);
            const int64_t start = cursor_;

            std::vector<std::shared_ptr<arrow::Array>> columns;
            columns.reserve(static_cast<size_t>(schema_->num_fields()));
            for (const auto& field : schema_->fields()) {
                columns.push_back(build(field->name(), start, n));
            }
            cursor_ += n;
            remaining_ -= n;
            return arrow::RecordBatch::Make(schema_, n, std::move(columns));
        }

    private:
        std::shared_ptr<arrow::Array> build(const std::string& column, int64_t start,
                                            int64_t n) const {
            if (column == "row_id") {
                std::vector<int64_t> values;
                values.reserve(static_cast<size_t>(n));
                for (int64_t i = start; i < start + n; ++i) {
                    values.push_back(shape_.duplicate_row_ids ? i / 2 : i);
                }
                return int64_column(std::move(values));
            }
            if (column == "ord") {
                arrow::Int64Builder builder;
                (void)builder.Reserve(n);
                for (int64_t i = start; i < start + n; ++i) {
                    if (shape_.null_ord_stride > 0 && i % shape_.null_ord_stride == 0) {
                        (void)builder.AppendNull();
                    } else {
                        (void)builder.Append((i * kScramble) % 1000000007);
                    }
                }
                std::shared_ptr<arrow::Array> array;
                (void)builder.Finish(&array);
                return array;
            }
            std::vector<std::string> values;
            values.reserve(static_cast<size_t>(n));
            for (int64_t i = start; i < start + n; ++i) {
                values.push_back(column == "payload" ? "payload_" + std::to_string(i) : witness_);
            }
            return string_column(std::move(values));
        }

        std::shared_ptr<arrow::Schema> schema_;
        int64_t remaining_;
        int64_t batch_size_;
        Shape shape_;
        std::string witness_;
        int64_t cursor_ = 0;
    };
};

}  // namespace

void register_static_scans(vgi::Worker& worker) {
    // The reference tables the constraint, comment, default and statistics
    // tests read. Values are pinned by those tests, not chosen here.
    worker.register_table(std::make_shared<StaticScan>(
        "departments_scan",
        Columns{{"id", int64_column({1, 2, 3})},
                {"name", string_column({"Engineering", "Sales", "HR"})},
                {"budget", double_column({500000.0, 300000.0, 200000.0})}}));
    worker.register_table(std::make_shared<StaticScan>(
        "products_scan",
        Columns{{"id", int64_column({1, 2, 3})},
                {"name", string_column({"Widget", "Gadget", "Doohickey"})},
                {"quantity", int64_column({100, 50, 200})},
                {"price", double_column({9.99, 24.99, 4.99})}}));
    worker.register_table(std::make_shared<StaticScan>(
        "employees_scan",
        Columns{{"id", int64_column({1, 2, 3, 4, 5})},
                {"name", string_column({"Alice", "Bob", "Carol", "Dave", "Eve"})},
                {"email", string_column({"alice@co.com", "bob@co.com", "carol@co.com",
                                         "dave@co.com", "eve@co.com"})},
                {"department_id", int64_column({1, 1, 2, 2, 3})}}));
    worker.register_table(std::make_shared<StaticScan>(
        "projects_scan",
        Columns{{"department_id", int64_column({1, 1, 2})},
                {"project_code", string_column({"P001", "P002", "P003"})},
                {"title", string_column({"Backend API", "Frontend UI", "Sales Portal"})}}));
    // Ordered by id, not by the ENUM ordinal the statistics are derived from:
    // the two disagreeing is exactly what `column_statistics.test` checks.
    worker.register_table(std::make_shared<StaticScan>(
        "colors_scan", Columns{{"id", int64_column({1, 2, 3})},
                               {"color", string_column({"blue", "green", "red"})},
                               {"hex_code", string_column({"#0000FF", "#00FF00", "#FF0000"})}}));

    // The `versioned_tables` catalog: one table per version, so the visible
    // table *set* changes with the resolved data version rather than the rows.
    worker.register_table(std::make_shared<StaticScan>(
        "versioned_tables_animals_scan",
        Columns{{"name", string_column({"chicken", "cow", "horse", "pig", "sheep"})},
                {"legs", int64_column({2, 4, 4, 4, 4})},
                {"sound", string_column({"cluck", "moo", "neigh", "oink", "baa"})}}));
    worker.register_table(std::make_shared<StaticScan>(
        "versioned_tables_animals_color_scan",
        Columns{{"name", string_column({"chicken", "cow", "horse", "pig", "sheep"})},
                {"legs", int64_column({2, 4, 4, 4, 4})},
                {"sound", string_column({"cluck", "moo", "neigh", "oink", "baa"})},
                {"color", string_column({"red", "brown", "black", "pink", "white"})}}));
    worker.register_table(std::make_shared<StaticScan>(
        "versioned_tables_plants_scan",
        Columns{{"name", string_column({"oak", "pine", "rose", "tomato", "wheat"})},
                {"kind", string_column({"tree", "tree", "flower", "vegetable", "grass"})},
                {"height_m", double_column({20.0, 25.0, 0.6, 1.5, 1.0})}}));

    worker.register_table(std::make_shared<RowIdSequence>());
    worker.register_table(std::make_shared<FilterEchoTableScan>());
    worker.register_table(std::make_shared<VersionedConstraintsScan>());
    worker.register_table(std::make_shared<TtPushdownScan>());
    worker.register_table(std::make_shared<TtPushdownColsScan>());
    worker.register_table(std::make_shared<LateMaterialization>());
}

}  // namespace example
