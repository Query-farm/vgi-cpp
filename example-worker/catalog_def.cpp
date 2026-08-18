// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// The declarative example catalog: schemas, tables and views.
//
// A VGI table is a name bound to a table function. Declaring one here is what
// turns `SELECT * FROM ex.data.cacheable_numbers` into a call to the
// `cacheable_numbers` function — the table has no storage of its own.

#include <string>
#include <vector>

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <vgi/worker.h>

#include "registry.h"

namespace example {
namespace {

std::shared_ptr<arrow::Schema> columns(
    std::vector<std::pair<std::string, std::shared_ptr<arrow::DataType>>> fields) {
    std::vector<std::shared_ptr<arrow::Field>> built;
    built.reserve(fields.size());
    for (auto& [name, type] : fields) {
        built.push_back(arrow::field(name, std::move(type), /*nullable=*/true));
    }
    return arrow::schema(std::move(built));
}

vgi::CatalogTable backed_by(std::string name, std::string scan_function,
                            std::shared_ptr<arrow::Schema> schema, std::string comment = {}) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.scan_function = std::move(scan_function);
    table.columns = std::move(schema);
    table.comment = std::move(comment);
    return table;
}

// A one-element array holding `value`, for a scan argument.
std::shared_ptr<arrow::Array> int64_arg(int64_t value) {
    arrow::Int64Builder builder;
    (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// A branch that scans `sequence(count)`.
vgi::CatalogBranch sequence_branch(int64_t count,
                                   std::optional<std::string> branch_filter = std::nullopt) {
    vgi::CatalogBranch branch;
    branch.function_name = "sequence";
    branch.scan_arguments = vgi::serialize_scan_arguments({int64_arg(count)});
    branch.branch_filter = std::move(branch_filter);
    return branch;
}

vgi::CatalogTable multi_branch(std::string name, std::vector<vgi::CatalogBranch> branches) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.columns = columns({{"n", arrow::int64()}});
    // No single scan function: the branches are the sources, and declaring one
    // here as well would be a second, contradictory answer.
    table.branches = std::move(branches);
    table.inline_scan = false;
    return table;
}

vgi::CatalogTable versioned_table(std::string name, std::string scan_function,
                                  std::shared_ptr<arrow::Schema> schema, std::string comment) {
    auto table = backed_by(std::move(name), std::move(scan_function), std::move(schema),
                           std::move(comment));
    // Not inlined: the engine asks for the scan separately, and only then does
    // the request carry the attachment whose resolved version decides which
    // definition of the table it is asking about.
    table.inline_scan = false;
    return table;
}

// Exact bounds for a `sequence(count)`-backed column: [0, count-1], no nulls,
// `count` distinct values. Declaring them is what lets the optimizer fold an
// out-of-range filter to an empty result without calling the scan.
vgi::ColumnStatistics sequence_statistics(std::string column, int64_t count) {
    vgi::ColumnStatistics stat;
    stat.column_name = std::move(column);
    stat.min = vgi::StatValue::integer(0);
    stat.max = vgi::StatValue::integer(count - 1);
    stat.distinct_count = count;
    return stat;
}

// A column carrying metadata: a default, a comment, a generated expression, or
// the row_id marker. Metadata is how all four travel — the wire has no field
// for any of them, only the column schema.
std::shared_ptr<arrow::Field> tagged(std::string name,
                                     std::shared_ptr<arrow::DataType> type,
                                     std::vector<std::string> keys,
                                     std::vector<std::string> values) {
    return arrow::field(std::move(name), std::move(type), /*nullable=*/true)
        ->WithMetadata(arrow::key_value_metadata(std::move(keys), std::move(values)));
}

// One column's bounds. `has_null=false, has_not_null=true` are the defaults and
// describe every column here but `products.quantity`.
vgi::ColumnStatistics stat(std::string column, vgi::StatValue min, vgi::StatValue max,
                           int64_t distinct) {
    vgi::ColumnStatistics statistics;
    statistics.column_name = std::move(column);
    statistics.min = std::move(min);
    statistics.max = std::move(max);
    statistics.distinct_count = distinct;
    return statistics;
}

// A string column's bounds, plus the two facts DuckDB tracks only for strings.
vgi::ColumnStatistics string_stat(std::string column, std::string min, std::string max,
                                  int64_t distinct, uint64_t max_length) {
    auto statistics = stat(std::move(column), vgi::StatValue::text(std::move(min)),
                           vgi::StatValue::text(std::move(max)), distinct);
    statistics.contains_unicode = false;
    statistics.max_string_length = max_length;
    return statistics;
}

// Little-endian WKB for a 2D point: byte order 1, geometry type 1, then x, y.
// The engine reinterprets these bytes as GEOMETRY, which is how a bounding box
// travels as a pair of column bounds.
std::string wkb_point(double x, double y) {
    std::string wkb("\x01\x01\x00\x00\x00", 5);
    wkb.append(reinterpret_cast<const char*>(&x), sizeof(x));
    wkb.append(reinterpret_cast<const char*>(&y), sizeof(y));
    return wkb;
}

// A one-element array holding `value`, for a named scan argument.
std::shared_ptr<arrow::Array> string_arg(const std::string& value) {
    arrow::StringBuilder builder;
    (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> bool_arg(bool value) {
    arrow::BooleanBuilder builder;
    (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// A rowid fixture: 20 rows of `rowid_sequence`, with the row_id column at the
// declared index. The layout argument and the column order have to agree —
// the table's declared columns are what the planner types against.
vgi::CatalogTable rowid_table(std::string name,
                              std::vector<std::shared_ptr<arrow::Field>> fields,
                              const std::string& layout, const std::string& row_id_type,
                              std::string comment) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.columns = arrow::schema(std::move(fields));
    table.scan_function = "rowid_sequence";
    table.scan_arguments = vgi::serialize_scan_arguments(
        {int64_arg(20)},
        {{"layout", string_arg(layout)}, {"row_id_type", string_arg(row_id_type)}});
    table.cardinality = 20;
    table.comment = std::move(comment);
    return table;
}

// A late-materialization fixture: 1000 rows of `late_materialization`, inlined
// so the Top-N → SEMI rewrite sees the rowid scan without a second round trip.
vgi::CatalogTable late_mat_table(
    std::string name, std::string comment,
    std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>> named) {
    vgi::CatalogTable table;
    table.name = std::move(name);
    table.columns = arrow::schema({
        tagged("row_id", arrow::int64(), {"is_row_id"}, {""}),
        arrow::field("ord", arrow::int64(), /*nullable=*/true),
        arrow::field("payload", arrow::utf8(), /*nullable=*/true),
        arrow::field("pushed", arrow::utf8(), /*nullable=*/true),
    });
    table.scan_function = "late_materialization";
    table.scan_arguments = vgi::serialize_scan_arguments({int64_arg(1000)}, std::move(named));
    table.cardinality = 1000;
    table.comment = std::move(comment);
    return table;
}

}  // namespace

// The `versioned_tables` catalog: a table set that changes with the resolved
// data version, which is what makes `version_schemas` observable from SQL.
void declare_versioned_tables(vgi::CatalogModel& catalog) {
    catalog.implementation_version = "11.0.0";
    catalog.data_version_spec = ">=1.0.0,<4.0.0";
    catalog.supported_data_versions = {"1.0.0", "1.1.0", "2.0.0", "3.0.0"};
    catalog.default_data_version = "3.0.0";
    catalog.supported_implementation_versions = {"10.0.0", "10.1.0", "11.0.0"};
    catalog.npm_version_resolution = true;
    catalog.comment = "Catalog whose visible tables depend on the resolved data version";

    const auto animals = versioned_table(
        "animals", "versioned_tables_animals_scan",
        columns({{"name", arrow::utf8()}, {"legs", arrow::int64()}, {"sound", arrow::utf8()}}),
        "Animals table for data_version 1.0.0");
    const auto animals_color = versioned_table(
        "animals", "versioned_tables_animals_color_scan",
        columns({{"name", arrow::utf8()},
                 {"legs", arrow::int64()},
                 {"sound", arrow::utf8()},
                 {"color", arrow::utf8()}}),
        "Animals table for data_version 1.1.0 (with color)");
    const auto plants = versioned_table(
        "plants", "versioned_tables_plants_scan",
        columns({{"name", arrow::utf8()},
                 {"kind", arrow::utf8()},
                 {"height_m", arrow::float64()}}),
        "Plants table for data_version 2.0.0 and 3.0.0");

    const auto main_with = [](std::vector<vgi::CatalogTable> tables) {
        vgi::CatalogSchema schema;
        schema.name = "main";
        schema.tables = std::move(tables);
        return std::vector<vgi::CatalogSchema>{std::move(schema)};
    };
    catalog.version_schemas = {
        {"1.0.0", main_with({animals})},
        {"1.1.0", main_with({animals_color})},
        {"2.0.0", main_with({animals, plants})},
        {"3.0.0", main_with({plants})},
    };
    // Declared empty as well: the engine asks for the schema list before it
    // has an attachment to resolve a version against.
    catalog.schema("main");
}

void declare_catalog(vgi::Worker& worker) {
    // Settings the catalog introduces. `SET greeting = 'Bonjour'` only works
    // because they are declared here; a function then names the ones it reads
    // in `required_settings` to have the values forwarded.
    worker.catalog().settings = {
        {"vgi_verbose_mode", "Enable verbose output", arrow::boolean()},
        {"greeting", "Greeting prefix", arrow::utf8()},
        {"multiplier", "Integer multiplier", arrow::int64()},
        {"threshold", "Row-filter threshold", arrow::int64()},
        {"scale_factor", "Floating-point scale factor", arrow::float64()},
        // A struct setting, not a string: the whole point of `config` is that
        // a setting's type can be composite, and `struct_settings` reads its
        // three fields rather than parsing them out of text.
        {"config", "Sequence configuration",
         arrow::struct_({arrow::field("start", arrow::int64(), true),
                         arrow::field("step", arrow::int64(), true),
                         arrow::field("label", arrow::utf8(), true)})},
    };

    // A field marked `redact` is one the engine must keep out of logs and
    // error text; declaring the schema is what carries that marker across.
    const auto redacted = arrow::key_value_metadata({"redact"}, {"true"});
    worker.catalog().secret_types = {
        {"vgi_example", "Example VGI secret for testing",
         arrow::schema({
             arrow::field("secret_string", arrow::utf8(), true)->WithMetadata(redacted),
             arrow::field("api_key", arrow::utf8(), true)->WithMetadata(redacted),
             arrow::field("port", arrow::int32(), true),
             arrow::field("use_ssl", arrow::boolean(), true),
             arrow::field("timeout", arrow::float64(), true),
         })},
    };

    auto& data = worker.catalog().schema("data");

    // The column is `value` while the scan emits `n`: the engine maps a
    // function-backed table's columns positionally, and the catalog's name is
    // the one SQL sees.
    {
        auto numbers = backed_by("numbers", "sequence", columns({{"value", arrow::int64()}}),
                                 "First 100 integers (demonstrates explicit columns)");
        numbers.scan_arguments = vgi::serialize_scan_arguments({int64_arg(100)});
        numbers.cardinality = 100;
        numbers.column_statistics = {sequence_statistics("value", 100)};
        data.tables.push_back(std::move(numbers));
    }
    data.tables.push_back(backed_by("cacheable_numbers", "cacheable_numbers",
                                    columns({{"n", arrow::int64()}}),
                                    "Cacheable 10-row result advertising vgi.cache.ttl"));
    data.tables.push_back(
        backed_by("cache_nonce", "cache_nonce", columns({{"nonce", arrow::int64()}}),
                  "One-row cacheable result whose value changes per real invocation"));
    data.tables.push_back(
        backed_by("cache_no_store", "cache_no_store", columns({{"n", arrow::int64()}}),
                  "Advertises vgi.cache.no_store — must never be cached"));
    data.tables.push_back(
        backed_by("cache_scoped_txn", "cache_scoped_txn", columns({{"n", arrow::int64()}}),
                  "Advertises vgi.cache.scope=transaction"));
    // Uncommented on purpose: `cache_bench` is a scaling fixture called
    // directly, and `table/comments.test` pins the commented set exactly.
    data.tables.push_back(
        backed_by("cache_bench", "cache_bench", columns({{"v", arrow::int64()}})));
    data.tables.push_back(backed_by("cache_multicol", "cache_multicol",
                                    columns({{"a", arrow::int64()},
                                             {"b", arrow::int64()},
                                             {"c", arrow::int64()}}),
                                    "Multi-column cacheable result (projection-coverage reuse)"));
    data.tables.push_back(
        backed_by("cache_big", "cache_big", columns({{"n", arrow::int64()}}),
                  "Large multi-batch cacheable result (advertises vgi.cache.ttl)"));
    data.tables.push_back(
        backed_by("cache_revalidatable", "cache_revalidatable",
                  columns({{"nonce", arrow::int64()}}),
                  "Always-revalidate result (304 not_modified reuses stored bytes)"));
    data.tables.push_back(
        backed_by("cache_whoami", "cache_whoami", columns({{"who", arrow::utf8()}}),
                  "Cacheable result echoing the caller's auth principal (identity-scoped)"));
    data.tables.push_back(
        backed_by("ten_thousand_table", "ten_thousand_table", columns({{"n", arrow::int64()}}),
                  "Function-backed table over the no-arg ten_thousand function"));
    data.tables.push_back(
        backed_by("cache_parallel", "cache_parallel", columns({{"v", arrow::int64()}})));

    // Multi-branch tables: one logical relation stitched from several scans.
    data.tables.push_back(
        multi_branch("multi_branch_numbers", {sequence_branch(50), sequence_branch(50)}));
    data.tables.push_back(multi_branch("multi_branch_filtered_numbers",
                                       {sequence_branch(100, "n < 50"),
                                        sequence_branch(100, "n >= 50")}));
    data.tables.push_back(multi_branch("multi_branch_empty", {}));

    data.tables.push_back(
        backed_by("cache_poison", "cache_poison", columns({{"n", arrow::int64()}}),
                  "Cacheable first batch then a mid-stream error (never-partial check)"));
    data.tables.push_back(
        backed_by("cache_filtered", "cache_filtered", columns({{"n", arrow::int64()}}),
                  "Cacheable sequence with static filter pushdown (filter_bytes keying)"));
    data.tables.push_back(
        backed_by("cache_ordered", "cache_ordered", columns({{"n", arrow::int64()}}),
                  "Multi-worker order-sensitive cacheable result (batch_index; parallel "
                  "capture, ordered serve)"));
    data.tables.push_back(
        backed_by("cache_external_fail", "cache_external_fail", columns({{"n", arrow::int64()}}),
                  "Cacheable first batch then an unresolvable external-location pointer"));
    for (const char* name : {"cache_partitioned", "cache_partition_scope",
                             "cache_partition_parallel"}) {
        data.tables.push_back(backed_by(name, name, columns({{"n", arrow::int64()}})));
    }

    // Time travel is the point: each version resolves to its own scan
    // arguments, so an `AT` clause keys a distinct cache entry.
    {
        vgi::CatalogTable versioned;
        versioned.name = "cache_versioned";
        versioned.columns = columns({{"v", arrow::int64()}});
        versioned.scan_function = "cache_versioned_scan";
        versioned.scan_arguments = vgi::serialize_scan_arguments({int64_arg(3)});
        versioned.comment = "Version-specific cacheable rows (AT-keyed cache isolation)";
        for (int64_t version = 1; version <= 3; ++version) {
            vgi::TimeTravelVersion entry;
            entry.version = version;
            entry.columns = versioned.columns;
            entry.scan_function = "cache_versioned_scan";
            entry.scan_arguments = vgi::serialize_scan_arguments({int64_arg(version)});
            versioned.time_travel.push_back(std::move(entry));
        }
        data.tables.push_back(std::move(versioned));
    }
    // A table whose scan is `sequence` with a fixed argument, and two views
    // over it. Declared here rather than as functions because that is what
    // they are to the engine — a name it resolves, not a call it makes.
    // Declares no statistics of its own: the optimizer has to fall back to
    // the scan function's, which is exactly what this table is here to pin.
    {
        auto funny = backed_by("funny_numbers", "sequence", columns({{"n", arrow::int64()}}),
                               "123456 integers; stats served by the sequence function, "
                               "not the table");
        funny.scan_arguments = vgi::serialize_scan_arguments({int64_arg(123456)});
        funny.cardinality = 123456;
        data.tables.push_back(std::move(funny));
    }

    {
        auto large = backed_by("large_sequence", "sequence", columns({{"n", arrow::int64()}}));
        large.scan_arguments = vgi::serialize_scan_arguments({int64_arg(1000000)});
        large.cardinality = 1000000;
        data.tables.push_back(std::move(large));
    }

    {
        // Inlined, and that is what makes AT work here: `catalog_table_get`
        // already carries the AT clause, so the record it returns can name the
        // version's own scan arguments. Left un-inlined the engine never asked
        // for the scan separately, and every AT ran version 3.
        vgi::CatalogTable versioned;
        versioned.name = "versioned_data";
        versioned.time_travel = versioned_data_versions();
        versioned.columns = versioned.time_travel.back().columns;
        versioned.scan_function = "versioned_data_scan";
        versioned.scan_arguments = versioned.time_travel.back().scan_arguments;
        versioned.inline_scan = true;
        versioned.comment =
            "Versioned data table demonstrating time travel with schema evolution";
        data.tables.push_back(std::move(versioned));
    }

    for (auto& table : rff_tables()) data.tables.push_back(std::move(table));

    // In `main`, not `data`: a view is looked up by the schema the user names,
    // and the suite names `main` for these two.
    auto& main = worker.catalog().schema("main");
    main.views.push_back(
        {"first_ten", "SELECT * FROM sequence(10)", "First 10 integers"});
    main.views.push_back({"even_numbers", "SELECT * FROM sequence(100) WHERE n % 2 = 0",
                          "Even numbers from 0 to 98"});
    data.views.push_back(
        {"small_numbers", "SELECT * FROM sequence(5)", std::nullopt});

    data.tables.push_back(
        backed_by("cache_projection", "cache_projection",
                  columns({{"a", arrow::int64()},
                           {"b", arrow::int64()},
                           {"c", arrow::int64()}}),
                  "Projection-pushdown cacheable result (SELECT a vs b are distinct keys)"));

    // The constraint / reference tables. Their rows are almost incidental:
    // what the suite reads from them is metadata — comments, defaults,
    // column statistics — so the declarations here are the fixture.
    {
        auto departments = backed_by(
            "departments", "departments_scan",
            arrow::schema({arrow::field("id", arrow::int64(), /*nullable=*/true),
                           arrow::field("name", arrow::utf8(), /*nullable=*/true),
                           tagged("budget", arrow::float64(), {"default"}, {"0"})}),
            "Department reference table");
        departments.cardinality = 3;
        // Bounds wider than the three rows on purpose: they describe the table
        // the fixture stands in for, and `column_statistics.test` folds
        // `id > 100` to EMPTY_RESULT against them, not against the data.
        departments.column_statistics = {
            stat("id", vgi::StatValue::integer(1), vgi::StatValue::integer(10), 10),
            string_stat("name", "Accounting", "Sales", 10, 20),
            stat("budget", vgi::StatValue::floating(50000.0),
                 vgi::StatValue::floating(500000.0), 10),
        };
        data.tables.push_back(std::move(departments));
    }

    {
        auto products = backed_by(
            "products", "products_scan",
            arrow::schema({
                tagged("id", arrow::int64(), {"comment"}, {"Unique product identifier"}),
                tagged("name", arrow::utf8(), {"default", "comment"},
                       {"'unknown'", "Product display name"}),
                tagged("quantity", arrow::int64(), {"default"}, {"0"}),
                tagged("price", arrow::float64(), {"default", "comment"},
                       {"9.99", "Unit price in USD"}),
            }),
            "Product table with column defaults");
        products.cardinality = 3;
        products.column_statistics = {
            stat("id", vgi::StatValue::integer(1), vgi::StatValue::integer(100), 100),
            string_stat("name", "Anvil", "Zebra Tape", 100, 30),
            stat("quantity", vgi::StatValue::integer(0), vgi::StatValue::integer(10000), 50),
            stat("price", vgi::StatValue::floating(0.99), vgi::StatValue::floating(999.99), 80),
        };
        // The only nullable column in the set, which is what the optimizer is
        // told here and what `quantity IS NOT NULL` then has to check.
        products.column_statistics[2].has_null = true;
        data.tables.push_back(std::move(products));
    }

    {
        auto employees = backed_by("employees", "employees_scan",
                                   columns({{"id", arrow::int64()},
                                            {"name", arrow::utf8()},
                                            {"email", arrow::utf8()},
                                            {"department_id", arrow::int64()}}),
                                   "Employee table with FK to departments");
        employees.cardinality = 5;
        data.tables.push_back(std::move(employees));
    }

    {
        auto projects = backed_by("projects", "projects_scan",
                                  columns({{"department_id", arrow::int64()},
                                           {"project_code", arrow::utf8()},
                                           {"title", arrow::utf8()}}),
                                  "Projects with composite PK and FK to departments");
        projects.cardinality = 3;
        data.tables.push_back(std::move(projects));
    }

    {
        // Statistics derived from an ENUM, so min/max follow the ordinal order
        // (red, green, blue) while the rows are ordered by id. The two
        // disagreeing is the point: it proves the bounds are the declared
        // strings and not dictionary indices.
        auto colors = backed_by("colors", "colors_scan",
                                columns({{"id", arrow::int64()},
                                         {"color", arrow::utf8()},
                                         {"hex_code", arrow::utf8()}}),
                                "Colors table with ENUM-derived statistics");
        colors.cardinality = 3;
        colors.column_statistics = {
            stat("id", vgi::StatValue::integer(1), vgi::StatValue::integer(3), 3),
            string_stat("color", "red", "blue", 3, 5),
            string_stat("hex_code", "#0000FF", "#FF0000", 3, 7),
        };
        data.tables.push_back(std::move(colors));
    }

    {
        // No scan: the fixture exists for its statistics. The bounds are the
        // corners of a 5x5 grid of points, which the engine reads back as
        // BOX(0 0, 4 4).
        vgi::CatalogTable geo;
        geo.name = "geo_points";
        geo.columns = arrow::schema(
            {arrow::field("id", arrow::int64(), /*nullable=*/true),
             tagged("geom",
                    arrow::struct_({arrow::field("x", arrow::float64(), /*nullable=*/true),
                                    arrow::field("y", arrow::float64(), /*nullable=*/true)}),
                    {"ARROW:extension:name"}, {"geoarrow.point"})});
        geo.scan_function = "sequence";
        geo.comment = "Geometry table with spatial bounding-box statistics";
        geo.column_statistics = {
            stat("id", vgi::StatValue::integer(1), vgi::StatValue::integer(25), 25),
            stat("geom", vgi::StatValue::bytes(wkb_point(0.0, 0.0)),
                 vgi::StatValue::bytes(wkb_point(4.0, 4.0)), 25),
        };
        data.tables.push_back(std::move(geo));
    }

    {
        // The generated columns are DuckDB expressions over `n`, not columns
        // the scan emits — `sequence(10)` produces one column and the engine
        // computes the other two.
        auto generated = backed_by(
            "generated_sequence", "sequence",
            arrow::schema({arrow::field("n", arrow::int64(), /*nullable=*/true),
                           tagged("doubled", arrow::int64(), {"generated_expression"}, {"n * 2"}),
                           tagged("label", arrow::utf8(), {"generated_expression"},
                                  {"'item_' || CAST(n AS VARCHAR)"})}),
            "Table with generated columns backed by sequence(10)");
        generated.scan_arguments = vgi::serialize_scan_arguments({int64_arg(10)});
        generated.cardinality = 10;
        data.tables.push_back(std::move(generated));
    }

    // The row_id column at each index in turn, and in each type the protocol
    // allows. Position is what these pin: an engine that assumed index 0 would
    // pass `rowid_first` and fail the other two.
    data.tables.push_back(rowid_table(
        "rowid_first",
        {tagged("row_id", arrow::int64(), {"is_row_id"}, {""}),
         arrow::field("name", arrow::utf8(), /*nullable=*/true),
         arrow::field("value", arrow::utf8(), /*nullable=*/true)},
        "first", "int64", "Table with row_id at column index 0"));
    data.tables.push_back(rowid_table(
        "rowid_middle",
        {arrow::field("name", arrow::utf8(), /*nullable=*/true),
         tagged("row_id", arrow::int64(), {"is_row_id"}, {""}),
         arrow::field("value", arrow::utf8(), /*nullable=*/true)},
        "middle", "int64", "Table with row_id at column index 1"));
    data.tables.push_back(rowid_table(
        "rowid_last",
        {arrow::field("name", arrow::utf8(), /*nullable=*/true),
         arrow::field("value", arrow::utf8(), /*nullable=*/true),
         tagged("row_id", arrow::int64(), {"is_row_id"}, {""})},
        "last", "int64", "Table with row_id at column index 2"));
    data.tables.push_back(rowid_table(
        "rowid_string",
        {tagged("row_id", arrow::utf8(), {"is_row_id"}, {""}),
         arrow::field("value", arrow::utf8(), /*nullable=*/true)},
        "first", "string", "Table with string row_id"));
    data.tables.push_back(rowid_table(
        "rowid_struct",
        {tagged("row_id",
                arrow::struct_({arrow::field("a", arrow::int64(), /*nullable=*/true),
                                arrow::field("b", arrow::utf8(), /*nullable=*/true)}),
                {"is_row_id"}, {""}),
         arrow::field("value", arrow::utf8(), /*nullable=*/true)},
        "first", "struct", "Table with struct row_id"));

    data.tables.push_back(late_mat_table(
        "late_mat", "Late-materialization table (1000 rows, unique rowid)", {}));
    // Violates the uniqueness the rewrite assumes, on purpose: the engine
    // trusts the worker's opt-in, so the wrong answer has to be demonstrable.
    data.tables.push_back(late_mat_table(
        "late_mat_dup",
        "Late-materialization table with deliberately non-unique rowid (contract violation)",
        {{"dup_row_id", bool_arg(true)}}));
    data.tables.push_back(late_mat_table(
        "late_mat_nulls", "Late-materialization table with NULLs in the ord column",
        {{"null_ord_stride", int64_arg(7)}}));

    {
        // Same scan and rows as `numbers`; what differs is that its statistics
        // carry no TTL, so the engine re-fetches them every query.
        auto volatile_numbers =
            backed_by("volatile_numbers", "sequence", columns({{"value", arrow::int64()}}),
                      "Numbers with volatile stats (TTL=0, always re-fetched)");
        volatile_numbers.scan_arguments = vgi::serialize_scan_arguments({int64_arg(100)});
        volatile_numbers.cardinality = 100;
        volatile_numbers.column_statistics = {sequence_statistics("value", 100)};
        data.tables.push_back(std::move(volatile_numbers));
    }

    {
        // The same function as `ten_thousand_table`, but with the row count
        // inlined on the table record: the engine reads it from there and
        // never fires the per-bind cardinality call.
        auto inlined = backed_by("cardinality_inlined_table", "ten_thousand_table",
                                 columns({{"n", arrow::int64()}}),
                                 "Function-backed table with inlined cardinality (10000 rows)");
        inlined.cardinality = 10000;
        data.tables.push_back(std::move(inlined));
    }

    {
        // Not inlined: the backing function resolves a secret during bind, and
        // that two-phase exchange only happens on the scan-function path.
        auto secret_table = backed_by("secret_demo_table", "secret_demo",
                                      columns({{"key", arrow::utf8()},
                                               {"value", arrow::utf8()},
                                               {"arrow_type", arrow::utf8()}}),
                                      "Function-backed table over the secret-using "
                                      "secret_demo function");
        secret_table.inline_scan = false;
        data.tables.push_back(std::move(secret_table));
    }

    data.tables.push_back(backed_by(
        "filter_echo_table", "filter_echo_table_scan",
        columns({{"n", arrow::int64()}, {"s", arrow::utf8()}, {"pushed_filters", arrow::utf8()}}),
        "Catalog table echoing pushed-down filters (filter-pushdown-through-view tests)."));

    {
        // The function-backed arm of time-travel pushdown: no versions are
        // declared, so the AT clause reaches the scan itself.
        auto pushdown_fn = backed_by("tt_pushdown_fn", "tt_pushdown_scan",
                                     columns({{"id", arrow::int64()},
                                              {"val", arrow::int64()},
                                              {"seen_version", arrow::int64()},
                                              {"pushed_filters", arrow::utf8()}}),
                                     "Function-backed: prunes by filter AND time-travels "
                                     "(AT read at init).");
        data.tables.push_back(std::move(pushdown_fn));
    }

    {
        // The columns-based arm: the catalog resolves AT to a version and
        // hands it to the scan as an argument. Inlined, because
        // `catalog_table_get` is the call that carries the AT clause — asked
        // for the scan separately, the engine has nothing to resolve against
        // and every AT reads the current version.
        auto pushdown_cols = backed_by("tt_pushdown_cols", "tt_pushdown_cols_scan",
                                       columns({{"id", arrow::int64()},
                                                {"val", arrow::int64()},
                                                {"seen_version", arrow::int64()},
                                                {"pushed_filters", arrow::utf8()}}),
                                       "Columns-based: prunes by filter AND time-travels "
                                       "(AT → version arg).");
        const int cols_years[] = {2000, 2021};
        for (int64_t version = 1; version <= 2; ++version) {
            vgi::TimeTravelVersion entry;
            entry.version = version;
            entry.columns = pushdown_cols.columns;
            entry.scan_function = "tt_pushdown_cols_scan";
            entry.scan_arguments = vgi::serialize_scan_arguments({int64_arg(version)});
            entry.timestamp_year = cols_years[version - 1];
            pushdown_cols.time_travel.push_back(std::move(entry));
        }
        pushdown_cols.scan_arguments = pushdown_cols.time_travel.back().scan_arguments;
        data.tables.push_back(std::move(pushdown_cols));
    }

    {
        // Both schema and rows evolve, which is what makes `SELECT email … AT
        // (VERSION => 1)` a binder error rather than a column of NULLs.
        vgi::CatalogTable constraints;
        constraints.name = "versioned_constraints";
        constraints.scan_function = "versioned_constraints_scan";
        constraints.comment = "Table with constraints that evolve across versions";
        const std::vector<std::vector<std::pair<std::string, std::shared_ptr<arrow::DataType>>>>
            per_version = {
                {{"id", arrow::int64()}, {"name", arrow::utf8()}},
                {{"id", arrow::int64()}, {"name", arrow::utf8()}, {"email", arrow::utf8()}},
                {{"id", arrow::int64()},
                 {"name", arrow::utf8()},
                 {"email", arrow::utf8()},
                 {"department_id", arrow::int64()}},
            };
        const int constraint_years[] = {2020, 2021, 2022};
        for (int64_t version = 1; version <= 3; ++version) {
            vgi::TimeTravelVersion entry;
            entry.version = version;
            entry.columns = columns(per_version[static_cast<size_t>(version) - 1]);
            entry.scan_function = "versioned_constraints_scan";
            entry.scan_arguments = vgi::serialize_scan_arguments({int64_arg(version)});
            entry.timestamp_year = constraint_years[version - 1];
            constraints.time_travel.push_back(std::move(entry));
        }
        constraints.columns = constraints.time_travel.back().columns;
        constraints.scan_arguments = constraints.time_travel.back().scan_arguments;
        data.tables.push_back(std::move(constraints));
    }
}

}  // namespace example
