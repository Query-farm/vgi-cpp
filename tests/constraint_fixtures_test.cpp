// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>

#include "../example-worker/constraint_fixtures.h"

namespace fixture = example::constraint_fixtures;

namespace {
template <class Builder, class T>
std::shared_ptr<arrow::Array> array(std::initializer_list<std::optional<T>> values) {
    Builder builder;
    for (const auto& value : values) {
        auto status = value ? builder.Append(*value) : builder.AppendNull();
        REQUIRE(status.ok());
    }
    return builder.Finish().ValueOrDie();
}
void replace(fixture::Columns& rows, const std::string& name, std::shared_ptr<arrow::Array> value) {
    for (auto& entry : rows)
        if (entry.first == name) {
            entry.second = std::move(value);
            return;
        }
    FAIL("missing fixture column");
}
}  // namespace

TEST_CASE("immutable scan rows prove every published constraint", "[constraint-fixtures]") {
    for (const std::string table : {"departments", "products", "employees", "projects"}) {
        INFO(table);
        REQUIRE_NOTHROW(fixture::validate_rows(table, fixture::rows(table)));
        for (const auto& bytes : fixture::records(table)) {
            auto info = vgi::wire::decode_ipc(bytes);
            REQUIRE(vgi::wire::get_enum(info, "enforcement") == "ENFORCED");
            REQUIRE(vgi::wire::get_enum(info, "validation") == "ALL_ROWS");
        }
    }
    for (int64_t version = 1; version <= 3; ++version) {
        INFO(version);
        REQUIRE(fixture::records("versioned_constraints", version).size() ==
                static_cast<size_t>(version == 1 ? 1 : version + 2));
    }
    auto historical = fixture::required_schema(
        arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8())}),
        "versioned_constraints", 1);
    REQUIRE_FALSE(historical->field(0)->nullable());
    REQUIRE(historical->field(1)->nullable());
    REQUIRE_THROWS(fixture::rows("versioned_constraints", 4));
}

TEST_CASE("invalid immutable rows cannot establish trusted constraints", "[constraint-fixtures]") {
    SECTION("NOT NULL") {
        auto rows = fixture::rows("departments");
        replace(rows, "name", array<arrow::StringBuilder, std::string>({"a", std::nullopt, "c"}));
        REQUIRE_THROWS_WITH(fixture::validate_rows("departments", rows),
                            "fixture NOT NULL violated: name");
    }
    SECTION("primary key") {
        auto rows = fixture::rows("products");
        replace(rows, "id", array<arrow::Int64Builder, int64_t>({1, 1, 3}));
        REQUIRE_THROWS_WITH(fixture::validate_rows("products", rows),
                            "fixture PRIMARY_KEY contains duplicate key");
    }
    SECTION("unique") {
        auto rows = fixture::rows("departments");
        replace(rows, "name", array<arrow::StringBuilder, std::string>({"a", "a", "c"}));
        REQUIRE_THROWS_WITH(fixture::validate_rows("departments", rows),
                            "fixture UNIQUE contains duplicate key");
    }
    SECTION("composite primary key") {
        auto rows = fixture::rows("projects");
        replace(rows, "project_code",
                array<arrow::StringBuilder, std::string>({"P001", "P001", "P003"}));
        REQUIRE_THROWS_WITH(fixture::validate_rows("projects", rows),
                            "fixture PRIMARY_KEY contains duplicate key");
    }
    SECTION("foreign key") {
        auto rows = fixture::rows("employees");
        replace(rows, "department_id", array<arrow::Int64Builder, int64_t>({1, 1, 2, 2, 99}));
        REQUIRE_THROWS_WITH(fixture::validate_rows("employees", rows),
                            "fixture FOREIGN KEY missing referenced row");
    }
    SECTION("check") {
        auto rows = fixture::rows("departments");
        replace(rows, "budget", array<arrow::DoubleBuilder, double>({0, -1, 100}));
        REQUIRE_THROWS_WITH(fixture::validate_rows("departments", rows),
                            "fixture CHECK budget >= 0 violated");
    }
    SECTION("historical unique and foreign key") {
        auto rows = fixture::rows("versioned_constraints", 3);
        replace(rows, "email", array<arrow::StringBuilder, std::string>({"same", "same", "other"}));
        REQUIRE_THROWS_WITH(fixture::validate_rows("versioned_constraints", rows, 3),
                            "fixture UNIQUE contains duplicate key");
        rows = fixture::rows("versioned_constraints", 3);
        replace(rows, "department_id", array<arrow::Int64Builder, int64_t>({1, 99, 1}));
        REQUIRE_THROWS_WITH(fixture::validate_rows("versioned_constraints", rows, 3),
                            "fixture FOREIGN KEY missing referenced row");
    }
    SECTION("nullable foreign key remains legal") {
        auto rows = fixture::rows("employees");
        replace(rows, "department_id",
                array<arrow::Int64Builder, int64_t>({1, 1, 2, 2, std::nullopt}));
        REQUIRE_NOTHROW(fixture::validate_rows("employees", rows));
    }
}

TEST_CASE("constraint nullability retains column and schema metadata", "[constraint-fixtures]") {
    auto metadata = arrow::key_value_metadata({"comment", "opaque"}, {"", "raw"});
    auto schema = arrow::schema(
        {arrow::field("id", arrow::int64())->WithMetadata(metadata),
         arrow::field("name", arrow::utf8()), arrow::field("budget", arrow::float64())},
        metadata);
    auto required = fixture::required_schema(schema, "departments");
    REQUIRE_FALSE(required->field(0)->nullable());
    REQUIRE_FALSE(required->field(1)->nullable());
    REQUIRE(required->field(2)->nullable());
    REQUIRE(required->field(0)->metadata()->Equals(*metadata));
    REQUIRE(required->metadata()->Equals(*metadata));
    REQUIRE(schema->field(0)->nullable());
}
