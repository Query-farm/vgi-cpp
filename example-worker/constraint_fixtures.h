// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/type.h>

#include <vgi/constraints.h>

namespace example {
namespace constraint_fixtures {

// The one CHECK the fixtures publish, kept for validate_rows to compare.
inline constexpr const char* kBudgetCheck = "budget >= 0";

// The departments primary key, which both foreign keys point at.
inline vgi::ConstraintReference departments_key() {
    return {{"data"}, "departments", {{"id"}}, "fixture:departments:primary"};
}

// Fixed provider keys, not engine OIDs. Every fixture fact is attested
// ENFORCED + ALL_ROWS: validate_rows() proves it against the immutable rows.
inline std::string constraint(const std::string& table, const std::string& key,
                              vgi::ConstraintKind kind, std::vector<vgi::ColumnPath> paths,
                              std::optional<std::string> check = std::nullopt,
                              std::optional<vgi::ConstraintReference> foreign = std::nullopt) {
    vgi::ConstraintRecordSpec spec;
    spec.schema_path = {"data"};
    spec.table = table;
    spec.id = key;
    spec.kind = kind;
    spec.columns = std::move(paths);
    spec.enforcement = vgi::ConstraintEnforcement::Enforced;
    spec.validation = vgi::ConstraintValidation::AllRows;
    spec.check_sql = std::move(check);
    spec.references = std::move(foreign);
    return vgi::constraint_record(spec);
}

// Private fixture definitions. records() validates the immutable scan arrays
// before publishing these trusted facts; there is no writable table endpoint.
inline std::vector<std::string> definitions(const std::string& table, int64_t version = 0) {
    if (table == "departments") {
        return {
            constraint(table, "fixture:departments:nn:id", vgi::ConstraintKind::NotNull, {{"id"}}),
            constraint(table, "fixture:departments:nn:name", vgi::ConstraintKind::NotNull,
                       {{"name"}}),
            constraint(table, "fixture:departments:primary", vgi::ConstraintKind::PrimaryKey,
                       {{"id"}}),
            constraint(table, "fixture:departments:unique:name", vgi::ConstraintKind::Unique,
                       {{"name"}}),
            constraint(table, "fixture:departments:check:budget", vgi::ConstraintKind::Check, {},
                       kBudgetCheck),
        };
    }
    if (table == "products") {
        return {
            constraint(table, "fixture:products:nn:id", vgi::ConstraintKind::NotNull, {{"id"}}),
            constraint(table, "fixture:products:primary", vgi::ConstraintKind::PrimaryKey,
                       {{"id"}}),
        };
    }
    if (table == "employees") {
        return {
            constraint(table, "fixture:employees:nn:id", vgi::ConstraintKind::NotNull, {{"id"}}),
            constraint(table, "fixture:employees:nn:name", vgi::ConstraintKind::NotNull,
                       {{"name"}}),
            constraint(table, "fixture:employees:nn:email", vgi::ConstraintKind::NotNull,
                       {{"email"}}),
            constraint(table, "fixture:employees:primary", vgi::ConstraintKind::PrimaryKey,
                       {{"id"}}),
            constraint(table, "fixture:employees:unique:email", vgi::ConstraintKind::Unique,
                       {{"email"}}),
            constraint(table, "fixture:employees:fk:department", vgi::ConstraintKind::ForeignKey,
                       {{"department_id"}}, std::nullopt, departments_key()),
        };
    }
    if (table == "projects") {
        return {
            constraint(table, "fixture:projects:nn:department", vgi::ConstraintKind::NotNull,
                       {{"department_id"}}),
            constraint(table, "fixture:projects:nn:code", vgi::ConstraintKind::NotNull,
                       {{"project_code"}}),
            constraint(table, "fixture:projects:nn:title", vgi::ConstraintKind::NotNull,
                       {{"title"}}),
            constraint(table, "fixture:projects:primary", vgi::ConstraintKind::PrimaryKey,
                       {{"department_id"}, {"project_code"}}),
            constraint(table, "fixture:projects:fk:department", vgi::ConstraintKind::ForeignKey,
                       {{"department_id"}}, std::nullopt, departments_key()),
        };
    }
    if (table == "versioned_constraints") {
        if (version < 1 || version > 3)
            throw std::invalid_argument("unknown constraint fixture version");
        std::vector<std::string> result = {
            constraint(table, "fixture:versioned_constraints:nn:id", vgi::ConstraintKind::NotNull,
                       {{"id"}}),
        };
        if (version >= 2) {
            result.push_back(constraint(table, "fixture:versioned_constraints:nn:name",
                                        vgi::ConstraintKind::NotNull, {{"name"}}));
            result.push_back(constraint(table, "fixture:versioned_constraints:primary",
                                        vgi::ConstraintKind::PrimaryKey, {{"id"}}));
            result.push_back(constraint(table, "fixture:versioned_constraints:unique:email",
                                        vgi::ConstraintKind::Unique, {{"email"}}));
        }
        if (version >= 3)
            result.push_back(constraint(table, "fixture:versioned_constraints:fk:department",
                                        vgi::ConstraintKind::ForeignKey, {{"department_id"}},
                                        std::nullopt, departments_key()));
        return result;
    }
    throw std::invalid_argument("unknown constraint fixture table: " + table);
}

using Columns = std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>;

// The one source of rows for both the scan and its constraint proof.
Columns rows(const std::string& table, int64_t version = 0);
void validate_rows(const std::string& table, const Columns& columns, int64_t version = 0);
std::shared_ptr<arrow::Schema> required_schema(const std::shared_ptr<arrow::Schema>& schema,
                                               const std::string& table, int64_t version = 0);
inline std::vector<std::string> records(const std::string& table, int64_t version = 0) {
    validate_rows(table, rows(table, version), version);
    return definitions(table, version);
}

}  // namespace constraint_fixtures
}  // namespace example
