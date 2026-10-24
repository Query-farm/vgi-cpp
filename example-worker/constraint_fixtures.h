// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/array/array_nested.h>

#include <vgi/generated/vgi_protocol_schemas.hpp>

#include "../src/wire.h"

namespace example {
namespace constraint_fixtures {

namespace gen = ::vgi::generated;
using vgi::wire::encode_ipc;
using vgi::wire::ResultBuilder;

// Fixed provider keys, not engine OIDs.
inline std::string ref(const std::string& table, const std::string& key) {
    return encode_ipc(ResultBuilder(gen::CatalogObjectRefSchema())
                          .set_string("kind", "constraint")
                          .set_string_list("schema_path", {"data"})
                          .set_string("name", key)
                          .set_string("owner_table", table)
                          .finish());
}

inline std::string department_foreign_key() {
    auto table = ResultBuilder(gen::CatalogObjectRefSchema())
                     .set_string("kind", "table")
                     .set_string_list("schema_path", {"data"})
                     .set_string("name", "departments")
                     .set_null("owner_table")
                     .finish();
    return encode_ipc(ResultBuilder(gen::ForeignKeyDefinitionSchema())
                          .set_binary("referenced_table", encode_ipc(table))
                          .set_string_list_list("referenced_column_paths", {{"id"}})
                          .set_enum("match", "SIMPLE")
                          .set_enum("on_update", "NO_ACTION")
                          .set_enum("on_delete", "NO_ACTION")
                          .finish());
}

inline std::string budget_check() {
    auto context = ResultBuilder(gen::QueryEvaluationContextSchema())
                       .set_string("timezone", "UTC")
                       .set_string("collation", "binary")
                       .set_string("default_order", "asc")
                       .set_string("default_null_order", "nulls_last")
                       .set_bool("preserve_insertion_order", true)
                       .finish();
    auto context_struct = arrow::StructArray::Make(context->columns(), context->schema()->fields());
    if (!context_struct.ok()) throw std::runtime_error(context_struct.status().ToString());
    return encode_ipc(ResultBuilder(gen::SqlExpressionSchema())
                          .set_string("sql", "budget >= 0")
                          .set_string("profile", "vgi.expression.duckdb.scalar.v1")
                          .set_string_list("schema_path", {"data"})
                          .set_array("evaluation_context", context_struct.MoveValueUnsafe())
                          .finish());
}

inline std::string constraint(const std::string& table, const std::string& key,
                              const std::string& kind,
                              const std::vector<std::vector<std::string>>& paths,
                              const std::string& check = {}, const std::string& foreign = {}) {
    auto definition = ResultBuilder(gen::ConstraintDefinitionSchema());
    definition.set_enum("kind", kind)
        .set_string_list_list("column_paths", paths)
        .set_null("declared_name");
    if (check.empty())
        definition.set_null("check_expression");
    else
        definition.set_binary("check_expression", check);
    if (foreign.empty())
        definition.set_null("foreign_key");
    else
        definition.set_binary("foreign_key", foreign);
    auto info = ResultBuilder(gen::ConstraintInfoSchema());
    info.set_binary("ref", ref(table, key))
        .set_binary("definition", encode_ipc(definition.finish()))
        .set_enum("enforcement", "ENFORCED")
        .set_enum("validation", "ALL_ROWS");
    if (foreign.empty())
        info.set_null("referenced_constraint");
    else
        info.set_binary("referenced_constraint", ref("departments", "fixture:departments:primary"));
    return encode_ipc(info.finish());
}

// Private fixture definitions. records() validates the immutable scan arrays
// before publishing these trusted facts; there is no writable table endpoint.
inline std::vector<std::string> definitions(const std::string& table, int64_t version = 0) {
    if (table == "departments") {
        return {
            constraint(table, "fixture:departments:nn:id", "NOT_NULL", {{"id"}}),
            constraint(table, "fixture:departments:nn:name", "NOT_NULL", {{"name"}}),
            constraint(table, "fixture:departments:primary", "PRIMARY_KEY", {{"id"}}),
            constraint(table, "fixture:departments:unique:name", "UNIQUE", {{"name"}}),
            constraint(table, "fixture:departments:check:budget", "CHECK", {}, budget_check()),
        };
    }
    if (table == "products") {
        return {
            constraint(table, "fixture:products:nn:id", "NOT_NULL", {{"id"}}),
            constraint(table, "fixture:products:primary", "PRIMARY_KEY", {{"id"}}),
        };
    }
    if (table == "employees") {
        return {
            constraint(table, "fixture:employees:nn:id", "NOT_NULL", {{"id"}}),
            constraint(table, "fixture:employees:nn:name", "NOT_NULL", {{"name"}}),
            constraint(table, "fixture:employees:nn:email", "NOT_NULL", {{"email"}}),
            constraint(table, "fixture:employees:primary", "PRIMARY_KEY", {{"id"}}),
            constraint(table, "fixture:employees:unique:email", "UNIQUE", {{"email"}}),
            constraint(table, "fixture:employees:fk:department", "FOREIGN_KEY", {{"department_id"}},
                       {}, department_foreign_key()),
        };
    }
    if (table == "projects") {
        return {
            constraint(table, "fixture:projects:nn:department", "NOT_NULL", {{"department_id"}}),
            constraint(table, "fixture:projects:nn:code", "NOT_NULL", {{"project_code"}}),
            constraint(table, "fixture:projects:nn:title", "NOT_NULL", {{"title"}}),
            constraint(table, "fixture:projects:primary", "PRIMARY_KEY",
                       {{"department_id"}, {"project_code"}}),
            constraint(table, "fixture:projects:fk:department", "FOREIGN_KEY", {{"department_id"}},
                       {}, department_foreign_key()),
        };
    }
    if (table == "versioned_constraints") {
        if (version < 1 || version > 3)
            throw std::invalid_argument("unknown constraint fixture version");
        std::vector<std::string> result = {
            constraint(table, "fixture:versioned_constraints:nn:id", "NOT_NULL", {{"id"}}),
        };
        if (version >= 2) {
            result.push_back(constraint(table, "fixture:versioned_constraints:nn:name", "NOT_NULL", {{"name"}}));
            result.push_back(constraint(table, "fixture:versioned_constraints:primary", "PRIMARY_KEY", {{"id"}}));
            result.push_back(constraint(table, "fixture:versioned_constraints:unique:email", "UNIQUE", {{"email"}}));
        }
        if (version >= 3)
            result.push_back(constraint(table, "fixture:versioned_constraints:fk:department",
                                        "FOREIGN_KEY", {{"department_id"}}, {},
                                        department_foreign_key()));
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
