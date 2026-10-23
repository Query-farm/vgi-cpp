// Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "wire.h"

TEST_CASE("Python catalog records survive the C++ worker codec", "[catalog-contracts]") {
    std::ifstream input(VGI_CATALOG_CONTRACT_VECTORS);
    REQUIRE(input.good());
    const auto records = nlohmann::json::parse(input).at("records");
    const auto fixture = [&](const char* name) {
        return vgi::wire::base64_decode(records.at(name).get<std::string>());
    };
    const auto expression_source = vgi::wire::decode_ipc(fixture("column_expression_metadata"));
    const auto expression_restored = vgi::wire::decode_ipc(vgi::wire::encode_ipc(expression_source));
    REQUIRE(expression_source->Equals(*expression_restored, true));
    REQUIRE(vgi::wire::base64_decode(expression_restored->schema()->field(0)->metadata()->Get("vgi.column.default.v1").ValueOrDie()) == fixture("column_default_expression"));
    REQUIRE(vgi::wire::base64_decode(expression_restored->schema()->field(1)->metadata()->Get("vgi.column.generated.v1").ValueOrDie()) == fixture("column_generated_expression"));
    REQUIRE(expression_restored->schema()->field(1)->metadata()->Get("vgi.column.generation.v1").ValueOrDie() == "virtual");
    namespace gen = vgi::generated;
    const std::vector<std::pair<const char*, std::shared_ptr<arrow::Schema>>> cases = {
        {"attach_default_schema_none", gen::CatalogAttachResultSchema()},
        {"attach_default_schema_main", gen::CatalogAttachResultSchema()},
        {"attach_default_schema_dotted", gen::CatalogAttachResultSchema()},
        {"index_key_column", gen::IndexKeySchema()},
        {"index_key_nested", gen::IndexKeySchema()},
        {"index_key_expression", gen::IndexKeySchema()},
        {"index_definition", gen::IndexDefinitionSchema()},
        {"index_definition_primary", gen::IndexDefinitionSchema()},
        {"index_info", gen::IndexInfoSchema()},
        {"index_create", gen::IndexCreateRequestSchema()},
        {"index_create_cross_schema", gen::IndexCreateRequestSchema()},
        {"sql_statement", gen::SqlStatementSchema()},
        {"constraint_ref", gen::CatalogObjectRefSchema()},
        {"foreign_key_definition", gen::ForeignKeyDefinitionSchema()},
        {"constraint_definition_not_null", gen::ConstraintDefinitionSchema()},
        {"constraint_definition_unique", gen::ConstraintDefinitionSchema()},
        {"constraint_definition_primary", gen::ConstraintDefinitionSchema()},
        {"constraint_definition_check", gen::ConstraintDefinitionSchema()},
        {"constraint_definition_foreign", gen::ConstraintDefinitionSchema()},
        {"constraint_info", gen::ConstraintInfoSchema()},
        {"constraint_info_untrusted", gen::ConstraintInfoSchema()},
        {"constraint_change_add", gen::ConstraintChangeSchema()},
        {"constraint_change_drop", gen::ConstraintChangeSchema()},
        {"constraint_change_rename", gen::ConstraintChangeSchema()},
        {"constraint_change_unname", gen::ConstraintChangeSchema()},
        {"constraint_change_replace", gen::ConstraintChangeSchema()},
        {"constraint_snapshot", gen::ConstraintSnapshotSchema()},
        {"constraint_snapshot_empty", gen::ConstraintSnapshotSchema()},
        {"constraints_apply", gen::ConstraintsApplyRequestSchema()},
        {"constraints_apply_autocommit", gen::ConstraintsApplyRequestSchema()},
        {"macro_parameter", gen::MacroParameterSchema()},
        {"macro_parameter_expression", gen::MacroParameterSchema()},
        {"macro_parameter_untyped", gen::MacroParameterSchema()},
        {"macro_overload", gen::MacroOverloadSchema()},
        {"macro_create", gen::MacroCreateRequestSchema()},
        {"macro_create_single", gen::MacroCreateRequestSchema()},
        {"macro_info", gen::MacroInfoSchema()},
        {"macro_info_single", gen::MacroInfoSchema()},
        {"query_offer", gen::ClientQueryCapabilitiesSchema()},
        {"query_capabilities", gen::QueryCapabilitiesSchema()},
        {"query_context", gen::QueryEvaluationContextSchema()},
        {"query_object_reference", gen::QueryObjectReferenceSchema()},
        {"query_prepare", gen::CatalogQueryPrepareRequestSchema()},
        {"query_accepted", gen::CatalogQueryPrepareResultSchema()},
        {"query_declined", gen::CatalogQueryPrepareResultSchema()},

        {"capabilities", gen::CatalogCapabilitiesSchema()},
        {"empty_capabilities", gen::CatalogCapabilitiesSchema()},
        {"definitions", gen::OptionDefinitionsSchema()},
        {"object_ref", gen::CatalogObjectRefSchema()},
        {"trigger_ref", gen::CatalogObjectRefSchema()},
        {"object_dependency", gen::ObjectDependencySchema()},
        {"dependency_membership", gen::ObjectDependencySchema()},
        {"dependency_column_literal", gen::ObjectDependencySchema()},
        {"dependency_column_nested", gen::ObjectDependencySchema()},
        {"dependency_generated", gen::ObjectDependencySchema()},
        {"dependency_generated_chain", gen::ObjectDependencySchema()},
        {"dependency_column_snapshot", gen::DependencySnapshotSchema()},
        {"dependency_snapshot", gen::DependencySnapshotSchema()},
        {"empty_dependency_snapshot", gen::DependencySnapshotSchema()},
        {"metadata_patch", gen::CatalogMetadataChangesSchema()},
        {"metadata_clear_comment", gen::CatalogMetadataChangesSchema()},
        {"metadata_noop", gen::CatalogMetadataChangesSchema()},
        {"changes", gen::OptionChangesSchema()},
        {"reset_only", gen::OptionChangesSchema()},
        {"state", gen::OptionStateSchema()},
        {"empty_state", gen::OptionStateSchema()},
        {"paths", gen::RequiredFilterPathsSchema()},
        {"empty_paths", gen::RequiredFilterPathsSchema()},
        {"expression", gen::SqlExpressionSchema()},
        {"sort_key", gen::SortKeySchema()},
        {"layout", gen::TableLayoutSchema()},
        {"empty_layout", gen::TableLayoutSchema()},
        {"layout_change", gen::TableLayoutChangeSchema()},
        {"layout_clear", gen::TableLayoutChangeSchema()},
        {"table_create", gen::TableCreateRequestSchema()},
        {"table_create_legacy", gen::TableCreateRequestSchema()},
        {"table_info", gen::TableInfoSchema()},
        {"view_info", gen::ViewInfoSchema()},
        {"view_info_implicit", gen::ViewInfoSchema()},
        {"view_info_barrier", gen::ViewInfoSchema()},
        {"column_add_literal", gen::CatalogTableColumnAddParamsSchema()},
        {"column_add_nested", gen::CatalogTableColumnAddParamsSchema()},
        {"column_drop_literal", gen::CatalogTableColumnDropParamsSchema()},
        {"column_drop_nested", gen::CatalogTableColumnDropParamsSchema()},
        {"column_rename_literal", gen::CatalogTableColumnRenameParamsSchema()},
        {"column_rename_nested", gen::CatalogTableColumnRenameParamsSchema()},
        {"column_type_change_literal", gen::CatalogTableColumnTypeChangeParamsSchema()},
        {"column_type_change_nested", gen::CatalogTableColumnTypeChangeParamsSchema()},
        {"column_default_set_literal", gen::CatalogTableColumnDefaultSetParamsSchema()},
        {"column_default_set_nested", gen::CatalogTableColumnDefaultSetParamsSchema()},
        {"column_default_drop_literal", gen::CatalogTableColumnDefaultDropParamsSchema()},
        {"column_default_drop_nested", gen::CatalogTableColumnDefaultDropParamsSchema()},
        {"column_comment_set_literal", gen::CatalogTableColumnCommentSetParamsSchema()},
        {"column_comment_set_nested", gen::CatalogTableColumnCommentSetParamsSchema()},
        {"not_null_set_literal", gen::CatalogTableNotNullSetParamsSchema()},
        {"not_null_set_nested", gen::CatalogTableNotNullSetParamsSchema()},
        {"not_null_drop_literal", gen::CatalogTableNotNullDropParamsSchema()},
        {"not_null_drop_nested", gen::CatalogTableNotNullDropParamsSchema()},

    };
    for (const auto& [name, expected] : cases) {
        INFO(name);
        const auto source = vgi::wire::decode_ipc(fixture(name));
        REQUIRE(source->schema()->Equals(expected, true));
        auto builder = vgi::wire::ResultBuilder(expected);
        for (const auto& field : expected->fields()) {
            builder.set_array(field->name(), vgi::wire::column(source, field->name()));
        }
        const auto restored = vgi::wire::decode_ipc(vgi::wire::encode_ipc(builder.finish()));
        REQUIRE(restored->Equals(*source, true));
        const std::string case_name(name);
        if (case_name.starts_with("column_default_set_") ||
            case_name.starts_with("column_type_change_")) {
            const auto payload = std::static_pointer_cast<arrow::BinaryArray>(
                vgi::wire::column(restored, "expression"));
            if (case_name == "column_type_change_literal") {
                REQUIRE(payload->IsNull(0));
            } else {
                REQUIRE_FALSE(payload->IsNull(0));
                const auto bytes = payload->GetView(0);
                const auto expression = vgi::wire::decode_ipc(
                    std::string(bytes.begin(), bytes.end()));
                REQUIRE(expression->schema()->Equals(gen::SqlExpressionSchema(), true));
                REQUIRE(vgi::wire::get_string_list(expression, "schema_path") ==
                        vgi::wire::get_string_list(restored, "schema_path"));
                REQUIRE_FALSE(vgi::wire::get_string(expression, "profile").empty());
            }
        }
    }
    REQUIRE(vgi::wire::get_string_list(vgi::wire::decode_ipc(fixture("view_info")), "aliases") ==
            std::vector<std::string>{"a.b", "with\"quote"});
    REQUIRE_FALSE(vgi::wire::get_optional_string(vgi::wire::decode_ipc(fixture("attach_default_schema_none")), "default_schema").has_value());
    REQUIRE(vgi::wire::get_optional_string(vgi::wire::decode_ipc(fixture("attach_default_schema_main")), "default_schema") == "main");
    REQUIRE(vgi::wire::get_optional_string(vgi::wire::decode_ipc(fixture("attach_default_schema_dotted")), "default_schema") == "sales.archive");
    REQUIRE(vgi::wire::get_string(vgi::wire::decode_ipc(fixture("view_info_barrier")), "security") == "barrier");
    const auto membership = vgi::wire::decode_ipc(fixture("dependency_membership"));
    const auto owned = vgi::wire::decode_ipc(fixture("object_dependency"));
    REQUIRE_FALSE(vgi::wire::get_bool(membership, "blocks_alter"));
    REQUIRE(vgi::wire::get_bool(owned, "blocks_alter"));
    for (const auto& edge : {membership, owned}) {
        REQUIRE(vgi::wire::column(edge, "subject_column_paths")->IsNull(0));
        REQUIRE(vgi::wire::column(edge, "dependent_column_path")->IsNull(0));
    }
    const auto dependency_paths = [](const std::shared_ptr<arrow::RecordBatch>& edge) {
        const auto outer = std::static_pointer_cast<arrow::ListArray>(vgi::wire::column(edge, "subject_column_paths"));
        REQUIRE_FALSE(outer->IsNull(0));
        const auto inner = std::static_pointer_cast<arrow::ListArray>(outer->values());
        const auto parts = std::static_pointer_cast<arrow::StringArray>(inner->values());
        std::vector<std::vector<std::string>> result;
        for (int64_t i = outer->value_offset(0); i < outer->value_offset(0) + outer->value_length(0); ++i) {
            std::vector<std::string> path;
            for (int64_t j = inner->value_offset(i); j < inner->value_offset(i) + inner->value_length(i); ++j) path.push_back(parts->GetString(j));
            result.push_back(std::move(path));
        }
        return result;
    };
    REQUIRE(dependency_paths(vgi::wire::decode_ipc(fixture("dependency_column_literal"))) == std::vector<std::vector<std::string>>{{"a.b"}});
    REQUIRE(dependency_paths(vgi::wire::decode_ipc(fixture("dependency_column_nested"))) == std::vector<std::vector<std::string>>{{"record", "a.b"}});
    const auto generated = vgi::wire::decode_ipc(fixture("dependency_generated"));
    REQUIRE(vgi::wire::get_binary(generated, "dependent") == vgi::wire::get_binary(generated, "subject"));
    REQUIRE(vgi::wire::get_string_list(generated, "dependent_column_path") == std::vector<std::string>{"generated"});
    REQUIRE(dependency_paths(generated) == std::vector<std::vector<std::string>>{{"a.b"}, {"record", "a.b"}});
    const auto chain = vgi::wire::decode_ipc(fixture("dependency_generated_chain"));
    REQUIRE(vgi::wire::get_string_list(chain, "dependent_column_path") == std::vector<std::string>{"second.generated"});
    REQUIRE(dependency_paths(chain) == std::vector<std::vector<std::string>>{{"generated"}});
    REQUIRE(vgi::wire::get_binary_list(vgi::wire::decode_ipc(fixture("dependency_column_snapshot")), "dependencies") ==
        std::vector<std::string>{fixture("dependency_column_literal"), fixture("dependency_generated"), fixture("dependency_generated_chain")});

    const auto index_info = vgi::wire::decode_ipc(fixture("index_info"));
    const auto index_create = vgi::wire::decode_ipc(fixture("index_create"));
    const auto index_definition = vgi::wire::decode_ipc(fixture("index_definition"));
    REQUIRE(vgi::wire::get_binary(index_info, "definition") == fixture("index_definition"));
    REQUIRE(vgi::wire::get_binary(index_create, "definition") == fixture("index_definition"));
    REQUIRE(vgi::wire::get_binary(index_info, "options") == fixture("state"));
    REQUIRE(vgi::wire::get_binary(index_definition, "options") == fixture("values"));
    REQUIRE(vgi::wire::get_binary_list(index_definition, "keys") == std::vector<std::string>{fixture("index_key_column"), fixture("index_key_expression")});
    REQUIRE(vgi::wire::get_string_list(vgi::wire::decode_ipc(fixture("index_key_column")), "column_path") == std::vector<std::string>{"a.b"});
    REQUIRE(vgi::wire::get_string_list(vgi::wire::decode_ipc(fixture("index_key_nested")), "column_path") == std::vector<std::string>{"record", "a.b"});

    const auto table_create = vgi::wire::decode_ipc(fixture("table_create"));
    const auto table_info = vgi::wire::decode_ipc(fixture("table_info"));
    const auto constraint_definitions = vgi::wire::get_binary_list(table_create, "constraint_definitions");
    const auto constraints = vgi::wire::get_binary_list(table_info, "constraints");
    REQUIRE(constraint_definitions.size() == 1);
    REQUIRE(constraints.size() == 1);
    REQUIRE(vgi::wire::get_binary_list(vgi::wire::decode_ipc(fixture("table_create_legacy")), "constraint_definitions").empty());
    REQUIRE(vgi::wire::get_binary(table_info, "definition_revision") == std::string("revision\0\xff", 10));
    const auto constraint = vgi::wire::decode_ipc(constraints[0]);
    REQUIRE(vgi::wire::get_binary(constraint, "definition") == constraint_definitions[0]);
    REQUIRE(vgi::wire::get_string(vgi::wire::decode_ipc(vgi::wire::get_binary(constraint, "ref")), "name") == "fixture-stable-key");
    const auto constraint_definition = vgi::wire::decode_ipc(constraint_definitions[0]);
    const auto paths = std::static_pointer_cast<arrow::ListArray>(vgi::wire::column(constraint_definition, "column_paths"));
    REQUIRE(paths->value_length(0) == 1);
    const auto path = std::static_pointer_cast<arrow::ListArray>(paths->values());
    REQUIRE(path->value_length(0) == 2);
    const auto components = std::static_pointer_cast<arrow::StringArray>(path->values());
    REQUIRE(components->GetString(0) == "record");
    REQUIRE(components->GetString(1) == "a.b");
    const auto definitions = vgi::wire::decode_ipc(fixture("definitions"));
    REQUIRE(vgi::wire::get_binary(definitions, "defaults") == fixture("values"));
    const auto changes = vgi::wire::decode_ipc(fixture("changes"));
    REQUIRE(vgi::wire::get_binary(changes, "set_values") == fixture("values"));
    const auto macro = vgi::wire::decode_ipc(fixture("macro_create"));
    const auto overloads = std::static_pointer_cast<arrow::ListArray>(vgi::wire::column(macro, "overloads"));
    REQUIRE(overloads->value_length(0) == 2);
    REQUIRE(std::static_pointer_cast<arrow::ListArray>(vgi::wire::column(vgi::wire::decode_ipc(fixture("macro_create_single")), "overloads"))->value_length(0) == 1);
    const auto parameter = vgi::wire::decode_ipc(fixture("macro_parameter"));
    const auto default_value = vgi::wire::decode_ipc(vgi::wire::get_binary(parameter, "default_value"));
    REQUIRE(default_value->column(0)->IsNull(0));
    REQUIRE(default_value->schema()->field(0)->metadata()->Get("domain").ValueOrDie() == "retained");
    REQUIRE(vgi::wire::column(parameter, "default_expression")->IsNull(0));
    REQUIRE(vgi::wire::column(vgi::wire::decode_ipc(fixture("macro_parameter_untyped")), "declared_type")->IsNull(0));
    for (const auto* name : {"logical_tuple", "logical_struct", "logical_variant_sql_null",
                            "logical_variant_null", "logical_variant_absent", "logical_variant_null_member"}) {
        INFO(name);
        const auto source = vgi::wire::decode_ipc(fixture(name));
        const auto restored = vgi::wire::decode_ipc(vgi::wire::encode_ipc(source));
        REQUIRE(restored->Equals(*source, true));
        REQUIRE(restored->num_rows() == 1);
        const auto field = restored->schema()->field(0);
        if (std::string(name) != "logical_tuple") {
            REQUIRE((!field->metadata() || !field->metadata()->Contains("vgi.logical_type.v1")));
        } else {
            REQUIRE(field->metadata()->Get("vgi.logical_type.v1").ValueOrDie() == "tuple");
        }
        REQUIRE(restored->column(0)->IsNull(0) == (std::string(name) == "logical_variant_sql_null"));
        if (std::string(name).find("logical_variant_") == 0) {
            REQUIRE(field->metadata()->Get("ARROW:extension:name").ValueOrDie() == "arrow.parquet.variant");
            REQUIRE(field->metadata()->Get("ARROW:extension:metadata").ValueOrDie() == "");
            const auto storage = std::static_pointer_cast<arrow::StructType>(field->type());
            REQUIRE(storage->num_fields() == 2);
            REQUIRE(storage->field(0)->name() == "metadata");
            REQUIRE_FALSE(storage->field(0)->nullable());
            REQUIRE(storage->field(1)->name() == "value");
            REQUIRE(storage->field(1)->nullable());
            if (std::string(name) != "logical_variant_sql_null") {
                const auto column = std::static_pointer_cast<arrow::StructArray>(restored->column(0));
                const std::string expected_metadata = std::string(name) == "logical_variant_null_member"
                    ? std::string("\x01\x01\x00\x07present", 11) : std::string("\x01\x00\x00", 3);
                const std::string expected_value = std::string(name) == "logical_variant_null"
                    ? std::string("\x00", 1) : std::string(name) == "logical_variant_absent"
                    ? std::string("\x02\x00\x00", 3) : std::string("\x02\x01\x00\x00\x01\x00", 6);
                REQUIRE(std::static_pointer_cast<arrow::BinaryArray>(column->field(0))->GetString(0) == expected_metadata);
                REQUIRE(std::static_pointer_cast<arrow::BinaryArray>(column->field(1))->GetString(0) == expected_value);
            }
        }
    }
    REQUIRE(fixture("logical_variant_sql_null") != fixture("logical_variant_null"));
    REQUIRE(fixture("logical_variant_absent") != fixture("logical_variant_null_member"));
    for (const auto* name : {"query_parameters", "query_empty_parameters"}) {
        INFO(name);
        const auto source = vgi::wire::decode_ipc(fixture(name));
        const auto restored = vgi::wire::decode_ipc(vgi::wire::encode_ipc(source));
        REQUIRE(restored->Equals(*source, true));
        REQUIRE(restored->num_rows() == 1);
        REQUIRE(restored->schema()->metadata()->Get("vgi.query.parameters.v1").ValueOrDie() ==
                (std::string(name) == "query_empty_parameters" ? "positional" : "named"));
        if (std::string(name) == "query_empty_parameters") {
            REQUIRE(restored->num_columns() == 0);
        } else {
            REQUIRE(restored->column(0)->IsNull(0));
            const auto type = std::static_pointer_cast<arrow::StructType>(restored->schema()->field(1)->type());
            REQUIRE(type->field(0)->metadata()->Get("nested").ValueOrDie() == "retained");
        }
    }
}
