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
    namespace gen = vgi::generated;
    const std::vector<std::pair<const char*, std::shared_ptr<arrow::Schema>>> cases = {
        {"macro_parameter", gen::MacroParameterSchema()},
        {"macro_parameter_expression", gen::MacroParameterSchema()},
        {"macro_parameter_untyped", gen::MacroParameterSchema()},
        {"macro_overload", gen::MacroOverloadSchema()},
        {"macro_create", gen::MacroCreateRequestSchema()},
        {"macro_create_legacy", gen::MacroCreateRequestSchema()},
        {"macro_info", gen::MacroInfoSchema()},
        {"macro_info_legacy", gen::MacroInfoSchema()},
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
    }
    REQUIRE(vgi::wire::get_string_list(vgi::wire::decode_ipc(fixture("view_info")), "aliases") ==
            std::vector<std::string>{"a.b", "with\"quote"});
    REQUIRE(vgi::wire::get_string(vgi::wire::decode_ipc(fixture("view_info_barrier")), "security") == "barrier");
    const auto definitions = vgi::wire::decode_ipc(fixture("definitions"));
    REQUIRE(vgi::wire::get_binary(definitions, "defaults") == fixture("values"));
    const auto changes = vgi::wire::decode_ipc(fixture("changes"));
    REQUIRE(vgi::wire::get_binary(changes, "set_values") == fixture("values"));
    const auto macro = vgi::wire::decode_ipc(fixture("macro_create"));
    const auto overloads = std::static_pointer_cast<arrow::ListArray>(vgi::wire::column(macro, "overloads"));
    REQUIRE(overloads->value_length(0) == 2);
    REQUIRE(vgi::wire::column(vgi::wire::decode_ipc(fixture("macro_create_legacy")), "overloads")->IsNull(0));
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
        if (std::string(name) == "logical_struct") {
            REQUIRE((!field->metadata() || !field->metadata()->Contains("vgi.logical_type.v1")));
        } else {
            REQUIRE(field->metadata()->Get("vgi.logical_type.v1").ValueOrDie() ==
                    (std::string(name) == "logical_tuple" ? "tuple" : "variant"));
        }
        REQUIRE(restored->column(0)->IsNull(0) == (std::string(name) == "logical_variant_sql_null"));
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
