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
        {"capabilities", gen::CatalogCapabilitiesSchema()},
        {"empty_capabilities", gen::CatalogCapabilitiesSchema()},
        {"definitions", gen::OptionDefinitionsSchema()},
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
    const auto definitions = vgi::wire::decode_ipc(fixture("definitions"));
    REQUIRE(vgi::wire::get_binary(definitions, "defaults") == fixture("values"));
    const auto changes = vgi::wire::decode_ipc(fixture("changes"));
    REQUIRE(vgi::wire::get_binary(changes, "set_values") == fixture("values"));
}
