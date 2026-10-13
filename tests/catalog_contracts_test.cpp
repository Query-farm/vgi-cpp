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
    const auto definitions = vgi::wire::decode_ipc(fixture("definitions"));
    REQUIRE(vgi::wire::get_binary(definitions, "defaults") == fixture("values"));
    const auto changes = vgi::wire::decode_ipc(fixture("changes"));
    REQUIRE(vgi::wire::get_binary(changes, "set_values") == fixture("values"));
}
