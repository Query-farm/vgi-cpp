// Copyright 2026 Query Farm LLC - https://query.farm
#include "landing.h"
#include <nlohmann/json.hpp>
#include "landing_assets.h"

namespace vgi {
void configure_landing(vgi_rpc::HttpConfig& config, const std::string& name,
                       const std::string& server_id, const std::string& doc) {
    const nlohmann::json status = {{"status", "ok"},
                                   {"server_id", server_id},
                                   {"protocol", "vgi.v2"},
                                   {"worker", name},
                                   {"doc", doc},
                                   {"version", VGI_SDK_VERSION},
                                   {"lang", "cpp"},
                                   {"oauth", false},
                                   {"cupola_base", "https://cupola.query-farm.services"}};
    config.static_assets["/"] = {
        std::string(reinterpret_cast<const char*>(landing_html), sizeof(landing_html)),
        "text/html; charset=utf-8", status.dump()};
    config.static_assets["/vgi-client.js"] = {
        std::string(reinterpret_cast<const char*>(client_js), sizeof(client_js)),
        "text/javascript; charset=utf-8", std::nullopt};
}
}  // namespace vgi
