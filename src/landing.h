// Copyright 2026 Query Farm LLC - https://query.farm
#pragma once
#include <string>
#include "vgi_rpc/http_config.h"

namespace vgi {
void configure_landing(vgi_rpc::HttpConfig& config, const std::string& name,
                       const std::string& server_id, const std::string& doc);
}
