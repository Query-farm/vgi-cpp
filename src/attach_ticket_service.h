// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// `vgi.attach_tickets.v1`: the protocol hosting `seal_attach` (spec §5).
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <vgi_rpc/server.h>

#include "vgi/attach_ticket.h"
#include "vgi/catalog.h"

namespace vgi {

// The attach options a catalog declares, or nullopt for a catalog this worker
// does not serve.
using DeclaredAttachOptions =
    std::function<std::optional<std::vector<AttachOptionSpec>>(const std::string& catalog)>;

// `seal_attach(request: SealAttachRequest) -> AttachTicket`, sealing with `key`
// and capping lifetimes at `max_ttl_seconds` (nullopt: no maximum).
vgi_rpc::ProtocolBuilder attach_tickets_protocol(DeclaredAttachOptions declared,
                                                 AttachTicketKey key,
                                                 std::optional<int64_t> max_ttl_seconds);

// The ticket lifetime ceiling: the grant keys' maximum when grant keys are
// configured, else `VGI_RPC_GRANT_MAX_TTL_SECONDS` when set, else none. A value
// that is not a positive integer throws `std::invalid_argument`.
std::optional<int64_t> attach_ticket_max_ttl(std::optional<int64_t> grant_keys_max_ttl);

}  // namespace vgi
