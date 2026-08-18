# © Copyright 2025-2026, Query.Farm LLC - https://query.farm
# SPDX-License-Identifier: Apache-2.0

# The integration-test fixture: one binary that stands in for every catalog the
# suite attaches, routed by VGI_WORKER_CATALOG_NAME the way the Rust example
# worker is.
add_executable(vgi-example-worker main.cpp)
target_link_libraries(vgi-example-worker PRIVATE vgi::vgi)
