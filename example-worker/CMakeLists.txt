# © Copyright 2025-2026, Query.Farm LLC - https://query.farm
# SPDX-License-Identifier: Apache-2.0

# The integration-test fixture: one binary that stands in for every catalog the
# suite attaches, routed by VGI_WORKER_CATALOG_NAME the way the Rust example
# worker is.
add_executable(vgi-example-worker
    main.cpp
    scalar/util.cpp
    scalar/arithmetic.cpp
    scalar/strings.cpp
    scalar/seeded.cpp
    scalar/geo.cpp
    scalar/types.cpp
)
target_include_directories(vgi-example-worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vgi-example-worker PRIVATE vgi::vgi)
