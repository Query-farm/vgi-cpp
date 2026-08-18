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
    scalar/same_name.cpp
    scalar/settings.cpp
    scalar/format.cpp
    scalar/secrets.cpp
    table/sequence.cpp
    table/cache.cpp
    table/more.cpp
    table/generators.cpp
    table/settings_fixtures.cpp
    table/secrets.cpp
    table/series.cpp
    table/versioned.cpp
    table/static_scan.cpp
    table/rff.cpp
    table/filters.cpp
    table/logging.cpp
    table/cache_more.cpp
    table/cache_partitions.cpp
    table/partitioned.cpp
    table/partition_broken.cpp
    copy_to/lines.cpp
    copy_from/lines.cpp
    catalog_def.cpp
    table_in_out/echo.cpp
    table_in_out/cached.cpp
    table_in_out/blended.cpp
    table_in_out/accumulate.cpp
    table_in_out/secrets.cpp
    table_in_out/cancellable.cpp
    table_in_out/tensor.cpp
    table_in_out/same_name.cpp
    aggregate/basic.cpp
    aggregate/more.cpp
    aggregate/window.cpp
    aggregate/tensor.cpp
    aggregate/same_name.cpp
    aggregate/percentile.cpp
    buffering/buffer_input.cpp
    buffering/sum_all_columns.cpp
    buffering/source_shapes.cpp
    buffering/large_state.cpp
)
target_include_directories(vgi-example-worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vgi-example-worker PRIVATE vgi::vgi)
