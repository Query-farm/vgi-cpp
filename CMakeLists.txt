# © Copyright 2025-2026, Query.Farm LLC - https://query.farm
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(vgi-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── The vgi-rpc C++ port ──────────────────────────────────────────────────
#
# Two ways in, and co-development is the common one: this SDK and the RPC
# framework beneath it move together, so by default we build a sibling
# checkout from source rather than resolving an installed copy that would go
# stale silently.  Point VGI_RPC_SOURCE_DIR elsewhere, or set
# VGI_USE_INSTALLED_RPC=ON to find_package an installed one instead.
option(VGI_USE_INSTALLED_RPC "Resolve vgi_rpc with find_package instead of building a sibling checkout" OFF)
set(VGI_RPC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vgi-rpc-c++"
    CACHE PATH "Sibling vgi-rpc-c++ checkout to build against")

if(VGI_USE_INSTALLED_RPC)
    find_package(vgi_rpc CONFIG REQUIRED)
else()
    if(NOT EXISTS "${VGI_RPC_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "vgi-rpc-c++ not found at ${VGI_RPC_SOURCE_DIR}.\n"
            "Clone it beside this repo, set -DVGI_RPC_SOURCE_DIR=<path>, or "
            "configure with -DVGI_USE_INSTALLED_RPC=ON to use an installed copy.")
    endif()
    # Its tests are its own business; we only want the library target.
    set(VGI_RPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${VGI_RPC_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/vgi-rpc" EXCLUDE_FROM_ALL)
endif()

find_package(Arrow CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# ── The SDK ───────────────────────────────────────────────────────────────
add_library(vgi
    src/worker.cpp
    src/dispatcher.cpp
    src/function.cpp
    src/methods.cpp
    src/wire.cpp
    src/catalog.cpp
    src/arg_schema.cpp
)
add_library(vgi::vgi ALIAS vgi)

target_include_directories(vgi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(vgi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(vgi PUBLIC vgi_rpc nlohmann_json::nlohmann_json)
if(TARGET Arrow::arrow_shared)
    target_link_libraries(vgi PUBLIC Arrow::arrow_shared)
else()
    target_link_libraries(vgi PUBLIC Arrow::arrow_static)
endif()

# ── The example worker: the integration-test fixture ──────────────────────
add_subdirectory(example-worker)

# ── Tests (opt-in) ────────────────────────────────────────────────────────
option(VGI_BUILD_TESTS "Build vgi unit tests" OFF)
if(VGI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
