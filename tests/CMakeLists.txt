# © Copyright 2025-2026, Query.Farm LLC - https://query.farm
# SPDX-License-Identifier: Apache-2.0

# Deliberately thin.  The integration suite in ~/Development/vgi is the
# definitive test source; what belongs here is internal utility behaviour that
# suite cannot see — type mapping, argument coercion, wire encoding.
find_package(Catch2 3 CONFIG REQUIRED)

add_executable(function_test function_test.cpp)
target_link_libraries(function_test PRIVATE vgi::vgi Catch2::Catch2WithMain)
target_include_directories(function_test PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(Catch)
catch_discover_tests(function_test)
