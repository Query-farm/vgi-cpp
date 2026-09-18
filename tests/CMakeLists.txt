# © Copyright 2025, 2026 Query Farm LLC - https://query.farm
# Deliberately thin.  The integration suite in ~/Development/vgi is the
# definitive test source; what belongs here is internal utility behaviour that
# suite cannot see — type mapping, argument coercion, wire encoding.
find_package(Catch2 3 CONFIG REQUIRED)

add_executable(function_test function_test.cpp)
target_link_libraries(function_test PRIVATE vgi::vgi Catch2::Catch2WithMain)
target_include_directories(function_test PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(Catch)
catch_discover_tests(function_test)

# Drives the built example worker over stdio, a unix socket and HTTP with
# vgi-rpc's own clients: the routing key is a statement about a running server,
# not about a constant, so it is tested against one.
add_executable(protocol_routing_test protocol_routing_test.cpp)
target_link_libraries(protocol_routing_test PRIVATE vgi::vgi Catch2::Catch2WithMain)
target_include_directories(protocol_routing_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(protocol_routing_test PRIVATE
    VGI_EXAMPLE_WORKER_PATH="$<TARGET_FILE:vgi-example-worker>")
add_dependencies(protocol_routing_test vgi-example-worker)
catch_discover_tests(protocol_routing_test)
