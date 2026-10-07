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

# A real Dispatcher behind a real vgi-rpc server, in-process so the test can
# count what building a function listing asks of each registered function.
add_executable(function_listing_test function_listing_test.cpp)
target_link_libraries(function_listing_test PRIVATE vgi::vgi Catch2::Catch2WithMain)
target_include_directories(function_listing_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(function_listing_test)

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

# Sealed grants end to end: mint through issue_grant, then authenticate a
# vgi.v2 call with the grant, against the built example worker.
add_executable(grant_auth_test grant_auth_test.cpp)
target_link_libraries(grant_auth_test PRIVATE vgi::vgi Catch2::Catch2WithMain)
target_include_directories(grant_auth_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(grant_auth_test PRIVATE
    VGI_EXAMPLE_WORKER_PATH="$<TARGET_FILE:vgi-example-worker>")
add_dependencies(grant_auth_test vgi-example-worker)
catch_discover_tests(grant_auth_test)

# Sealed attach / transaction values: the envelope, the boundary that opens
# them, and a Dispatcher in-process, unsealed and sealed.
add_executable(opaque_seal_test opaque_seal_test.cpp)
target_link_libraries(opaque_seal_test PRIVATE vgi::vgi Catch2::Catch2WithMain)
target_include_directories(opaque_seal_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(opaque_seal_test)

# The same over HTTP against the built example worker: replay, tamper,
# plaintext, cross-attach transaction, and nothing raw in the log.
add_executable(opaque_seal_e2e_test opaque_seal_e2e_test.cpp)
target_link_libraries(opaque_seal_e2e_test PRIVATE vgi::vgi Catch2::Catch2WithMain)
target_include_directories(opaque_seal_e2e_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(opaque_seal_e2e_test PRIVATE
    VGI_EXAMPLE_WORKER_PATH="$<TARGET_FILE:vgi-example-worker>")
add_dependencies(opaque_seal_e2e_test vgi-example-worker)
catch_discover_tests(opaque_seal_e2e_test)
