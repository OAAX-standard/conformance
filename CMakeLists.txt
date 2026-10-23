cmake_minimum_required(VERSION 3.13)
project(oaax_conformance C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_compile_options(-Wall -Wextra)

# The spec version under test: the harness is built against that version's pinned header.
set(OAAX_SPEC_VERSION 2.0 CACHE STRING "OAAX specification version")
set(SPEC_INCLUDE ${CMAKE_SOURCE_DIR}/spec/${OAAX_SPEC_VERSION})

find_package(Threads REQUIRED)
add_compile_definitions(_GNU_SOURCE)
add_executable(oaax-harness harness/src/main.c harness/src/common.c harness/src/scenarios_runtime.c harness/src/scenarios_model.c)
target_include_directories(oaax-harness PRIVATE ${SPEC_INCLUDE} ${CMAKE_SOURCE_DIR}/harness/src)
target_link_libraries(oaax-harness ${CMAKE_DL_LIBS} Threads::Threads)

# Mock runtimes: the suite's own test fixtures (faults are switched by OAAX_MOCK_FAULT).
# Only the 9 runtime_* functions are exported, like a clean runtime.
foreach(variant mock_runtime mock_missing_export mock_extra_export mock_versioned)
  add_library(${variant} SHARED mock-runtime/mock.c)
  target_include_directories(${variant} PRIVATE ${SPEC_INCLUDE})
  target_link_libraries(${variant} ${CMAKE_DL_LIBS} pthread)
  target_compile_options(${variant} PRIVATE -fvisibility=hidden)
endforeach()
target_compile_definitions(mock_missing_export PRIVATE MOCK_OMIT_INFO)
target_compile_definitions(mock_extra_export PRIVATE MOCK_EXTRA_EXPORT)
# Exports controlled by a version script, like the reference runtime: its version node must not count as an extra export.
target_link_options(mock_versioned PRIVATE -Wl,--version-script=${CMAKE_SOURCE_DIR}/mock-runtime/versioned.map)
