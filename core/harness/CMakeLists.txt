# Builds the SiliconBadgers harness executables against a pinned llama.cpp checkout.
# One CMake project replaces the hand-written per-OS compiler/linker command: CMake knows the
# import libraries (Windows), rpaths (Linux/macOS) and where the shared libraries land.
#
#   cmake -S core/harness -B work/build-baseline -G Ninja -DLLAMA_CPP_SOURCE=work/llama.cpp ...
#
# Targets: sb-profile (timing/logit harness), sb-profile-prompt (same loop, prompt taken from a file),
# sb-capture-graph (scheduled-graph capture).
cmake_minimum_required(VERSION 3.14)
project(sb_harness C CXX)

set(LLAMA_CPP_SOURCE "" CACHE PATH "Pinned llama.cpp checkout")
if(NOT EXISTS "${LLAMA_CPP_SOURCE}/CMakeLists.txt")
  message(FATAL_ERROR "LLAMA_CPP_SOURCE must point at a llama.cpp checkout (got '${LLAMA_CPP_SOURCE}')")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Executables and shared libraries share one directory: Windows finds the DLLs next to the
# executable, and Unix build-tree rpaths point here.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# The same llama.cpp options the 2026-09-22 runner used (GGML_METAL and GGML_NATIVE come from the command line).
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_SOURCE}" llama.cpp)

function(sb_harness_target name source)
  add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  target_link_libraries(${name} PRIVATE llama ggml)
  target_include_directories(${name} PRIVATE "${LLAMA_CPP_SOURCE}/src")
  if(WIN32)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(${name} PRIVATE "/FI${CMAKE_CURRENT_SOURCE_DIR}/compat_win.h")
    else()
      target_compile_options(${name} PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/compat_win.h")
    endif()
    target_compile_definitions(${name} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  endif()
endfunction()

sb_harness_target(sb-profile profile.cpp)
sb_harness_target(sb-profile-prompt profile_prompt.cpp)
sb_harness_target(sb-capture-graph capture_graph.cpp)
target_include_directories(sb-capture-graph PRIVATE "${LLAMA_CPP_SOURCE}/vendor")
