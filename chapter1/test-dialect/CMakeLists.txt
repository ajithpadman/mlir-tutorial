cmake_minimum_required(VERSION 3.20.0)
project(my-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# Find MLIR package using LLVM_INSTALL_DIR
# LLVM_INSTALL_DIR can be set as:
#   1. Environment variable: export LLVM_INSTALL_DIR=/path/to/llvm-install
#   2. CMake cache variable: cmake .. -DLLVM_INSTALL_DIR=/path/to/llvm-install
#   3. Or specify MLIR_DIR directly: cmake .. -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir

if(DEFINED ENV{LLVM_INSTALL_DIR})
  set(LLVM_INSTALL_DIR $ENV{LLVM_INSTALL_DIR} CACHE PATH "Path to LLVM installation directory")
endif()

if(LLVM_INSTALL_DIR AND NOT MLIR_DIR)
  set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH "Path to MLIR CMake config")
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set up include directories and library paths
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Include MLIR CMake modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Set output directories
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(lib)

