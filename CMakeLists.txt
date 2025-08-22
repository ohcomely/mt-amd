cmake_minimum_required(VERSION 3.22)
project(ParAMD VERSION 1.0 LANGUAGES C CXX)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SuiteSparse include and lib paths (user-known)
include_directories(/usr/include/suitesparse /usr/local/include/suitesparse)
link_directories(/usr/lib/x86_64-linux-gnu /usr/local/lib)

# Define macro to indicate SuiteSparse usage
add_definitions(-DUSE_SUITESPARSE)

# Your headers
include_directories(include)

# Source files
set(SOURCES
  src/config.cpp
  src/matrix.cpp
  src/paramd.cpp
  src/paramd_optimized.cpp
  src/utils.cpp
)

# Shared library
add_library(paramd_lib SHARED ${SOURCES})

set_target_properties(paramd_lib PROPERTIES
  OUTPUT_NAME paramd
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER include/paramd/paramd.h
)

# Main executable
add_executable(paramd bin/main.cpp)

# Link your lib and SuiteSparse
target_link_libraries(paramd PRIVATE paramd_lib amd)

# Add OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  target_link_libraries(paramd_lib PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(paramd PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install rules
install(TARGETS paramd_lib
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/paramd
)

install(TARGETS paramd
  RUNTIME DESTINATION bin
)
