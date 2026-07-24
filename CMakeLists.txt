cmake_minimum_required(VERSION 3.18)
project(geomflow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Header-only C++ library ---
add_library(geomflow_headers INTERFACE)
target_include_directories(geomflow_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- C++ tests (Catch2) ---
option(BUILD_TESTS "Build C++ unit tests" ON)
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

# --- C++ examples ---
option(BUILD_EXAMPLES "Build C++ example applications" ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples/cpp)
endif()

# --- Python module (pybind11) ---
option(BUILD_PYTHON "Build Python pybind11 module" OFF)
if(BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 REQUIRED)

  pybind11_add_module(_geomflow
    python/bindings.cpp
  )

  target_link_libraries(_geomflow PRIVATE
    pybind11::module
  )

  target_include_directories(_geomflow PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS}
  )

  install(TARGETS _geomflow
    LIBRARY DESTINATION geomflow
    RUNTIME DESTINATION geomflow
    ARCHIVE DESTINATION geomflow
  )
endif()
