include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(geomflow_tests
  test_geometry.cpp
  test_cnf.cpp
  test_adjoint.cpp
)

target_link_libraries(geomflow_tests PRIVATE
  geomflow_headers
  Catch2::Catch2WithMain
)

target_include_directories(geomflow_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/examples/cpp
)

include(Catch)
catch_discover_tests(geomflow_tests TEST_SPEC "~[slow]")
add_test(NAME geomflow_slow COMMAND geomflow_tests "[slow]")
set_tests_properties(geomflow_slow PROPERTIES LABELS slow)
