cmake_minimum_required(VERSION 3.18)

set(EXAMPLE_SOURCES train_cnf.cpp infer_cnf.cpp)

add_executable(train_cnf train_cnf.cpp)
target_link_libraries(train_cnf PRIVATE geomflow_headers)
target_include_directories(train_cnf PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

add_executable(infer_cnf infer_cnf.cpp)
target_link_libraries(infer_cnf PRIVATE geomflow_headers)
target_include_directories(infer_cnf PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)