include_directories(${RNNCPP_SOURCE_DIR}/include)

add_executable(input input.cpp)
target_link_libraries(input core ${ARMADILLO_LIBRARIES})
