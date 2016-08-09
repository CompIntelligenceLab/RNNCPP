include_directories(${RNNCPP_SOURCE_DIR}/include)

add_executable(weights weights.cpp)
target_link_libraries(weights core ${ARMADILLO_LIBRARIES})
