
include_directories(${RNNCPP_SOURCE_DIR}/include)
add_executable(test_recurrent_model4 test_recurrent_model4.cpp)
target_link_libraries(test_recurrent_model4  core ${ARMADILLO_LIBRARIES})

