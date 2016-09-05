
include_directories(${RNNCPP_SOURCE_DIR}/include)
add_executable(test_recurrent_model3  test_recurrent_model3.cpp)
target_link_libraries(test_recurrent_model3 core ${ARMADILLO_LIBRARIES}) 

