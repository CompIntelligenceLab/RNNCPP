
include_directories(${RNNCPP_SOURCE_DIR}/include)
add_executable(test_recurrent_model_bias2 test_recurrent_model_bias2.cpp)
target_link_libraries(test_recurrent_model_bias2 core ${ARMADILLO_LIBRARIES}) 

