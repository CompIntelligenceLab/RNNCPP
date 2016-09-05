
include_directories(/include)
add_executable(test_recurrent_model_bias1 test_recurrent_model_bias1.cpp)
target_link_libraries(test_recurrent_model_bias1  core ${ARMADILLO_LIBRARIES}) 

