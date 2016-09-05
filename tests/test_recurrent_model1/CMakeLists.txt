include_directories(${RNNCPP_SOURCE_DIR}/include)

add_executable(test_recurrent_model1 test_recurrent_model1.cpp)
target_link_libraries(test_recurrent_model1 core ${ARMADILLO_LIBRARIES})
