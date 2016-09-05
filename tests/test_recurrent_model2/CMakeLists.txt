
include_directories(${RNNCPP_SOURCE_DIR}/include)
add_executable(test_recurrent_model2  test_recurrent_model2.cpp)
target_link_libraries(test_recurrent_model2  core  ${ARMADILLO_LIBRARIES}) 

