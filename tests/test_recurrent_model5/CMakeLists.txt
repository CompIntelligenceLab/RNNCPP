
include_directories(${RNNCPP_SOURCE_DIR}/include)
add_executable(test_recurrent_model5  test_recurrent_model5.cpp)
target_link_libraries(test_recurrent_model5  core ${ARMADILLO_LIBRARIES}) 
