include_directories(${RNNCPP_SOURCE_DIR}/include)

add_executable(go.x test_recurrent_model1.cpp)
target_link_libraries(go.x core ${ARMADILLO_LIBRARIES})
