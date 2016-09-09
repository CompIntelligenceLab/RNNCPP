
include_directories(${RNNCPP_SOURCE_DIR}/include)
add_executable(softmax_rec1 softmax_rec1.cpp)
target_link_libraries(softmax_rec1 core ${ARMADILLO_LIBRARIES}) 

