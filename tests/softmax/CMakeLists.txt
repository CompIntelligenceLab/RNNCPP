
include_directories(${RNNCPP_SOURCE_DIR}/include)
add_executable(softmax softmax.cpp)
target_link_libraries(softmax core ${ARMADILLO_LIBRARIES}) 

