
include_directories(${RNNCPP_SOURCE_DIR}/include)

add_executable(activations activations.cpp)
#target_link_libraries(activations core)
target_link_libraries(activations core ${ARMADILLO_LIBRARIES})

