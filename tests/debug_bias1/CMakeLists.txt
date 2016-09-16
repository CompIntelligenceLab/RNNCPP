
include_directories(/include)
add_executable(debug_bias1 debug_bias1 .cpp)
target_link_libraries(debug_bias1 core ${ARMADILLO_LIBRARIES}) 

