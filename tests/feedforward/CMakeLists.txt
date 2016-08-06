
include_directories(${RNNCPP_SOURCE_DIR}/include)

add_executable(feed_forward one_layer_feedforward.cpp)
target_link_libraries(feed_forward core)

