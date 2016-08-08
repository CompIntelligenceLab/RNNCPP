include_directories(${RNNCPP_SOURCE_DIR}/include)

add_executable(copy_constructors copy_constructors.cpp)
target_link_libraries(copy_constructors core ${ARMADILLO_LIBRARIES})
