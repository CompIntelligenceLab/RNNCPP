
include_directories(/include)
add_executable(test_diff_eq3 test_diff_eq3.cpp)
target_link_libraries(test_diff_eq3  core ${ARMADILLO_LIBRARIES}) 

