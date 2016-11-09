
include_directories(/include)
add_executable(test_diff_eq2 test_diff_eq2.cpp)
target_link_libraries(test_diff_eq2  core ${ARMADILLO_LIBRARIES}) 

