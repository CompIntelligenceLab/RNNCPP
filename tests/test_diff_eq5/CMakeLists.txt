
include_directories(/include)
add_executable(test_diff_eq5 test_diff_eq5.cpp)
target_link_libraries(test_diff_eq5  core ${ARMADILLO_LIBRARIES}) 

