
include_directories(/include)
add_executable(test_diff_eq6 test_diff_eq6.cpp)
target_link_libraries(test_diff_eq6  core ${ARMADILLO_LIBRARIES}) 

