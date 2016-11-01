
include_directories(/include)
add_executable(test_diff_eq1 test_diff_eq1.cpp)
target_link_libraries(test_diff_eq1  core ${ARMADILLO_LIBRARIES}) 

