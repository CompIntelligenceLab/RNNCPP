
include_directories(/include)
add_executable(test_diff_eq4 test_diff_eq4.cpp)
target_link_libraries(test_diff_eq4  core ${ARMADILLO_LIBRARIES}) 

