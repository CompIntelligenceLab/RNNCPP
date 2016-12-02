
include_directories(/include)
add_executable(test_diff_eq7 test_diff_eq7.cpp)
target_link_libraries(test_diff_eq7  core ${ARMADILLO_LIBRARIES}) 

