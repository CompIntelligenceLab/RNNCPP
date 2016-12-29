
include_directories(/include)
add_executable(gmm1d gmm1d)
target_link_libraries(gmm1d  core ${ARMADILLO_LIBRARIES}) 

