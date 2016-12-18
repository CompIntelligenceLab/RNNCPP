
include_directories(/include)
add_executable(categorical_classification categorical_classification)
target_link_libraries(categorical_classification  core ${ARMADILLO_LIBRARIES}) 

