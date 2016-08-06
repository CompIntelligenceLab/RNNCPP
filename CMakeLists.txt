project(RNNCPP)
cmake_minimum_required(VERSION 2.8)

# Tells CMake to look for the Find<Package>.cmake files in the cmake folder
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Require Eigen3. This defines all our variables for includes/libs etc
find_package(Eigen3 REQUIRED)
find_package(Armadillo REQUIRED)
message(${ARMADILLO_LIBRARIES})
message("=====================")

# This is defined in previous line
include_directories(
   ${EIGEN3_INCLUDE_DIR}
   ${ARMADILLO_INCLUDE_DIR}
   ${RNNCPP_SOURCE_DIR}/include
)

# Call the CMakeLists.txt file in the src directory
add_subdirectory(src)
add_subdirectory(tests)

add_executable(rnn main.cpp)

target_link_libraries(rnn core ${ARMADILLO_LIBRARIES})
