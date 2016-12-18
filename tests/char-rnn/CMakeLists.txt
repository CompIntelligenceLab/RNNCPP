
include_directories(/include)
add_executable(char-rnn char-rnn.cpp)
target_link_libraries(char-rnn  core ${ARMADILLO_LIBRARIES}) 

