cmake_minimum_required(VERSION 2.8)
Project(rnn)

include_directories(Eigen)

add_executable(rnn model.cpp weights.cpp activations.cpp main.cpp layers.cpp gmm_layer.cpp lstm_layer.cpp dense_layer.cpp)
