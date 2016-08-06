include_directories(${RNNCPP_SOURCE_DIR}/include)

set(SOURCE 
    optimizer.cpp
	activations.cpp
	dense_layer.cpp
	layers.cpp
	weights.cpp
	gmm_layer.cpp
	lstm_layer.cpp
	model.cpp)

add_library(core ${SOURCE})
