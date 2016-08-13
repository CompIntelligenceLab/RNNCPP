include_directories(${RNNCPP_SOURCE_DIR}/include)

set(SOURCE 
    optimizer.cpp
	activations.cpp
	dense_layer.cpp
	layers.cpp
	#weights.cpp
	connection.cpp
	gmm_layer.cpp
	lstm_layer.cpp
	objective.cpp
	input.cpp
	input_layer.cpp
	model.cpp)

add_library(core ${SOURCE})
