#include "input_layer.h"

InputLayer::InputLayer(int layer_size, std::string name /* "input" */) 
    : Layer(layer_size, name)
{
	// For this layer, the layer_size and input_dim are identical
	if (activation) {
		delete activation;
		activation = new Identity("identity");
	}
	this->input_dim = layer_size;
	type = "input";
}
//----------------------------------------------------------------------
InputLayer::~InputLayer()
{
	printf("InputLayer destructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
InputLayer::InputLayer(InputLayer&)
{
}
//----------------------------------------------------------------------
