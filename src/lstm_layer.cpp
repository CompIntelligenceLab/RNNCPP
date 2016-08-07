#include "lstm_layer.h"

LSTMLayer::LSTMLayer(int layer_size, std::string name) : Layer(layer_size, name)
{
	//this->name = name;
}
//----------------------------------------------------------------------
LSTMLayer::~LSTMLayer()
{
	printf("LSTMLayer destructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
LSTMLayer::LSTMLayer(LSTMLayer&)
{
}
//----------------------------------------------------------------------
