#include "lstm_layer.h"

LSTMLayer::LSTMLayer(int layer_size, std::string name /* lstm */) 
    : Layer(layer_size, name)
{
	//this->name = name;
	type = "lstm";
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
