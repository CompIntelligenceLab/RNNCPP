#include "dense_layer.h"

DenseLayer::DenseLayer(int layer_size, std::string name /* "dense" */) 
    : Layer(layer_size, name)
{
	type = "dense";
	printf("DenseLayer constructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
DenseLayer::~DenseLayer()
{
	printf("DenseLayer destructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
DenseLayer::DenseLayer(DenseLayer&)
{
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
