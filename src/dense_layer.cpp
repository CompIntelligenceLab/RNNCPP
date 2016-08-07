#include "dense_layer.h"

DenseLayer::DenseLayer(int layer_size, std::string name) : Layer(layer_size, name)
{
	//this->name = "dense";
}
//----------------------------------------------------------------------
DenseLayer::~DenseLayer()
{
}
//----------------------------------------------------------------------
DenseLayer::DenseLayer(DenseLayer&)
{
}
//----------------------------------------------------------------------
