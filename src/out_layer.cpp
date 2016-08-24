#include <assert.h>
#include "out_layer.h"

OutLayer::OutLayer(int layer_size, std::string name /* "dense" */) 
    : Layer(layer_size, name)
{
	printf("OutLayer constructor (%s)\n", name.c_str());

	// output connection will join last layer with layer_size nodes with the out layer with 1 node. 
	assert("layer_size == 1");
}
//----------------------------------------------------------------------
OutLayer::~OutLayer()
{
	printf("OutLayer destructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
OutLayer::OutLayer(OutLayer&)
{
}
//----------------------------------------------------------------------
