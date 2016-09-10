#include "gmm_layer.h"

GMMLayer::GMMLayer(int layer_size, std::string name /* "ea"gmm */) 
   : Layer(layer_size, name)
{
	//this->name = name;
	type = "input";
}
//----------------------------------------------------------------------
GMMLayer::~GMMLayer()
{
	printf("GMMLayer destructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
GMMLayer::GMMLayer(GMMLayer&)
{
}
//----------------------------------------------------------------------
