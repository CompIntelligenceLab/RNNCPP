#include "recurrent_layer.h"


// allows for default constructor
RecurrentLayer::RecurrentLayer(int layer_size /*1*/, std::string name /*rec_layer*/)
	: Layer(layer_size, name)
{
	printf("RecurrentLayer (%s): inside constructor\n", this->name.c_str());
}
//----------------------------------------------------------------------
RecurrentLayer::~RecurrentLayer()
{
	printf("RecurrentLayer (%s): inside destructor\n", this->name.c_str());
}
//----------------------------------------------------------------------
RecurrentLayer::RecurrentLayer(const RecurrentLayer&) 
{
	printf("RecurrentLayer (%s): inside copy constructor\n", this->name.c_str());
}
//----------------------------------------------------------------------
const RecurrentLayer& RecurrentLayer::operator=(const RecurrentLayer&)
{
	printf("RecurrentLayer (%s): inside operator=\n", this->name.c_str());
}
//----------------------------------------------------------------------

