#include "recurrent_layer.h"


// allows for default constructor
RecurrentLayer::RecurrentLayer(int layer_size /*1*/, std::string name /*rec_layer*/)
	: Layer(layer_size, name)
{
	printf("RecurrentLayer (%s): constructor\n", this->name.c_str());
	recurrent_conn = new Connection(layer_size, layer_size, "loop_conn");
}
//----------------------------------------------------------------------
RecurrentLayer::~RecurrentLayer()
{
	printf("RecurrentLayer (%s): destructor\n", this->name.c_str());
	delete recurrent_conn;
}
//----------------------------------------------------------------------
RecurrentLayer::RecurrentLayer(const RecurrentLayer& l) //: Layer(RecurrentLayer) // FIX
{
	name    = l.name + 'c';
	printf("RecurrentLayer (%s): copy constructor\n", this->name.c_str());

	// what to do with RecurrentConnection?  MUST FIX
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
const RecurrentLayer& RecurrentLayer::operator=(const RecurrentLayer& l)
{
	name    = l.name + '=';
	printf("RecurrentLayer (%s): operator=\n", this->name.c_str());
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void RecurrentLayer::forwardData(Connection* conn, VF2D_F& prod)
{
		VF2D_F& from_outputs = getOutputs();
		WEIGHT& wght = conn->getWeight();

		// matrix multiplication
		for (int b=0; b < from_outputs.size(); b++) {
			prod(b) = wght * from_outputs[b];
		}

		// handle self loop

}
//----------------------------------------------------------------------
