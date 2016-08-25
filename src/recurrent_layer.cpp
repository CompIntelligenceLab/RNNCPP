#include "recurrent_layer.h"


// allows for default constructor
RecurrentLayer::RecurrentLayer(int layer_size /*1*/, std::string name /*rec_layer*/)
	: Layer(layer_size, name)
{
	printf("RecurrentLayer (%s): constructor\n", this->name.c_str());
	recurrent_conn = new Connection(layer_size, layer_size, "loop_conn");
	recurrent_conn->initialize();
	loop_input.set_size(nb_batch);
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
void RecurrentLayer::forwardData(Connection* conn, VF2D_F& prod, int seq)
{
	printf("recurrent: forward data\n");
	// forward data to spatial connections
	Layer::forwardData(conn, prod, seq);

	// forward data to temporal connections
	// handle self loop
	WEIGHT& loop_wght = recurrent_conn->getWeight();

	if (areIncomingLayerConnectionsComplete()) {
		for (int b=0; b < loop_input.n_rows; b++) {
			loop_input(b) = loop_wght * outputs[b];
		}
	}
	loop_wght.print("loop weight");
	outputs.print("loop outputs");
	loop_input.print("loop");
	exit(0);
}
//----------------------------------------------------------------------
#if 0
bool RecurrentLayer::areIncomingLayerConnectionsComplete()
{
	int nb_arrivals = prev.size();

	// +1 to account for the self-loop of the recurrent node. 
	return ((nb_hit+1) == nb_arrivals);
}
#endif
//----------------------------------------------------------------------
void RecurrentLayer::processData(Connection* conn, VF2D_F& prod)
{
		printf("recurrent: process data\n");
		Layer::processData(conn, prod);
}
//----------------------------------------------------------------------
