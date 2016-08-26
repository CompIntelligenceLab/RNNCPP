#include "print_utils.h"
#include "recurrent_layer.h"


// allows for default constructor
RecurrentLayer::RecurrentLayer(int layer_size /*1*/, std::string name /*rec_layer*/)
	: Layer(layer_size, name)
{
	printf("RecurrentLayer (%s): constructor\n", this->name.c_str());
	recurrent_conn = new Connection(layer_size, layer_size, "loop_conn");
	recurrent_conn->initialize();
	recurrent_conn->from = this;
	recurrent_conn->to = this;
	recurrent_conn->setTemporal(true);
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
void RecurrentLayer::forwardLoops(int seq_i)
{
	// forward data to temporal connections
	// handle self loop
	WEIGHT& loop_wght = recurrent_conn->getWeight();

	//U::matmul(loop_input, loop_wght, outputs); // out of bounds
	// calculate for sequence 0, store in sequence 1

	if (seq_i >= 0) {
		U::matmul(loop_input, loop_wght, outputs, seq_i, seq_i+1); // out of bounds
	}
}
//----------------------------------------------------------------------
void RecurrentLayer::forwardLoops()
{
	// forward data to temporal connections
	// handle self loop
	printf("inside RecurrentLayer::forwardLoops\n");
	WEIGHT& loop_wght = recurrent_conn->getWeight();

	//U::print(loop_input, "loop_input");
	//U::print(loop_wght, "loop_wght");
	//U::print(outputs, "outputs");

	U::matmul(loop_input, loop_wght, outputs); // out of bounds
	// calculate for sequence 0, store in sequence 1
	U::matmul(loop_input, loop_wght, outputs, 0, 1); // out of bounds

	//loop_input.print("loop input");
}
//----------------------------------------------------------------------
void RecurrentLayer::initVars(int nb_batch)
{
	Layer::initVars(nb_batch);

	loop_input.set_size(nb_batch);
	loop_delta.set_size(nb_batch);

    for (int b=0; b < nb_batch; b++) {
        loop_input[b] = VF2D(layer_size, seq_len);   // << NEED proper sequence length, maybe
        loop_delta[b] = VF2D(layer_size, seq_len);
		loop_input[b].zeros();
		loop_delta[b].zeros();
    }   
    
    reset();
}  
//----------------------------------------------------------------------
