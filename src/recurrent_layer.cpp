#include "print_utils.h"
#include "recurrent_layer.h"


// allows for default constructor
RecurrentLayer::RecurrentLayer(int layer_size /*1*/, std::string name /*rec_layer*/)
	: Layer(layer_size, name)
{
	printf("RecurrentLayer (%s): constructor\n", this->name.c_str());
	recurrent_conn = new Connection(layer_size, layer_size, "loop_conn");
	recurrent_conn->from = this;
	recurrent_conn->to = this;
	recurrent_conn->setTemporal(true);
	recurrent_conn->setTTo(1);    // by default, recurrent connections have delay of 1
	loop_input.set_size(nb_batch);
	recurrent_conn->initialize("xavier"); // must be last to call. 
	type = "recurrent";
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
	printf("RecurrentLayer copy constructor not complete\n");
	exit(1);

	// what to do with RecurrentConnection?  MUST FIX
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
const RecurrentLayer& RecurrentLayer::operator=(const RecurrentLayer& l)
{
	name    = l.name + '=';
	printf("RecurrentLayer (%s): operator=\n", this->name.c_str());
	printf("RecurrentLayer operator= not complete\n");
	exit(1);
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
void RecurrentLayer::processData(Connection* conn, VF2D_F& prod)
{
		printf("recurrent: process data\n");
		Layer::processData(conn, prod);
}
//----------------------------------------------------------------------
void RecurrentLayer::forwardLoops(int t)
{
	//printf("inside forward loops, t=%d\n", t);
	// forward data to temporal connections
	// handle self loop
	const WEIGHT& loop_wght = recurrent_conn->getWeight();

	loop_input.print("loop_input");

	//printf("forward Loops, inside\n");
	if (t >= 0) {
		U::matmul(loop_input, loop_wght, outputs, t, t+1); // out of bounds
	}
}
//----------------------------------------------------------------------
void RecurrentLayer::forwardLoops()
{
	// forward data to temporal connections
	// handle self loop
	//printf("inside RecurrentLayer::forwardLoops\n");
	const WEIGHT& loop_wght = recurrent_conn->getWeight();
	U::matmul(loop_input, loop_wght, outputs, 0, 1); // out of bounds
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
