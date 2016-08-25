#include "layers.h"
#include "print_utils.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int Layer::counter = 0;

Layer::Layer(int layer_size, std::string name /* "layer" */)
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("Layer constructor (%s)\n", this->name.c_str());

	counter++;

	// Input dimension has no significance if the layer is connected to several upstream layers
	// How to access connection to previous layer if this is the first layer? 

	this->layer_size = layer_size;
	//output_dim   = layer_size;  // no such member
	input_dim   = -1; // no assignment yet. 
	nb_batch    =  1;   // HOW TO SET BATCH FOR LAYERS? 
	seq_len     =  1; 
	print_verbose   = true;
	clock = 0;

	initVars(nb_batch);

	// Default activation: tanh
	activation = new Tanh("tanh");
}

void Layer::initVars(int nb_batch)
{
	inputs.set_size(nb_batch);
	outputs.set_size(nb_batch);
	delta.set_size(nb_batch);
	//printf("nb_batch= %d\n", nb_batch); exit(0);

	for (int i=0; i < nb_batch; i++) {
		inputs[i]  = VF2D(layer_size, 1);
		outputs[i] = VF2D(layer_size, 1);
	}
	nb_hit = 0;

	reset();
}

Layer::~Layer()
{
	printf("Layer destructor (%s)\n", name.c_str());
	if (activation) delete activation;
	activation = 0;
}

Layer::Layer(const Layer& l) : layer_size(l.layer_size), input_dim(l.input_dim),
   print_verbose(l.print_verbose), seq_len(l.seq_len), nb_batch(l.nb_batch),
   inputs(l.inputs), outputs(l.outputs), clock(l.clock), delta(l.delta)
{
	name    = l.name + 'c';
	printf("Layer copy constructor (%s)\n", name.c_str());

	//TODO
	// How does activation work with polymorphism ?
	//activation = new Activation(); // HOW TO DO?   (does not work)
	//*activation = *l.activation; // check name
	//print("Layer copy constructor, activation->name= ", activation->name);
}

const Layer& Layer::operator=(const Layer& l)
{

	// if no copying done, name does not change
	if (this != &l) {
		name = l.name + "=";
		layer_size = l.layer_size;
		input_dim = l.input_dim;
		print_verbose = l.print_verbose;
		inputs = l.inputs;
		outputs = l.outputs;
		delta = l.delta;
		inputs = l.inputs;
		outputs = l.outputs;
		seq_len = l.seq_len;
		nb_batch = l.seq_len;
		clock = l.clock;

		//Weights* w1; // remove class Weights
		Activation *a1;

		// remove class Weights (Nathan)
		try {
			a1 = new Tanh();
		} catch (...) {
			delete a1;
			printf("throw\n");
			throw;
		}

		*activation = *a1; 

		printf("Layer::operator= (%s)\n", name.c_str());
	}

	return *this;
}

void Layer::print(std::string msg /* "" */)
{
	printf("  -- layer: %s ---\n", name.c_str());
    if (msg != "") printf("%s\n", msg.c_str());
	printf("layer size: %d\n", layer_size);
	printf("input_dim: %d\n", input_dim);
	printf("inputs size: %llu\n", inputs.size());
	printf("outputs size: %llu\n", outputs.size());
	printf("print_verbose: %d\n", print_verbose);

	if (print_verbose == false) return;
}

void Layer::printSummary(std::string msg) 
{
	printf("%sLayer (%s), layer_size: %d\n", msg.c_str(), name.c_str(), layer_size);
}

void Layer::printName(std::string msg /*""*/)
{
	cout << "-- " << msg << ", layer (" << name << ") --" << endl;
}

// Take the inputs (dimensionality based on previous layer), and generate the outputs
void Layer::execute()
{
	outputs = (*activation)(inputs);
}

void Layer::reset()
{
	for (int b=0; b < inputs.size(); b++) {
		inputs(b).zeros();
		outputs(b).zeros();
		clock = 0;
	}
}

void Layer::resetBackprop()
{
	for (int b=0; b < delta.size(); b++) {
		delta(b).zeros();
	}
}

void Layer::incrOutputs(VF2D_F& x)
{
	for (int b=0; b < x.n_rows; b++) {
		outputs[b] += x[b];
	}
}

void Layer::incrInputs(VF2D_F& x)
{
	//printf("incrInputs: x.n_rows= %d\n", x.n_rows);
	//printf("inputs.n_rows= %d\n", inputs.n_rows);
	// inputs has incorrect number of fields.
	for (int b=0; b < x.n_rows; b++) {
		inputs[b] += x[b];
	}
}

void Layer::resetInputs()
{
	for (int b=0; b < inputs.n_rows; b++) {
		inputs[b].zeros();
	}
}


void Layer::incrDelta(VF2D_F& x)
{
	//printf("delta.rows: %d\n", delta.n_rows);
	//U::print(delta, "delta");
	//U::print(x, "incrDelta");
	//printf("x.n_rows = %d\n", x.n_rows);
	//printf("deltax.n_rows = %d\n", delta.n_rows);

	if (delta[0].n_rows == 0) {
		for (int b=0; b < x.n_rows; b++) {
			delta[b] = x[b];
		}
		//printf("deltax.n_rows = %d\n", delta.n_rows);
	} else {
		for (int b=0; b < x.n_rows; b++) {
			delta[b] += x[b];
		}
	}
}
//----------------------------------------------------------------------
void Layer::computeGradient()
{
	gradient = activation->derivative(outputs);
}
//----------------------------------------------------------------------
void Layer::forwardData(Connection* conn, VF2D_F& prod, int seq)
{
	// forward data to spatial connections

	VF2D_F& from_outputs = getOutputs();
	WEIGHT& wght = conn->getWeight();
	//U::matmul(prod, wght, from_outputs, 0);  // sequence element zero)
	U::matmul(prod, wght, from_outputs);

	prod.print("prod, forwardData");

	// Data is not actually forwarded. It should be forwarded to the input 
	// of the following layer. 
}
//----------------------------------------------------------------------
bool Layer::areIncomingLayerConnectionsComplete()
{
	return (nb_hit == prev.size());
}
//----------------------------------------------------------------------
// Perhaps break this up into processing by Connection then by output layer
void Layer::processOutputDataFromPreviousLayer(Connection* conn, VF2D_F& prod)
{
		printf("enter Layer::processOutputDataFromPreviousLayern");
		++nb_hit;

		VF2D_F& from_outputs = conn->from->getOutputs();
		WEIGHT& wght = conn->getWeight();  // what if connection operates differently
		VF2D_F& to_inputs = layer_inputs[conn->which_lc];

		#if 0
		conn->printSummary("conn");
		printf("which_lc= %d\n", conn->which_lc);
		printf("layer_inputs.size= %d\n",layer_inputs.size());
		layer_inputs[conn->which_lc].print("layer_inputs, which_lc");  // ===> zero
		wght.print("wght");
		from_outputs.print("from_outputs");
		#endif


		// TEMPORARY LOOP
		#if 0
		for (int i=0; i < layer_inputs.size(); i++) {
			//layer_inputs[i] = VF2D_F(nb_batch);
			int input_dim = getLayerSize();
			int seq_len   = getSeqLen();
			printf("input_dim, seq_len= %d, %d\n", input_dim, seq_len);
			for (int b=0; b < nb_batch; b++) {
				//layer_inputs[i](b) = VF2D(input_dim, seq_len);
				U::print(layer_inputs[i](b), "U:: layer_inputs (tmp)");
			}
		}
		#endif

		//U::matmul(to_inputs, wght, from_outputs);  // w * x
		U::matmul(prod, wght, from_outputs);  // w * x
		to_inputs = prod;


		prod.print("enter processData, prod");

		// Where are the various inputs added up? So derivatives will work if layer_size=1, but not otherwise. 

		// completeness only happens once per layer and per input value into the predicition module
		if (areIncomingLayerConnectionsComplete()) {
			 // sum up all the inputs + the temporal input if there is one
			 resetInputs();
			 for (int i=0; i < layer_inputs.size(); i++) {
			 	incrInputs(layer_inputs[i]);
				//layer_inputs[i].print("layer_inputs incr");
				//printf("incrInputs, input: %d\n", i);
			 }
			 // add the self-looping if there. 
			 incrInputs(loop_input);
			 //loop_input.print("add loop_input, ");
			 prod = getActivation()(getInputs());
			 //prod.print("processData, output, ");
			 setOutputs(prod);
		}
}
//----------------------------------------------------------------------
void Layer::processData(Connection* conn, VF2D_F& prod)
{
		++nb_hit;

		VF2D_F& to_inputs = layer_inputs[conn->which_lc];
		to_inputs = prod;
		prod.print("enter processData, prod");

		// Where are the various inputs added up? So derivatives will work if layer_size=1, but not otherwise. 

		if (areIncomingLayerConnectionsComplete()) {
			 prod = getActivation()(prod);
			 prod.print("processData, set output");
			 setOutputs(prod);
		}
}
//----------------------------------------------------------------------
void Layer::forwardLoops()
{
}
//----------------------------------------------------------------------
