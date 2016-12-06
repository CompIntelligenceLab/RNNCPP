#include "layers.h"
#include "print_utils.h"
#include <stdio.h>
#include <iostream>
#include <assert.h>

using namespace std;

int Layer::counter = 0;

Layer::Layer(int layer_size, std::string name /* "layer" */)
{
	this->layer_size = layer_size;
	input_dim   = -1; // no assignment yet. 
	nb_batch    =  1;   // HOW TO SET BATCH FOR LAYERS? 
	seq_len     =  1; 
	print_verbose   = true;
	clock = 0;
	recurrent_conn = 0;

	char cname[80];
	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;

	// Not sure this is still needed
	loop_input.set_size(nb_batch);    // <<<< SOMETHING WRONG? 

	counter++;

	// Input dimension has no significance if the layer is connected to several upstream layers
	// How to access connection to previous layer if this is the first layer? 

	initVars(nb_batch);

	// Default activation: tanh
	activation = new Tanh("tanh");
}
//----------------------------------------------------------------------

void Layer::initVars(int nb_batch)
{
	inputs.set_size(nb_batch);
	outputs.set_size(nb_batch);
	delta.set_size(nb_batch);
	gradient.set_size(nb_batch);
	bias.set_size(layer_size);
	bias_delta.set_size(layer_size);
	//activation_delta.set_size(getActivation().getNbParams());

	// activation may not be set yet. 
	//printf("nb_batch= %d\n", nb_batch); exit(0);

	for (int i=0; i < nb_batch; i++) {
		inputs[i]   = VF2D(layer_size, seq_len);
		outputs[i]  = VF2D(layer_size, seq_len);
		gradient[i] = VF2D(layer_size, seq_len);
		delta[i]    = VF2D(layer_size, seq_len);
	}
	bias.zeros();
	bias_delta.zeros();
	activation_delta.zeros();
	nb_hit = 0;

	// In the future, generalize to a number of temporal links > 1
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

Layer::~Layer()
{
	printf("Layer destructor (%s)\n", name.c_str());
	if (activation) delete activation;
	activation = 0;
	if (recurrent_conn) delete recurrent_conn;
	recurrent_conn = 0;
}

Layer::Layer(const Layer& l) : layer_size(l.layer_size), input_dim(l.input_dim),
   print_verbose(l.print_verbose), seq_len(l.seq_len), nb_batch(l.nb_batch),
   inputs(l.inputs), outputs(l.outputs), clock(l.clock), delta(l.delta), 
	recurrent_conn(l.recurrent_conn)
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
		nb_batch = l.nb_batch;
		clock = l.clock;
		recurrent_conn = l.recurrent_conn;

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
	printf("%sLayer (%s, %s, %s), layer_size: %d\n", msg.c_str(), name.c_str(), activation->getName().c_str(), type.c_str(), layer_size);
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

//----------------------------------------------------------------------
void Layer::incrOutputs(VF2D_F& x)
{
	for (int b=0; b < x.n_rows; b++) {
		outputs(b) += x[b];
	}
}
//----------------------------------------------------------------------
void Layer::incrOutputs(VF2D_F& x, int t)
{
	// I do not know why "this->" is necessary since I do not notice any ambiguity 

	for (int b=0; b < x.n_rows; b++) {
		this->outputs(b).col(t) += x[b].col(t);
	}
}
//----------------------------------------------------------------------
void Layer::incrInputs(VF2D_F& x)
{
	for (int b=0; b < x.n_rows; b++) {
		inputs[b] += x[b];
	}
}
//----------------------------------------------------------------------
void Layer::incrInputs(VF2D_F& x, int t)
{
	for (int b=0; b < x.n_rows; b++) {
		inputs[b].col(t) += x[b].col(t);
	}
}
//----------------------------------------------------------------------
void Layer::resetInputs()
{
	for (int b=0; b < inputs.n_rows; b++) {
		inputs[b].zeros();
	}
}
//----------------------------------------------------------------------
void Layer::resetInputs(int t)
{
	for (int b=0; b < inputs.n_rows; b++) {
		inputs[b].col(t).zeros();
	}
}
//----------------------------------------------------------------------
void Layer::incrDelta(VF2D_F& x)
{
	if (delta[0].n_rows == 0) {
		for (int b=0; b < x.n_rows; b++) {
			delta[b] = x[b];
		}
	} else {
		for (int b=0; b < x.n_rows; b++) {
			delta[b] += x[b];
		}
	}
}
//----------------------------------------------------------------------
void Layer::incrDelta(VF2D_F& x, int t)
{
	if (t < 0) return;

	for (int b=0; b < x.n_rows; b++) {
		delta[b].col(t) += x[b].col(t);
	}
}
//----------------------------------------------------------------------
void Layer::incrBiasDelta(VF1D& x)
{
	bias_delta += x;
}
//----------------------------------------------------------------------
void Layer::incrActivationDelta(VF1D& x)
{
	activation_delta += x;
}
//----------------------------------------------------------------------
void Layer::computeGradient()
{
	//Error. Derivatives must be evaluated for the input argument!
	//gradient = activation->derivative(outputs);
	//gradient = activation->derivative(inputs);

	if (getActivation().getDerivType() == "decoupled") {
		gradient = activation->derivative(inputs);
	} else {
		;  // Do not compute the gradient. Compute the Jacobian on the fly
	}
}
//----------------------------------------------------------------------
void Layer::computeGradient(int t)
{
	// Error. Derivatives must be evaluated for the input argument!
	//gradient = activation->derivative(outputs);
	// Cannot do lazy assignment if using columns

	// More efficient would be: 
	// gradient[b].col(t) = activation->derivative(gradient, inputs, k); 
	//     to avoid a copy. 

	//for (int b=0; b < inputs.n_rows; b++) {
		//gradient[b].col(t) = activation->derivative(inputs[b].col(t));
	//}
	if (getActivation().getDerivType() == "decoupled") {
		for (int b=0; b < inputs.n_rows; b++) {
			gradient[b].col(t) = activation->derivative(inputs[b].col(t));
		}
	} else {
		; // Do not compute the gradient. Compute Jacobian when required
	}
	//printf("gradient, t= %d  .", t); printSummary(); gradient.print("gradient");
}
//----------------------------------------------------------------------
void Layer::forwardData(Connection* conn, VF2D_F& prod, int seq)
{
	// forward data to spatial connections

	const VF2D_F& from_outputs = getOutputs();
	const WEIGHT& wght = conn->getWeight();
	U::matmul(prod, wght, from_outputs);
}
//----------------------------------------------------------------------
bool Layer::areIncomingLayerConnectionsComplete()
{
	return (nb_hit == prev.size());
}
//----------------------------------------------------------------------
void Layer::processOutputDataFromPreviousLayer(Connection* conn, VF2D_F& prod, int t)
{
	//printf("enter Layer::processOutputDataFromPreviousLayer, t= %d\n", t);
	++nb_hit;

	const VF2D_F& from_outputs = conn->from->getOutputs();
	const WEIGHT& wght = conn->getWeight();  // what if connection operates differently
	VF2D_F& to_inputs = layer_inputs[conn->which_lc];

	U::matmul(to_inputs, wght, from_outputs, t, t);  // w * x

	// completeness only happens once per layer and per input value into the predicition module
	if (areIncomingLayerConnectionsComplete()) {
		 // sum up all the inputs + the temporal input if there is one
		 resetInputs(t);   // incorrect answer for a10, correct for a00
		 for (int i=0; i < layer_inputs.size(); i++) {
		 	incrInputs(layer_inputs[i], t);
		 }
		 // add the self-looping if there. 
		 //loop_input[0].raw_print(cout, "layer, loop input");
		 incrInputs(loop_input, t); 

		 // add all other temporal links
		 // ......

		 // Add layer biases. must loop over batch and over sequence size. 
		 addBiasToInput(t);

		 prod.reset();  // Should avoid memory leaks
		 prod = getActivation()(getInputs());

		 setOutputs(prod); 
	}
	return;
}
//----------------------------------------------------------------------
void Layer::processData(Connection* conn, VF2D_F& prod)
{
		++nb_hit;

		VF2D_F& to_inputs = layer_inputs[conn->which_lc];
		to_inputs = prod;

		// Where are the various inputs added up? So derivatives will work if layer_size=1, but not otherwise. 

		if (areIncomingLayerConnectionsComplete()) {
			 prod.reset(); 
			 prod = getActivation()(prod);
			 setOutputs(prod);
		}
}
//----------------------------------------------------------------------
void Layer::forwardLoops()
{ }
void Layer::forwardLoops(int t)
{ }
void Layer::forwardLoops(int t1, int t2)
{ }
void Layer::forwardLoops(Connection* con, int t)
{
	//printf("inside forward loops, t=%d\n", t);
	// forward data to temporal connections
	// handle self loop
	const WEIGHT& wght = con->getWeight();
	//wght.printSummary("wght"); 
	//wght.print("wght"); printf("row/col= %d, %d\n", wght.n_rows, wght.n_cols);
	//this->printSummary();
	//U::print(wght, "wght");
	//U::print(loop_input, "loop_input"); //loop_input.print("loop_input");
	//U::print(outputs, "outputs"); //outputs.print("outputs");
	//printf("loop_input[b].col[t+1] = wght * outputs[b].col[t]\n");
	//outputs[0].raw_print(cout, "loop, outputs");

	// loop_input = wght * outputs

	// For now, assume that a layer can have a maximum of one temporal input
	if (t >= 0) {
		U::matmul(loop_input, wght, outputs, t, t+1); // out of bounds
	}
}
void Layer::forwardLoops(Connection* con, int t1, int t2)
{
	//printf("inside forward loops, t=%d\n", t);
	// forward data to temporal connections
	// handle self loop
	const WEIGHT& wght = con->getWeight();
	U::matmul(loop_input, wght, outputs, t1, t2); // out of bounds
}
//----------------------------------------------------------------------
void Layer::reset() // Must I reset recurrent connection? 
{
	U::zeros(inputs);
	U::zeros(loop_input);
	U::zeros(outputs);
	U::zeros(delta);
	U::zeros(gradient);
	clock = 0;
	//printf("Layer (%s), delta size: %d\n", name.c_str(), delta.n_rows);
}

void Layer::resetBackprop()
{
	for (int b=0; b < delta.size(); b++) {
		delta(b).zeros();
	}
}
void Layer::resetState()
{
	U::zeros(inputs);
	U::zeros(loop_input);
	U::zeros(outputs);
	U::zeros(delta);
	bias_delta.zeros();

	//getOutputs().print("layer reset outputs");

	for (int i=0; i < layer_inputs.size(); i++) {
		U::zeros(layer_inputs[i]);
		U::zeros(layer_deltas[i]);
	}
}
//----------------------------------------------------------------------
void Layer::resetDelta()
{
	U::zeros(delta);
	activation_delta.zeros();
}
//----------------------------------------------------------------------
void Layer::addBiasToInput(int t)
{
	for (int b=0; b < nb_batch; b++) {
		inputs(b).col(t) += bias;
	}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
