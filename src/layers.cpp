#include "layers.h"
#include <stdio.h>

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

	// Input dimension has no significance if the layer is connect to several upstream layers
	// How to access connection to previous layer if this is the first layer? 

	this->layer_size = layer_size;
	//output_dim   = layer_size;  // no such member
	input_dim   = -1; // no assignment yet. 
	nb_batch    = 1; 
	seq_len     = 1; 
	print_verbose   = true;
	clock = 0;

	// Default activation: tanh
	activation = new Tanh("tanh");
}

Layer::~Layer()
{
	printf("Layer destructor (%s)\n", name.c_str());
	if (activation) delete activation;
	activation = 0;
}

Layer::Layer(const Layer& l) : layer_size(l.layer_size), input_dim(l.input_dim),
   print_verbose(l.print_verbose), seq_len(l.seq_len), nb_batch(l.nb_batch),
   inputs(l.inputs), outputs(l.outputs), clock(l.clock)
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
	printf("inputs size: %d\n", inputs.size());
	printf("outputs size: %d\n", outputs.size());
	printf("print_verbose: %d\n", print_verbose);

	if (print_verbose == false) return;
}

// Take the inputs (dimensionality based on previous layer), and generate the outputs
void Layer::execute()
{
	outputs = (*activation)(inputs);
}

