#include "layers.h"
#include <stdio.h>

int Layer::counter = 0;

Layer::Layer(int layer_size, std::string name) : input_dim(3)
{
	char cname[80];

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	counter++;

	this->layer_size = layer_size;
	int batch_size  = 1;  // default value
	int seq_len     = 1; // default value
	int input_dim   = 1; // default: scalar
	weights         = new Weights(1,1, "weights"); // default size
	printf("weights= %ld\n", weights); 
	//printf("weights: %d, %d\n", weights->getWeights().n_rows, weights->getWeights().n_cols); exit(0);
	print_verbose   = true;

	// Default activation: tanh
	activation = new Tanh("tanh");
}

Layer::~Layer()
{
	printf("Layer destructor (%s)\n", name.c_str());
	delete weights;
	delete activation;
}

Layer::Layer(const Layer& l) : name(l.name), seq_len(l.seq_len), 
   batch_size(l.batch_size), layer_size(l.layer_size), input_dim(l.input_dim), 
   print_verbose(l.print_verbose)
{
	printf("Layer copy constructor (%s)\n", l.name.c_str());
	inputs = l.inputs;
	outputs = l.outputs;
	weights = l.weights; 

	//TODO
	// How does activation work with polymorphism ?
	//activation = new Activation(); // HOW TO DO?   (does not work)
	//*activation = *l.activation; // check name
	//print("Layer copy constructor, activation->name= ", activation->name);
}

Layer& Layer::operator=(const Layer& l)
{
	printf("Layer::operator= (%s)\n", name.c_str());

	if (this != &l) {
		name = l.name;
		seq_len = l.seq_len;
		batch_size = l.batch_size;
		layer_size = l.layer_size;
		input_dim = l.input_dim;
		print_verbose = l.print_verbose;
	
		//Weights* weights;
		//Activation* activation;

		Weights* w1;
		Activation *a1;

		try {
			w1 = new Weights(*l.weights); // copy constructor
		} catch (...) {
			delete w1;
			throw;
		}

		delete weights; // what if weights is 0? 
		*weights = *w1;
	}

	return *this;
}

void Layer::print(std::string msg)
{
	printf("  -- layer: %s ---\n", name.c_str());
    if (msg != "") printf("%s\n", msg.c_str());
	printf("seq len: %d\n", seq_len);
	printf("batch_size: %d\n", batch_size);
	printf("layer size: %d\n", layer_size);
	printf("input_dim: %d\n", input_dim);
	printf("inputs size: %d\n", inputs.size());
	printf("outputs size: %d\n", outputs.size());
	printf("print_verbose: %d\n", print_verbose);

	if (print_verbose == false) return;

	if (weights) {
		weights->print();
	}

	//if (activation) {
		//activation->print();
	//}
}

// Take the inputs (dimensionality based on previous layer), and generate the outputs
void Layer::execute()
{
	outputs = (*activation)(inputs);
}

void Layer::createWeights(int in, int out)
{
	weights = new Weights(in, out, this->name+"_"+"weights");
}

void Layer::initializeWeights(std::string initialize_type)
{
		weights->initializeWeights(initialize_type);
}
