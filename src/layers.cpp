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
	weights         = 0;
	print_verbose   = true;

	// Default activation: tanh
	activation = new Tanh("tanh");
}

Layer::~Layer()
{
	delete weights;
	delete activation;
}

Layer::Layer(Layer&)
{
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

	if (activation) {
		activation->print();
	}
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
		if (initialize_type == "gaussian") {
		} else if (initialize_type == "uniform") {
		} else if (initialize_type == "orthogonal") {
		} else {
			printf("initialize_type: %s not implemented\n", initialize_type.c_str());
			exit(1);
		}
}
