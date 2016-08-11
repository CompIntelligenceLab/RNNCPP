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

	this->layer_size = layer_size;
	input_dim   = 1; // default: scalar
	nb_batch    = 1; 
	seq_len     = 1; 
	//weights         = new Weights(1,1, "weights"); // default size
	weights    = WEIGHTS(1,1);
	print_verbose   = true;

	// Default activation: tanh
	activation = new Tanh("tanh");
}

Layer::~Layer()
{
	printf("Layer destructor (%s)\n", name.c_str());
	//if (weights)    delete weights; // no longer required
	if (activation) delete activation;
	//weights = 0; 
	activation = 0;
}

Layer::Layer(const Layer& l) : layer_size(l.layer_size), input_dim(l.input_dim),
   print_verbose(l.print_verbose), seq_len(l.seq_len), nb_batch(l.nb_batch),
   inputs(l.inputs), outputs(l.outputs), weights(l.weights)
{
	//weights = new Weights(1,1, "weights_c"); // remove class Weights (for now)
	//*weights = *l.weights;  // removed because of Nathan simplification (remove class Weights)
	//weights = WEIGHTS(l.weights); // for Nathan's changes
	//weights = l.weights;
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
		weights = l.weights;
		inputs = l.inputs;
		outputs = l.outputs;
		seq_len = l.seq_len;
		nb_batch = l.seq_len;

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

		//delete weights; // what if weights is 0? 
		*activation = *a1; 
		// remove class weights 

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

	#if 0
	// Weight class removed
	if (weights) {
		weights->print();
	}
	#endif

	weights.print();

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
	// weights = new Weights(in, out, this->name+"_"+"weights"); // Nathan
	weights = WEIGHTS(out, in);
}

void Layer::initializeWeights(std::string initialize_type /* "uniform" */)
{
		//weights->initializeWeights(initialize_type); // Nathan
	//printf("-- Layer::initializeWeights, in_dim, out_dim= %d, %d\n", in_dim, out_dim);
	printf("--  Layer::weights size: %d, %d\n", weights.n_rows, weights.n_cols);

	if (initialize_type == "gaussian") {
	} else if (initialize_type == "uniform") {
		//printf("in_dim, out_dim= %d, %d\n", in_dim, out_dim);
		//printf("in_dim, out_dim= %d, %d\n", weights.n_rows, weights.n_cols); 
		//arma_rng::set_seed_random(); // put at beginning of code // DOES NOT WORK
		//arma::Mat<float> ww = arma::randu<arma::Mat<float> >(3, 4); //arma::size(*weights));
		weights = arma::randu<WEIGHTS>(arma::size(weights)); //arma::size(*weights));
		printf("weights: %f\n", weights[0,0]);
		weights = arma::randu<WEIGHTS>(arma::size(weights)); //arma::size(*weights));
		printf("weights: %f\n", weights[0,0]);
		printf("weights size: %d\n", weights.size());
		printf("rows, col= %d, %d\n", weights.n_rows, weights.n_cols);
	} else if (initialize_type == "orthogonal") {
	} else {
		printf("initialize_type: %s not implemented\n", initialize_type.c_str());
		exit(1);
	}
}
