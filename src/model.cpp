#include "model.h"
#include "objective.h"
#include <stdio.h>
#include <armadillo>
#include "model.h"

Model::Model(int input_dim, std::string name)
{
	this->name = name;
	learning_rate = 1.e-5;
	return_sequences = false;
	print_verbose = true;
	this->input_dim = input_dim;
}
//----------------------------------------------------------------------
Model::~Model()
{
	printf("Model destructor (%s)\n", name.c_str());
	for (int i=0; i < layers.size(); i++) {
		delete layers[i];
	}
}
//----------------------------------------------------------------------
Model::Model(const Model& m) : name(m.name), stateful(m.stateful), learning_rate(m.learning_rate), 
    return_sequences(m.return_sequences), input_dim(m.input_dim), batch_size(m.batch_size),
	seq_len(m.seq_len), print_verbose(m.print_verbose), initialization_type(m.initialization_type)

	// What to do with name (perhaps add a "c" at the end for copy-construcor?)
{
	optimizer = new Optimizer();
	*optimizer = *m.optimizer;
	Objective* loss = new Objective(); // ERROR
	//loss->setName(m.loss->getName() + "c");
	layers = m.layers;
	LAYERS layers; // operate by value for safety)
}
//----------------------------------------------------------------------
Model& Model::operator=(const Model& m) 
{
	printf("Model::operator= %s\n", name.c_str());

	if (this != &m) {
		name = m.name;
		stateful = m.stateful;
		learning_rate = m.learning_rate;
		return_sequences = m.return_sequences;
		input_dim = m.input_dim;
		batch_size = m.batch_size;
		seq_len = m.seq_len;
		print_verbose= m.print_verbose;
		initialization_type = m.initialization_type;

		Optimizer* opt1 = 0;
		Objective* loss1 = 0;

		try {
			opt1 = new Optimizer(*m.optimizer);
			loss1 = new Objective(*m.loss);
		} catch (...) {
			delete opt1;
			delete loss1;
			throw;
		}

		// Superclass::operator=(that)
		delete optimizer;
		optimizer = opt1;
		delete loss;;
		loss = loss1;
	}
	return *this;
}
//----------------------------------------------------------------------
void Model::add(Layer* layer)
{
	layers.push_back(layer);
}
//----------------------------------------------------------------------
void Model::setOptimizer(Optimizer* opt)
{
	optimizer = opt;
}
//----------------------------------------------------------------------
Optimizer* Model::getOptimizer()
{
	return optimizer;
}
//----------------------------------------------------------------------
void Model::setStateful(bool stateful)
{
	this->stateful = stateful;
}
//----------------------------------------------------------------------
bool Model::getStateful()
{
	return stateful;
}
//----------------------------------------------------------------------
void Model::setReturnSequences(bool ret_seq)
{
	return_sequences = ret_seq;
}
//----------------------------------------------------------------------
bool Model::getReturnSequences()
{
	return return_sequences;
}
//----------------------------------------------------------------------
void Model::setLearningRate(float lr)
{
	learning_rate = lr;
}
//----------------------------------------------------------------------
float Model::getLearningRate()
{
	return learning_rate;
}
//----------------------------------------------------------------------
void Model::print(std::string msg)
{
	printf("*** Model printout: ***\n");
    if (msg != "") printf("%s\n", msg.c_str());
	printf("name: %s\n", name.c_str());
	printf("stateful: %d\n", stateful);
	printf("learning_rate: %f\n", learning_rate);
	printf("return_sequences: %d\n", return_sequences);
	printf("print_verbose: %d\n", print_verbose);
	optimizer->print();
	loss->print();

	if (print_verbose == false) return;

	for (int i=0; i < layers.size(); i++) {
		layers[i]->print();
	}
}
//----------------------------------------------------------------------
void Model::predict(VF3D x)
{
	/* x is input to the network */
	/* Must first initialize the weights  for the layers */

	// input to layer 0
	WeightList& wl = layers[0]->getWeights();
	printf("predict\n");
	WEIGHTS& wght = *(wl[0].getWeights()); // for now, only a single weight
/* */
	//VF3D y = x * wght;   // Mat<float> * cube<float>  (in,out) * (batch,seq,dim)
	                     //   sum(over in): (i,o) * (b,s,i)
	                     // = sum(over in): (b,s,i) * (i,o) = x * W = f(b,s,o)
						 // Armadillo does not allow multiplication of a cube*matrix on the inner index.
						 // For now, use a loop. for didactic purposes. 
	VF3D prod(5,6,7);
	for (int b=0; b < x.n_rows; b++) {
		for (int s=0; s < x.n_cols; s++) {
			for (int o=0; o < x.n_slices; o++) {
				for (int i=0; i < wght.n_rows; i++) {
					prod(b,s,o) += x(b,s,i) * wght(i,o);
				}
				printf("prod(%d,%d,%d)= %f\n", b,s,o, prod(b,s,o));
			}
		}
	}
/* */

	/*
	VF3D& xx = x;
	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i];
		VF3D& y = layer->execute(xx);
		VF3D& xx = y;
	}
	*/
}
//----------------------------------------------------------------------
void Model::initializeWeights(std::string initialization_type)
{
	int in_dim, out_dim;

	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i];
		in_dim = (i == 0) ? input_dim : layers[i-1]->getInputDim();
		out_dim = layer->getInputDim();
		printf("Model::init weights: layer %d, in_dim, out_dim= %d, %d\n", i, in_dim, out_dim);
		layer->createWeights(in_dim, out_dim);
		layer->initializeWeights(initialization_type);
	}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
