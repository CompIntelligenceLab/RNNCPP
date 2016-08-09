#include "model.h"
#include "objective.h"
#include <stdio.h>
#include <armadillo>

Model::Model(int input_dim, std::string name /* "model" */) 
{
	this->name = name;
	learning_rate = 1.e-5;
	return_sequences = false;
	print_verbose = true;
	this->input_dim = input_dim;
	printf("Model constructor (%s)\n", this->name.c_str());
  optimizer = NULL;
  loss = NULL; // I believe this should just be a string or something specifying
               // the name of the objective function we would like to use
  batch_size = 1; // batch size of 1 is equivalent to online learning
  nb_epoch = 10; // Just using the value that Keras uses for now
  //stateful = false;
	//int seq_len;
	//initialization_type;
	//LAYERS layers; // operate by value for safety)

}
//----------------------------------------------------------------------
Model::~Model()
{
	printf("Model destructor (%s)\n", name.c_str());

	for (int i=0; i < layers.size(); i++) {
		if (layers[i]) {delete layers[i]; layers[i] = 0;}
	}
}
//----------------------------------------------------------------------
Model::Model(const Model& m) : stateful(m.stateful), learning_rate(m.learning_rate), 
    return_sequences(m.return_sequences), input_dim(m.input_dim), batch_size(m.batch_size),
	seq_len(m.seq_len), print_verbose(m.print_verbose), initialization_type(m.initialization_type),
  nb_epoch(m.nb_epoch)

	// What to do with name (perhaps add a "c" at the end for copy-construcor?)
{
	name = m.name + "c";
	optimizer = new Optimizer();
  *optimizer = *m.optimizer;// Careful here, we need to implement a copy
                            // assignment operator for the Optimizer class
	loss = new Objective(); // ERROR
  *loss = *m.loss;// Careful here, we need to implement a copy
                  // assignment operator for the Optimizer class
	layers = m.layers; // Careful here, we need to implement a copy
                     // assignment operator for the Optimizer class
	printf("Model copy constructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
Model& Model::operator=(const Model& m) 
{
	if (this != &m) {
		name = m.name + "=";
		stateful = m.stateful;
		learning_rate = m.learning_rate;
		return_sequences = m.return_sequences;
		input_dim = m.input_dim;
		batch_size = m.batch_size;
    nb_epoch = m.nb_epoch;
		seq_len = m.seq_len;
		print_verbose= m.print_verbose;
		initialization_type = m.initialization_type;

		Optimizer* opt1 = NULL;
		Objective* loss1 = NULL;

		try {
			opt1 = new Optimizer(*m.optimizer);
			loss1 = new Objective(*m.loss);
		} catch (...) {
			delete opt1;
			delete loss1;
			printf("Model throw\n");
			throw;
		}

		// Superclass::operator=(that)
		optimizer = opt1;
		loss = loss1;
		printf("Model::operator= %s\n", name.c_str());
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
void Model::print(std::string msg /* "" */)
{
	printf("*** Model printout: ***\n");
    if (msg != "") printf("%s\n", msg.c_str());
	printf("name: %s\n", name.c_str());
	printf("stateful: %d\n", stateful);
	printf("learning_rate: %f\n", learning_rate);
	printf("return_sequences: %d\n", return_sequences);
	printf("print_verbose: %d\n", print_verbose);

  if (optimizer != NULL) 
	  optimizer->print();
  if (loss != NULL)
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
// This was hastily decided on primarily as a means to construct feed forward
// results to begin implementing the backprop. Should be reevaluated
void Model::train(MATRIX x, MATRIX y, int batch_size /*=0*/, int nb_epoch /*=0*/) 
{
  if (batch_size == 0) // Means no value for batch_size was passed into this function
    batch_size = this->batch_size; // Use the current value stored in model

  // First we should construct the input for the predict routine 
  VF3D input;
}
//----------------------------------------------------------------------
void Model::initializeWeights(std::string initialization_type /* "uniform" */)
{
	int in_dim, out_dim;
	printf("inside initialize\n");

	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i];
		in_dim = (i == 0) ? input_dim : layers[i-1]->getInputDim();
		out_dim = layer->getInputDim();
		printf("layer %d, in_dim, out_dim= %d, %d\n", i, in_dim, out_dim);
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
