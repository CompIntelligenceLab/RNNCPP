#include "model.h"
#include <stdio.h>
#include <armadillo>

Model::Model(std::string name)
{
	this->name = name;
	learning_rate = 1.e-5;
	return_sequences = false;
}
//----------------------------------------------------------------------
Model::~Model()
{
	for (int i=0; i < layers.size(); i++) {
		delete layers[i];
	}
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
	//optimizer->print();
	//loss->print();
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
void Model::initialize_weights(std::string initialization_type)
{
	int in_dim, out_dim;

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
