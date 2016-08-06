#include "model.h"

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
void Model::print()
{
	printf("*** Model printout: ***\n");
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
//----------------------------------------------------------------------