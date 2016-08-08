#include "optimizer.h"
#include <stdio.h>

Optimizer::Optimizer(std::string name)
{
	this->name = name;
	this->learning_rate = 1.e-5;
}

Optimizer::~Optimizer()
{
	printf("Optimization destructor (%s)\n", name.c_str());
}

Optimizer::Optimizer(const Optimizer& o) : learning_rate(o.learning_rate)
{
	printf("Optimizer copy constructor (%s)\n", o.name.c_str());
	loss = o.loss;
	this->name = name + "c";
}

Optimizer& Optimizer::operator=(const Optimizer& o) 
{
	printf("Optimizer::operator= (%s)\n", o.name.c_str());

	if (this != &o) {
		learning_rate = o.learning_rate;
		name = o.name + "=";
		loss = o.loss;
	}
	//printf("exit optimizer=\n");
	return *this;
}

void Optimizer::print(std::string msg)
{
	//printf("enter optimizer print\n"); // How can this print more than once? 
	//printf("xxx\n");
	//printf("Optimizer: %s\n", name.c_str());
	if (msg != "") printf("%s\n", msg.c_str());
	printf("learning_rate: %f\n", learning_rate);
	printf("exit optimizer print\n");
	//exit(0);
}

//----------------------------------------------------------------------

RMSProp::RMSProp(std::string name) : Optimizer(name) {;}
//----------------------------------------------------------------------

