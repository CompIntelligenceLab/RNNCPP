#include "optimizer.h"
#include <stdio.h>

Optimizer::Optimizer(std::string name)
{
	this->name = name;
	this->learning_rate = 1.e-5;
}

Optimizer::~Optimizer()
{
}

Optimizer::Optimizer(Optimizer&)
{
}

void Optimizer::print(std::string msg)
{
	printf("Optimizer: %s\n", name.c_str());
	if (msg != "") printf("%s\n", msg.c_str());
	printf("learning_rate: %f\n", learning_rate);
}

//----------------------------------------------------------------------

RMSProp::RMSProp(std::string name) : Optimizer(name) {;}
//----------------------------------------------------------------------

