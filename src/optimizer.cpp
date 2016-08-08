#include "optimizer.h"
#include <stdio.h>

int Optimizer::counter = 0;

Optimizer::Optimizer(std::string name)
{
	char cname[80];

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("Optimizer constructor (%s)\n", this->name.c_str());

	this->learning_rate = 1.e-5;
}

Optimizer::~Optimizer()
{
	printf("Optimization destructor (%s)\n", name.c_str());
}

Optimizer::Optimizer(const Optimizer& o) : learning_rate(o.learning_rate)
{
	loss = o.loss;
	name = o.name + "c";
	printf("Optimizer copy constructor (%s)\n", name.c_str());
}

Optimizer& Optimizer::operator=(const Optimizer& o) 
{
	if (this != &o) {
		learning_rate = o.learning_rate;
		name = o.name + "=";
		loss = o.loss;
		printf("Optimizer::operator= (%s)\n", name.c_str());
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

