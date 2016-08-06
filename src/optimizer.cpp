#include "optimizer.h"

Optimizer::Optimizer(std::string name)
{
	this->name = name;
	this->learning_rate = learning_rate;
}

Optimizer::~Optimizer()
{
}

Optimizer::Optimizer(Optimizer&)
{
}

void Optimizer::print()
{
	print("Optimizer: %s\n", name->c_str());
	print("learning_rate: %f\n", learning_rate);
}
