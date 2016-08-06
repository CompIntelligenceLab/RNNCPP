#include "weights.h"

Weights::Weights(int in, int out, std::string name)
{
	printf("weights constructor\n");
	in_dim = in;
	out_dim = out;
	weights = new WEIGHTS(in_dim, out_dim);
	this->name = name;
}

Weights::~Weights()
{
}

Weights::Weights(Weights& w) 
{
}

void Weights::initialize() 
{
}

void Weights::print(std::string msg)
{
	printf("weights: %s\n", name.c_str());
	if (msg != "") printf("%s\n", msg.c_str());
}
