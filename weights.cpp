#include "weights.h"

Weights::Weights(int in, int out)
{
	in_dim = in;
	out_dim = out;
	weights = new WEIGHTS(in_dim, out_dim);
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
