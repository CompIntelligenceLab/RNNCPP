#include "weights.h"
#include <stdio.h>

int Weights::counter = 0;

Weights::Weights(int in, int out, std::string name)
{
	printf("weights constructor\n");
	in_dim = in;
	out_dim = out;
	weights = new WEIGHTS(in_dim, out_dim);
	print_verbose = true;

	char cname[80];

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	counter++;
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
	printf("in: %d, out: %d\n", in_dim, out_dim);
	printf("print_verbose= %d", print_verbose);
	if (msg != "") printf("%s\n", msg.c_str());

	if (print_verbose == false) return;
}
