#include "weights.h"
#include <stdio.h>
//#include <armadillo_bits/arma_rng.hpp>

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

Weights& Weights::operator=(const Weights& w)
{
    static int counter;
    std::string name;
    WEIGHTS* weights;
    int in_dim, out_dim;
    bool print_verbose;


	if (this != &w) {
		name = w.name + "c";
		in_dim = w.in_dim;
		out_dim = w.out_dim;
		print_verbose = w.print_verbose;

		WEIGHTS* w1 = 0;

		try {
			//w1 = new WEIGHTS(w.weights->n_rows, w.weights->n_cols);
			w1 = new WEIGHTS(*w.weights); // copy constructor
		} catch (...) {
			delete w1;
			throw;
		}

		delete weights; // what if weights is 0? 
		*weights = *w1;
	}

	return *this;
}

void Weights::initializeWeights(std::string initialize_type)
{
	if (initialize_type == "gaussian") {
	} else if (initialize_type == "uniform") {
		printf("in_dim, out_dim= %d, %d\n", in_dim, out_dim);
		//arma_rng::set_seed_random(); // put at beginning of code // DOES NOT WORK
		//arma::Mat<float> ww = arma::randu<arma::Mat<float> >(3, 4); //arma::size(*weights));
		*weights = arma::randu<WEIGHTS>(arma::size(*weights)); //arma::size(*weights));
		printf("weights: %f\n", (*weights)[0,0]);
		*weights = arma::randu<WEIGHTS>(arma::size(*weights)); //arma::size(*weights));
		printf("weights: %f\n", (*weights)[0,0]);
		*weights = arma::randu<WEIGHTS>(arma::size(*weights)); //arma::size(*weights));
		printf("weights: %f\n", (*weights)[0,0]);
		*weights = arma::randu<WEIGHTS>(arma::size(*weights)); //arma::size(*weights));
		printf("weights: %f\n", (*weights)[0,0]);
		printf("weights size: %d\n", weights->size());
		printf("rows, col= %d, %d\n", weights->n_rows, weights->n_cols);
	} else if (initialize_type == "orthogonal") {
	} else {
		printf("initialize_type: %s not implemented\n", initialize_type.c_str());
		exit(1);
	}
}

void Weights::print(std::string msg)
{
	printf("weights: %s\n", name.c_str());
	printf("in: %d, out: %d\n", in_dim, out_dim);
	printf("print_verbose= %d", print_verbose);
	if (msg != "") printf("%s\n", msg.c_str());

	if (print_verbose == false) return;
}
