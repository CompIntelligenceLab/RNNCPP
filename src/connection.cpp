#include "connection.h"
#include <stdio.h>

int Connection::counter = 0;

Connection::Connection(int in, int out, std::string name /* "weights" */)
{
	in_dim = in;
	out_dim = out;
	weights = WEIGHTS(in_dim, out_dim);
	weights_f = WEIGHTS_F(1);
	weights_f[0] = WEIGHTS(in_dim, out_dim);   // 1 refers to batch_size
	print_verbose = true;
	temporal = false; // all connections false for feedforward networks

	char cname[80];

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("Connection constructor, in= %d, out=%d (%s)\n", in, out, this->name.c_str());
	counter++;
}

Connection::~Connection()
{
	printf("Connection destructor (%s)\n", name.c_str());
}

Connection::Connection(const Connection& w) : in_dim(w.in_dim), out_dim(w.out_dim), print_verbose(w.print_verbose),
     temporal(w.temporal)
{
	name = w.name + "c";
	weights = WEIGHTS(in_dim, out_dim);
	weights_f = w.weights_f; 
	printf("Connection::copy_constructor (%s)\n", name.c_str());
}

Connection::Connection(Connection&& w) = default;

const Connection& Connection::operator=(const Connection& w)
{
	if (this != &w) {
		name = w.name + "=";
		in_dim = w.in_dim;
		out_dim = w.out_dim;
		print_verbose = w.print_verbose;
		temporal = w.temporal;
		//printf("size w.weights: %d\n", w.weights.size());
		//printf("weights: %d\n", weights.size());
		weights   = w.weights; 
		weights_f = w.weights_f; 
		printf("Connection::operator= (%s)\n", name.c_str());
	}

	return *this;
}

void Connection::print(std::string msg /* "" */)
{
	printf("weights: %s\n", name.c_str());
	printf("in: %d, out: %d\n", in_dim, out_dim);
	printf("print_verbose= %d\n", print_verbose);
	if (msg != "") printf("%s\n", msg.c_str());

	if (print_verbose == false) return;
}

Connection Connection::operator+(const Connection& w) 
{
	Connection tmp(*this);  // Ideally, this should initialize all components, including weights_f
	printf("after tmp declaration and definition\n");

	printf("weights.size= %d, %d", weights.n_rows, weights.n_cols);
	printf("weights.size= %d", weights.size()); // n_rows * n_cols

	tmp.weights += w.weights;

	for (int i=0; i < w.weights_f.size(); i++) {
		tmp.weights_f[i] += w.weights_f[i];
	}
		
	return tmp;
};

Connection Connection::operator*(const Connection& w) 
{
	Connection tmp(*this);  // Ideally, this should initialize all components, including weights_f
	printf("after tmp declaration and definition\n");

	tmp.weights = tmp.weights * w.weights;

	for (int i=0; i < w.weights_f.size(); i++) {
		tmp.weights_f[i] = this->weights_f[i] * w.weights_f[i];
	}
		
	return tmp;
};

VF2D_F Connection::operator*(const VF2D_F& x)
{
    // w * x  ==> w(layer[k], layer[k-1]) * x[batch](dim, seq)
	int nb_batch = x.n_rows;
	int dim      = x[0].n_rows;   // if x[0] exists
	int nb_seq   = x[0].n_cols;   // if x[0] exists

	VF2D_F tmp(nb_batch);  // Ideally, this should initialize all components, including weights_f

	for (int i=0; i < nb_batch; i++) {
		tmp[i] = this->weights * x[i]; // benchmark for large arrays
	}

	return tmp;
}

void Connection::initialize(std::string initialize_type /*"uniform"*/ )
{
	printf("--  Connection::initialize size: %d, %d\n", weights.n_rows, weights.n_cols);

	if (initialize_type == "gaussian") {
	} else if (initialize_type == "uniform") {
		//printf("in_dim, out_dim= %d, %d\n", in_dim, out_dim);
		//printf("in_dim, out_dim= %d, %d\n", weights.n_rows, weights.n_cols); 
		//arma_rng::set_seed_random(); // put at beginning of code // DOES NOT WORK
		//arma::Mat<float> ww = arma::randu<arma::Mat<float> >(3, 4); //arma::size(*weights));
		weights = arma::randu<WEIGHTS>(arma::size(weights)); //arma::size(*weights));
		//printf("weights: %f\n", weights[0,0]);
		weights = arma::randu<WEIGHTS>(arma::size(weights)); //arma::size(*weights));
		//printf("weights: %f\n", weights[0,0]);
		//printf("weights size: %d\n", weights.size());
		//printf("rows, col= %d, %d\n", weights.n_rows, weights.n_cols);
		//weights.print("initializeConnection");
	} else if (initialize_type == "orthogonal") {
	} else {
		printf("initialize_type: %s not implemented\n", initialize_type.c_str());
		exit(1);
	}
}
