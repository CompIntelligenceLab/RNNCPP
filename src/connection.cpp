#include <stdio.h>
#include "connection.h"
#include "print_utils.h"

using namespace std;

int Connection::counter = 0;

Connection::Connection(int in, int out, std::string name /* "weight" */)
{
	in_dim = in;
	out_dim = out;
	weight = WEIGHT(out_dim, in_dim);
	//printf("weight(%d, %d)\n", out_dim, in_dim);
	print_verbose = true;
	temporal = false; // all connections false for feedforward networks
	clock = 0;
	from = to = 0;

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
     temporal(w.temporal), clock(w.clock), to(w.to), from(w.from)
{
	name = w.name + "c";
	weight = WEIGHT(out_dim, in_dim);
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
		weight   = w.weight; 
		clock = w.clock;
		from = w.from; 
		to = w.to;
		printf("Connection::operator= (%s)\n", name.c_str());
	}

	return *this;
}

void Connection::print(std::string msg /* "" */)
{
	printf("weight: %s\n", name.c_str());
	printf("in: %d, out: %d\n", in_dim, out_dim);
	printf("print_verbose= %d\n", print_verbose);
	if (msg != "") printf("%s\n", msg.c_str());

	if (print_verbose == false) return;
}

void Connection::printSummary(std::string msg) 
{
	std::string type = (temporal) ? "temporal" : "spatial";
	cout << msg << "Connection (" << name << "), weight(" << weight.n_rows << ", " << weight.n_cols << "), " << type << endl;
	//printf("%sConnection (%s), weight(%d, %d), %s\n", msg.c_str(), name.c_str(), weight.n_rows, weight.n_cols, type.c_str());
}

//----------------------------------------------------------------------

Connection Connection::operator+(const Connection& w) 
{
	Connection tmp(*this);  // Ideally, this should initialize all components
	printf("after tmp declaration and definition\n");

	U::print(weight, "weight");
	//printf("weight.size= %d, %d", weight.n_rows, weight.n_cols);
	//printf("weight.size= %d", weight.size()); // n_rows * n_cols

	tmp.weight += w.weight;

	return tmp;
};

Connection Connection::operator*(const Connection& w) 
{
	Connection tmp(*this);  // Ideally, this should initialize all components
	printf("after tmp declaration and definition\n");

	tmp.weight = tmp.weight * w.weight;

	return tmp;
};

VF2D_F Connection::operator*(const VF2D_F& x)
{
    // w * x  ==> w(layer[k], layer[k-1]) * x[batch](dim, seq)
	int nb_batch = x.n_rows;
	int dim      = x[0].n_rows;   // if x[0] exists
	int nb_seq   = x[0].n_cols;   // if x[0] exists

	VF2D_F tmp(nb_batch);  // Ideally, this should initialize all components

	for (int i=0; i < nb_batch; i++) {
		tmp[i] = this->weight * x[i]; // benchmark for large arrays
	}

	return tmp;
}

void Connection::initialize(std::string initialize_type /*"uniform"*/ )
{
	U::print(weight, "--  Connection::initialize, weight");
	clock = 0;

	if (initialize_type == "gaussian") {
	} else if (initialize_type == "uniform") {
		//arma_rng::set_seed_random(); // put at beginning of code // DOES NOT WORK
		//arma::Mat<float> ww = arma::randu<arma::Mat<float> >(3, 4); //arma::size(*weight));
		weight = arma::randu<WEIGHT>(arma::size(weight)); //arma::size(*weight));
		weight = arma::randu<WEIGHT>(arma::size(weight)); //arma::size(*weight));
		//weight.print("initializeConnection");
	} else if (initialize_type == "orthogonal") {
	} else {
		printf("initialize_type: %s not implemented\n", initialize_type.c_str());
		exit(1);
	}
}

void Connection::weightUpdate(float learning_rate) {  // simplest algorithm
	for (int b=0; b < delta.n_rows;  b++) {
		weight = weight - learning_rate * delta;
	}
}

