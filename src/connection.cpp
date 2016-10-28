#include <stdio.h>
#include "connection.h"
#include "print_utils.h"
#include "layers.h"

using namespace std;

int Connection::counter = 0;

Connection* Connection::ConnectionFactory(int in_dim, int out_dim, std::string conn_type)
{
	if (conn_type == "all-all") {
		return new Connection(in_dim, out_dim);
	}
}
//----------------------------------------------------------------------
Connection::Connection(int in, int out, std::string name /* "weight" */)
{
	type = "standard";
	in_dim = in;
	out_dim = out;
	weight   = WEIGHT(out_dim, in_dim);
	weight_t = WEIGHT(out_dim, in_dim);
	delta = WEIGHT(out_dim, in_dim);
	t_from = t_to = t_clock = 0;
	print_verbose = true;
	temporal = false; // all connections false for feedforward networks
	clock = 0;
	from = to = 0;
	hit = 0;
	freeze = false;
	type = "all-all";

	char cname[80];

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	//printf("Connection constructor, in= %d, out=%d (%s)\n", in, out, this->name.c_str());
	counter++;
}

Connection::~Connection()
{
	printf("Connection destructor (%s)\n", name.c_str());
}

Connection::Connection(const Connection& w) : in_dim(w.in_dim), out_dim(w.out_dim), print_verbose(w.print_verbose),
     temporal(w.temporal), clock(w.clock), to(w.to), from(w.from), freeze(w.freeze), type(w.type), t_from(w.t_from),
	 t_to(w.t_to), t_clock(w.t_clock)
{
	name = w.name + "c";
	//weight = WEIGHT(out_dim, in_dim);
	weight = w.getWeight();
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
		weight_t = w.weight_t; 
		clock = w.clock;
		from = w.from; 
		to = w.to;
		freeze = w.freeze;
		type = w.type;
		t_from = w.t_from;
		t_to = w.t_to;
		t_clock = w.t_clock;
		printf("Connection::operator= (%s)\n", name.c_str());
	}

	return *this;
}

void Connection::print(std::string msg /* "" */)
{
	printf("connection: %s\n", name.c_str());
	printf("in: %d, out: %d\n", in_dim, out_dim);
	printf("print_verbose= %d\n", print_verbose);
	printf("freeze= %d\n", freeze);
	if (msg != "") printf("%s\n", msg.c_str());

	if (print_verbose == false) return;
}

void Connection::printSummary(std::string msg) 
{
	std::string type = (temporal) ? "temporal" : "spatial";

	string name_from = (from == 0) ?  "NONE" : from->getName(); 
	string name_to   = (to   == 0) ?  "NONE" :   to->getName(); 
	cout << msg << "Connection (" << name << "), weight(" << weight.n_rows << ", " << weight.n_cols << "), " 
	     << "layers: (" << name_from << ", " << name_to << "), type: " << type  
		 << std::endl;
}

//----------------------------------------------------------------------

Connection Connection::operator+(const Connection& w) 
{
	Connection tmp(*this);  // Ideally, this should initialize all components

	//(*this).weight.print("operator+, *this");
	//tmp.weight.print("operator+, tmp");
	//w.weight.print("operator+, w");
	tmp.weight += w.weight;

	//tmp.weight.print("operator+, tmp += w");

	return tmp;
};

Connection Connection::operator*(const Connection& w) 
{
	Connection tmp(*this);  // Ideally, this should initialize all components
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
	//x[0].print("x[i]");
	//weight.print("weight");
	//exit(0);

	for (int i=0; i < nb_batch; i++) {
		tmp[i] = this->weight * x[i]; // benchmark for large arrays
	}

	return tmp;
}

//----------------------------------------------------------------------
void Connection::initialize(std::string initialize_type /*"xavier"*/ )
{
	clock = 0;
	printf("Weight initialization: type: %s\n", initialize_type.c_str());

	if (initialize_type == "gaussian") {
	} else if (initialize_type == "uniform") {
		//arma_rng::set_seed_random(); // put at beginning of code // DOES NOT WORK
		//arma::Mat<REAL> ww = arma::randu<arma::Mat<REAL> >(3, 4); //arma::size(*weight));
		weight = arma::randu<WEIGHT>(arma::size(weight)); //arma::size(*weight));
		//weight.print("initializeConnection");
	} else if (initialize_type == "orthogonal") {
	} else if (initialize_type == "xavier") {
		if (!temporal) {
			// IMPLEMENT XAVIER with UNIFORM DISTRIBUTION 
			//weight = arma::randn<WEIGHT>(arma::size(weight)); //Gaussian N(0,1)
			weight = arma::randu<WEIGHT>(arma::size(weight)); //Gaussian N(0,1)
			// I want the standard deviation to be 1/n
			REAL n_outs = weight.n_rows;   // inputs to layer: connection->to->getLayerSize()
			REAL n_ins  = weight.n_cols;
			n_outs = sqrt(n_outs);
			weight = weight / n_outs;
		} else {  // uniform with values of [-.08, .08]
			weight = arma::randu<WEIGHT>(arma::size(weight)); //Uniform(0,1)
			weight = 2.0*(weight - 0.5) * 0.08;    //Uniform(-.08, 0.08)
			//weight.print("weight temporal");
	    //printf("***\n");exit(0);
		}
	} else if (initialize_type == "xavier_iden") {   // initialize recurrent weights to identity matrix
		weight = arma::randn<WEIGHT>(arma::size(weight)); //Gaussian N(0,1)
		REAL n_outs = weight.n_rows;   // inputs to layer: connection->to->getLayerSize()
		n_outs = sqrt(n_outs);
		weight = weight / n_outs;

		if (temporal) {
			weight.eye(size(weight));
			// Below 0.8 and the errors for long sequences do not accumulate
			// This initialization should only be required for recurrent connections. 
			// The deduction was for a network with a single recurrent layer. More generally, one 
			// only surmises that the results hold for more general recurrent networks. 
			weight *= 0.98; // make matrix slightly stable. 
			//weight *= 1.02; // make matrix slightly stable. 
			weight.print("weight temporal");
		}
	} else {
		printf("initialize_type: %s not implemented\n", initialize_type.c_str());
		exit(1);
	}
	weight_t = weight.t();
}

//----------------------------------------------------------------------
void Connection::weightUpdate(REAL learning_rate) {  // simplest algorithm
	// delta is of type WEIGHT, which is not a field, but a VF2D
	//for (int b=0; b < delta.n_rows;  b++) {    
		weight = weight - learning_rate * delta;
	//}
	weight_t = weight.t();
}

//----------------------------------------------------------------------
void Connection::incrDelta(WEIGHT& x)
{
	if (delta.n_rows == 0) {
		delta = x;
	} else {
		delta += x;
	}
}
//----------------------------------------------------------------------
void Connection::computeWeightTranspose()
{
	weight_t = weight.t();
}
//----------------------------------------------------------------------
void Connection::gradMulDLda(int ti_from, int ti_to)
{
	//assert(this == conn.to);
	Layer* layer_to   = to;
	Layer* layer_from = from;
	int nb_batch = layer_from->getNbBatch(); 
	VF2D_F prod(nb_batch);

	const VF2D_F& old_deriv = layer_to->getDelta();  // 3
	const WEIGHT& wght   = getWeight(); // invokes copy constructor, or what?  3 x 4

	Activation& activation = layer_to->getActivation();

	// CHECK AGAINST EXACT, ANALYTICAL!!! 

	if (activation.getDerivType() == "decoupled") {   
		//printf("gradMulDLda, decoupled\n");
		const VF2D_F& grad 		= layer_to->getGradient();
		for (int b=0; b < nb_batch; b++) {
			prod(b) = VF2D(size(layer_from->getDelta()(0)));
		}
		const WEIGHT& wght_t = getWeightTranspose();
		U::rightTriad(prod, wght_t, grad, old_deriv, ti_from, ti_to);  // dL/da
		//this->printSummary();
		//printf("Connection::gradMulDLda, "); prod.print("prod = dL/da");
	} else { // "coupled"
		//printf("gradMulDLda, coupled\n");
		for (int b=0; b < nb_batch; b++) {
			const VF1D& x   =  layer_to->getInputs()(b).col(ti_from);
			const VF1D& y   = layer_to->getOutputs()(b).col(ti_from);
			const VF2D grad = activation.jacobian(x, y); // not stored (3,3)
			// parentheses required to ensure that left hand multiplication occurs first
			// since old_deriv is a vector and grad/wght are matrices
			const VF2D& gg = (old_deriv[b].col(ti_from).t() * grad) * wght; 
			prod(b) = VF2D(size(layer_from->getDelta()(0)));
			prod(b).col(ti_to) = gg.t();
		}
	}

	if (ti_from == ti_to) {
		layer_from->incrDelta(prod, ti_from);   // spatial
		#ifdef DEBUG
		layer_from->deltas.push_back(prod);
		#endif
	} else {
		if (ti_to >= 0) {  // temporal
			layer_from->incrDelta(prod, ti_to);  // I do not like this conditional, temporal
			#ifdef DEBUG
			layer_from->deltas.push_back(prod);
			#endif
		}
	}
}
//----------------------------------------------------------------------
void Connection::dLdaMulGrad(int t)
{
	Layer* layer_from = from;
	Layer* layer_to   = to;
	int nb_batch = layer_from->getNbBatch();
	int seq_len = layer_from->getSeqLen();

	const VF2D_F& old_deriv = layer_to->getDelta();
	const VF2D_F& out = layer_from->getOutputs();
	Activation& activation = layer_to->getActivation();
	WEIGHT delta = VF2D(size(weight));

	if (activation.getDerivType() == "decoupled") {
		const VF2D_F& grad      = layer_to->getGradient();

		for (int b=0; b < nb_batch; b++) {
			const VF2D& out_t = out(b).t();
			if (!temporal) {
				delta = (old_deriv[b].col(t) % grad[b].col(t)) * out_t.row(t);
				//this->printSummary("spatial connection");
				//printf("dLdaMulGrad, t= %f,"); delta.print("delta");
			} else {
				//printf("TEMPORAL LINK\n");
				if (t+1 == seq_len) continue;    // ONLY FOR seq_len == 2
				delta = (old_deriv[b].col(t+1) % grad[b].col(t+1)) * out_t.row(t);

				//this->printSummary("temporal connection");
				//printf("dLdaMulGrad, t= %f,"); delta.print("delta");
			}
			incrDelta(delta);
			#ifdef DEBUG
			deltas.push_back(delta);
			#endif
		}
	} else { // "coupled derivatives"
		for (int b=0; b < nb_batch; b++) {
			const VF2D& out_t = out(b).t();

			if (!temporal) {
				const VF1D& x = layer_to->getInputs()(b).col(t);
				const VF1D& y = layer_to->getOutputs()(b).col(t);
				const VF2D grad = activation.jacobian(x, y); // not stored
	
				const VF2D& gg = out(b).col(t) * (old_deriv[b].col(t).t() * grad);
				// Must generalize for when times are not separated by 1, TODO (Need different arguments)
           		delta = gg.t();
			} else {
				if (t+1 == seq_len) continue;  // ONLY FOR seq_len == 2
				const VF1D& x =  layer_to->getInputs()(b).col(t+1);
				const VF1D& y = layer_to->getOutputs()(b).col(t+1);
				const VF2D grad = activation.jacobian(x, y); // not stored
				const VF2D& gg = out(b).col(t) * (old_deriv[b].col(t+1).t() * grad);
           		delta = gg.t();
			}

           	incrDelta(delta);
			#ifdef DEBUG
			deltas.push_back(delta);
			#endif
		}
	}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
