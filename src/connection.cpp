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
	type = "all-all";
	frozen = false;

	// For Adam method  (memory waste?)
	mom = WEIGHT(out_dim, in_dim);
	vel = WEIGHT(out_dim, in_dim);
	mom.zeros();
	vel.zeros();
	adam_count = 0;


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
     temporal(w.temporal), clock(w.clock), to(w.to), from(w.from), frozen(w.frozen), type(w.type), t_from(w.t_from),
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
		frozen = w.frozen;
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
	if (msg != "") printf("%s\n", msg.c_str());
	printf("   in: %d, out: %d\n", in_dim, out_dim);
	printf("   print_verbose= %d\n", print_verbose);
	printf("   frozen= %d\n", frozen);
	printf("   temporal= %d\n", temporal);

	if (print_verbose == false) return;
}

void Connection::printSummary(std::string msg) 
{
	//printf("enter Connection::printSummary\n");
	//printf("Connection::temporal= %d\n", temporal);
	std::string type = (temporal) ? "temporal" : "spatial";

	//printf("from= %ld\n", from);
	//printf("to=   %ld\n",   to);
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
	// Routine does not appear to be used. So not complicit in memory leaks.

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
			weight = arma::randu<WEIGHT>(arma::size(weight)); //Uniform N(0,1)
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
	} else if (initialize_type == "xavier-char-rnn") {
	// initialization identical to char-rnn.py code by Karpathy
		//weight = arma::randn<WEIGHT>(arma::size(weight)); //Gaussian N(0,1)
		weight = arma::randn<WEIGHT>(arma::size(weight)); //Gaussian N(0,1)
		// I want the standard deviation to be 1/n
		weight = weight * init_weight_rms;
	} else if (initialize_type == "xavier_iden") {   // initialize recurrent weights to identity matrix
		weight = arma::randn<WEIGHT>(arma::size(weight)); //Gaussian N(0,1)
		printf("inside Connection::initialize()\n");
		weight.print("weight");

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
	} else if (initialize_type == "unity") {
		weight.ones();
	} else if (initialize_type == "identity") {
		weight.eye();
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
void Connection::dLossDOutput(int ti_from, int ti_to, bool prt)
{
	// Compute derivative of Loss wrt outputs
	// If time delay is 1,ti_to == -1 when ti_from == 0
	if (prt) {
		printf("\n-----------------\n");
		printf("********* ENTER Connection::dLossDOutput ****** ti_from, ti_to= =%d, %d\n", ti_from, ti_to);
		this->printSummary();
	}

	if (ti_to < 0) return;  

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
		const VF2D_F& grad 		= layer_to->getGradient();
		for (int b=0; b < nb_batch; b++) {
			prod(b) = VF2D(size(layer_from->getDelta()(0)));
		}
		const WEIGHT& wght_t = getWeightTranspose();
		// prod[-1] cannot be allowed
		//void U::rightTriad(VF2D_F& prod, const VF2D& a, const VF2D_F& b, const VF2D_F& c, int from, int to)
		//prod(p).col(to) = a * (b(p).col(from) % c(p).col(from));    // ERROR 
		//   prod = wght * (grad % old_deriv) 
		U::rightTriad(prod, wght_t, grad, old_deriv, ti_from, ti_to);  // dL/da
		if (prt) {
			printf("dLossDOutput, decoupled\n");
			layer_from->printSummary("from");
			layer_to->printSummary("to");
			old_deriv(0).raw_print(arma::cout, "old_deriv, dL/dOutput, layer_to");
			grad(0).raw_print(arma::cout, "layer_to->grad, activation grad");
			layer_from->getDelta()(0).raw_print(arma::cout, "layer_from->getDelta");
			layer_to->getDelta()(0).raw_print(arma::cout, "layer_to->getDelta");
		    wght_t.raw_print(arma::cout, "wght_t");
			printf("dLDOut: prod = wght_t * (grad %% old_deriv)\n");
			prod(0).raw_print(arma::cout, "prod, rightTriad");
		}
	} else { // "coupled"
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
		if (prt) { 
			printf("spatial link: d(L)/d(output)\n");
			layer_from->getDelta()(0).raw_print(arma::cout, "... before getDelta (layer_from->delta)");
		}
		layer_from->incrDelta(prod, ti_from);   // spatial
		if (prt) {
			layer_from->getDelta()(0).raw_print(arma::cout, "... after getDelta (layer_from->delta)");
		}
		#ifdef DEBUG
		layer_from->deltas.push_back(prod);
		#endif
	} else {
		if (prt) {
			printf("temporal link: d(L)/d(output)\n");
			prod(0).raw_print(arma::cout, "... prod (layer_from->delta)");
		}
		if (ti_to >= 0) {  // temporal
			layer_from->incrDelta(prod, ti_to);  // I do not like this conditional, temporal
			#ifdef DEBUG
			layer_from->deltas.push_back(prod);
			#endif
		}
	}
	prod.reset();
	//printf("Connection::EXIT dLossDOutput ****** ti_from, ti_to= =%f, %f\n", ti_from, ti_to);
}
//----------------------------------------------------------------------
void Connection::dLossDWeight(int t, bool prnt=false)
{
	Layer* layer_from = from;
	Layer* layer_to   = to;
	int nb_batch = layer_from->getNbBatch();
	int seq_len = layer_from->getSeqLen();

	const VF2D_F& old_deriv = layer_to->getDelta();
	const VF2D_F& out = layer_from->getOutputs();
	Activation& activation = layer_to->getActivation();
	const VF2D_F& previous_state = layer_from->getPreviousState();

	if (prnt) {
		printf("\n------------------------\n");
		printf("****** ENTER dLossDWeight *******, t= %d\n", t);
		this->printSummary("");
		//layer_from->printSummary("layer_from, ");
		previous_state(0).raw_print(arma::cout, "previous_state");
	}
	WEIGHT delta = VF2D(size(weight));
	//this->printSummary("dLossDWeight");

	if (activation.getDerivType() == "decoupled") {
		const VF2D_F& grad      = layer_to->getGradient();

		for (int b=0; b < nb_batch; b++) {
			const VF2D& out_t = out(b).t();
			if (!temporal) { // ERROR IN THIS PART OF THE CODE
				delta = (old_deriv[b].col(t) % grad[b].col(t)) * out_t.row(t);

				if (prnt) {
				    printf("GE: spatial\n");
					//delta.raw_print(arma::cout, "delta, dLossDWeight");
					old_deriv[b].raw_print(arma::cout, "old_deriv, layer_to->getDelta,  dLossDWeight");
					grad[b].raw_print(arma::cout, "layer_to->getGradient, dLossDWeight");
					out_t.raw_print(arma::cout, "layer_from->getOutputs, dLossDWeight");
					printf("delta = (old_deriv[b].col(t) %% grad[b].col(t)) * out_t.row(t);\n");
				}
			} else {
				//printf("TEMPORAL LINK\n");
				//printf("t+1= %d\n", t+1);
				// Added comment 12/24/16 (as a test only) (I get exception)
				// seq_len == 1: take previous state into account
				// MIGHT OR MIGHT NOT WORK, 12/24/16
				// Does not work. Screws up iteration one compared to Karpathy. 

				// This is not logical. should this be a test on the first 
				// time step instead? Will fix later

				if (t == 0) {
					// ERROR when layer_size = 2
					delta = (old_deriv[b].col(t) % grad[b].col(t)) * previous_state[b].t();
					if (prnt)  {
						printf("GE: temporal, t=0\n");
						this->printSummary("temporal connection");
						previous_state[0].raw_print(arma::cout, "..previous_state");
						old_deriv[0].raw_print(arma::cout, "old_deriv");
						grad[0].raw_print(arma::cout, "grad");
					    printf("delta = (old_deriv[b].col(t) %% grad[b].col(t)) * previous_state[b].t();\n");
						delta.raw_print(arma::cout, "TEMPORAL delta, dL/dW");
					}
				//} else if (t+1 == seq_len) {
					//if (prnt) printf("GE: temporal, t+1 == seq_len\n");
					//continue;    
				} else {
					//delta = (old_deriv[b].col(t+1) % grad[b].col(t+1)) * out_t.row(t);  // orig
					delta = (old_deriv[b].col(t) % grad[b].col(t)) * out_t.row(t-1);
					if (prnt) {
						printf("GE: temporal, else\n");
						this->printSummary("temporal connection");
						previous_state[0].raw_print(arma::cout, "..previous_state");
						old_deriv[0].raw_print(arma::cout, "old_deriv");
						grad[0].raw_print(arma::cout, "grad");
						delta.raw_print(arma::cout, "TEMPORAL delta dL/dW");
					}
				}
			}
			incrDelta(delta);
			if (prnt) {
				delta.raw_print(arma::cout, "dLossDWeight, incr delta");
				getDelta().raw_print(arma::cout, "total connection delta");
			}
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
	//printf("**** EXIT dLossDWeight *******\n");
}
//----------------------------------------------------------------------
void Connection::resetDelta() 
{ 
	delta.zeros();
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
