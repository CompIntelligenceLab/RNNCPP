#include "layers.h"
#include "print_utils.h"
#include <stdio.h>
#include <iostream>
#include <assert.h>

using namespace std;

int Layer::counter = 0;

Layer::Layer(int layer_size, std::string name /* "layer" */)
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("Layer constructor (%s)\n", this->name.c_str());

	counter++;

	// Input dimension has no significance if the layer is connected to several upstream layers
	// How to access connection to previous layer if this is the first layer? 

	this->layer_size = layer_size;
	//output_dim   = layer_size;  // no such member
	input_dim   = -1; // no assignment yet. 
	nb_batch    =  1;   // HOW TO SET BATCH FOR LAYERS? 
	seq_len     =  1; 
	print_verbose   = true;
	clock = 0;
	recurrent_conn = 0;
	bias.set_size(layer_size);
	bias_delta.set_size(layer_size);

	initVars(nb_batch);

	// Default activation: tanh
	activation = new Tanh("tanh");
}

void Layer::initVars(int nb_batch)
{
	inputs.set_size(nb_batch);
	outputs.set_size(nb_batch);
	delta.set_size(nb_batch);
	gradient.set_size(nb_batch);
	//printf("nb_batch= %d\n", nb_batch); exit(0);

	for (int i=0; i < nb_batch; i++) {
		inputs[i]   = VF2D(layer_size, seq_len);
		outputs[i]  = VF2D(layer_size, seq_len);
		gradient[i] = VF2D(layer_size, seq_len);
		delta[i]    = VF2D(layer_size, seq_len);
	}
	bias.zeros();
	bias_delta.zeros();
	nb_hit = 0;

	reset();
}

Layer::~Layer()
{
	printf("Layer destructor (%s)\n", name.c_str());
	if (activation) delete activation;
	activation = 0;
}

Layer::Layer(const Layer& l) : layer_size(l.layer_size), input_dim(l.input_dim),
   print_verbose(l.print_verbose), seq_len(l.seq_len), nb_batch(l.nb_batch),
   inputs(l.inputs), outputs(l.outputs), clock(l.clock), delta(l.delta), 
	recurrent_conn(l.recurrent_conn)
{
	name    = l.name + 'c';
	printf("Layer copy constructor (%s)\n", name.c_str());

	//TODO
	// How does activation work with polymorphism ?
	//activation = new Activation(); // HOW TO DO?   (does not work)
	//*activation = *l.activation; // check name
	//print("Layer copy constructor, activation->name= ", activation->name);
}

const Layer& Layer::operator=(const Layer& l)
{

	// if no copying done, name does not change
	if (this != &l) {
		name = l.name + "=";
		layer_size = l.layer_size;
		input_dim = l.input_dim;
		print_verbose = l.print_verbose;
		inputs = l.inputs;
		outputs = l.outputs;
		delta = l.delta;
		inputs = l.inputs;
		outputs = l.outputs;
		seq_len = l.seq_len;
		nb_batch = l.nb_batch;
		clock = l.clock;
		recurrent_conn = l.recurrent_conn;

		//Weights* w1; // remove class Weights
		Activation *a1;

		// remove class Weights (Nathan)
		try {
			a1 = new Tanh();
		} catch (...) {
			delete a1;
			printf("throw\n");
			throw;
		}

		*activation = *a1; 

		printf("Layer::operator= (%s)\n", name.c_str());
	}

	return *this;
}

void Layer::print(std::string msg /* "" */)
{
	printf("  -- layer: %s ---\n", name.c_str());
    if (msg != "") printf("%s\n", msg.c_str());
	printf("layer size: %d\n", layer_size);
	printf("input_dim: %d\n", input_dim);
	printf("inputs size: %llu\n", inputs.size());
	printf("outputs size: %llu\n", outputs.size());
	printf("print_verbose: %d\n", print_verbose);

	if (print_verbose == false) return;
}

void Layer::printSummary(std::string msg) 
{
	printf("%sLayer (%s, %s), layer_size: %d\n", msg.c_str(), name.c_str(), type.c_str(), layer_size);
}

void Layer::printName(std::string msg /*""*/)
{
	cout << "-- " << msg << ", layer (" << name << ") --" << endl;
}

// Take the inputs (dimensionality based on previous layer), and generate the outputs
void Layer::execute()
{
	outputs = (*activation)(inputs);
}

void Layer::reset() // Must I reset recurrent connection? 
{
	for (int b=0; b < inputs.size(); b++) {
		inputs(b).zeros();
		outputs(b).zeros();
		delta(b).zeros();
		gradient(b).zeros();
		clock = 0;
	}
}

void Layer::resetBackprop()
{
	for (int b=0; b < delta.size(); b++) {
		delta(b).zeros();
	}
}
//----------------------------------------------------------------------
void Layer::incrOutputs(VF2D_F& x)
{
	for (int b=0; b < x.n_rows; b++) {
		outputs(b) += x[b];
	}
}
//----------------------------------------------------------------------
void Layer::incrOutputs(VF2D_F& x, int t)
{
	// I do not know why "this->" is necessary since I do not notice any ambiguity 

	for (int b=0; b < x.n_rows; b++) {
		this->outputs(b).col(t) += x[b].col(t);
	}
	/*   ?????
	/Users/erlebach/src/RNNCPP/src/layers.cpp:170:14: error: reference to non-static member function must be called
	outputs[b].col[t] += x[b].col(t);
	*/
}
//----------------------------------------------------------------------
void Layer::incrInputs(VF2D_F& x)
{
	//printf("incrInputs: x.n_rows= %d\n", x.n_rows);
	//printf("inputs.n_rows= %d\n", inputs.n_rows);
	// inputs has incorrect number of fields.

	//U::print(x, "incrInputs x");
	//U::print(inputs, "incrInputs inputs");

	for (int b=0; b < x.n_rows; b++) {
		inputs[b] += x[b];
	}
}
//----------------------------------------------------------------------
void Layer::incrInputs(VF2D_F& x, int t)
{
	for (int b=0; b < x.n_rows; b++) {
		inputs[b].col(t) += x[b].col(t);
	}
}
//----------------------------------------------------------------------
void Layer::resetInputs()
{
	for (int b=0; b < inputs.n_rows; b++) {
		inputs[b].zeros();
	}
}
//----------------------------------------------------------------------
void Layer::resetInputs(int t)
{
	for (int b=0; b < inputs.n_rows; b++) {
		inputs[b].col(t).zeros();
	}
}
//----------------------------------------------------------------------
void Layer::incrDelta(VF2D_F& x)
{
	//printf("delta.rows: %d\n", delta.n_rows);
	//U::print(delta, "delta");
	//U::print(x, "incrDelta");
	//printf("x.n_rows = %d\n", x.n_rows);
	//printf("deltax.n_rows = %d\n", delta.n_rows);

	if (delta[0].n_rows == 0) {
		for (int b=0; b < x.n_rows; b++) {
			delta[b] = x[b];
		}
		//printf("deltax.n_rows = %d\n", delta.n_rows);
	} else {
		for (int b=0; b < x.n_rows; b++) {
			delta[b] += x[b];
		}
	}
}
//----------------------------------------------------------------------
void Layer::incrDelta(VF2D_F& x, int t)
{
	if (t < 0) return;

	for (int b=0; b < x.n_rows; b++) {
		delta[b].col(t) += x[b].col(t);
	}
}
//----------------------------------------------------------------------
void Layer::incrBiasDelta(VF1D& x)
{
	bias_delta += x;
}
//----------------------------------------------------------------------
void Layer::computeGradient()
{
	// Error. Derivatives must be evaluated for the input argument!
	//gradient = activation->derivative(outputs);
	//gradient = activation->derivative(inputs);

	if (getActivation().getDerivType() == "decoupled") {
		gradient = activation->derivative(inputs);
	} else {
		;  // Do not compute the gradient. Compute the Jacobian on the fly
	}
}
//----------------------------------------------------------------------
void Layer::computeGradient(int t)
{
	// Error. Derivatives must be evaluated for the input argument!
	//gradient = activation->derivative(outputs);
	// Cannot do lazy assignment if using columns

	// More efficient would be: 
	// gradient[b].col(t) = activation->derivative(gradient, inputs, k); 
	//     to avoid a copy. 

	//for (int b=0; b < inputs.n_rows; b++) {
		//gradient[b].col(t) = activation->derivative(inputs[b].col(t));
	//}
	if (getActivation().getDerivType() == "decoupled") {
		for (int b=0; b < inputs.n_rows; b++) {
			gradient[b].col(t) = activation->derivative(inputs[b].col(t));
		}
	} else {
		; // Do not compute the gradient. Compute Jacobian when required
	}


	//printf("t= %d, ", t);
	//this->printSummary();
	//inputs.print("inputs");
	//gradient.print("gradient");
}
//----------------------------------------------------------------------
void Layer::forwardData(Connection* conn, VF2D_F& prod, int seq)
{
	// forward data to spatial connections

	const VF2D_F& from_outputs = getOutputs();
	const WEIGHT& wght = conn->getWeight();
	//U::matmul(prod, wght, from_outputs, 0);  // sequence element zero)
	U::matmul(prod, wght, from_outputs);

	//prod.print("prod, forwardData");

	// Data is not actually forwarded. It should be forwarded to the input 
	// of the following layer. 
}
//----------------------------------------------------------------------
bool Layer::areIncomingLayerConnectionsComplete()
{
	return (nb_hit == prev.size());
}
//----------------------------------------------------------------------
#if 1
// Perhaps break this up into processing by Connection then by output layer
void Layer::processOutputDataFromPreviousLayer(Connection* conn, VF2D_F& prod)
{
	//printf("enter Layer::processOutputDataFromPreviousLayern");
	++nb_hit;

	const VF2D_F& from_outputs = conn->from->getOutputs();
	const WEIGHT& wght = conn->getWeight();  // what if connection operates differently
	VF2D_F& to_inputs = layer_inputs[conn->which_lc];

	#if 0
	conn->printSummary("conn");
	printf("which_lc= %d\n", conn->which_lc);
	printf("layer_inputs.size= %d\n",layer_inputs.size());
	layer_inputs[conn->which_lc].print("layer_inputs, which_lc");  // ===> zero
	wght.print("wght");
	from_outputs.print("from_outputs");
	#endif


	// TEMPORARY LOOP
	#if 0
	for (int i=0; i < layer_inputs.size(); i++) {
		//layer_inputs[i] = VF2D_F(nb_batch);
		int input_dim = getLayerSize();
		int seq_len   = getSeqLen();
		printf("input_dim, seq_len= %d, %d\n", input_dim, seq_len);
		for (int b=0; b < nb_batch; b++) {
			//layer_inputs[i](b) = VF2D(input_dim, seq_len);
			U::print(layer_inputs[i](b), "U:: layer_inputs (tmp)");
		}
	}
	#endif

	//U::matmul(to_inputs, wght, from_outputs);  // w * x
	U::matmul(prod, wght, from_outputs);  // w * x
	to_inputs = prod;


	//prod.print("enter processData, prod");

	// Where are the various inputs added up? So derivatives will work if layer_size=1, but not otherwise. 

	// completeness only happens once per layer and per input value into the predicition module
	if (areIncomingLayerConnectionsComplete()) {
		 // sum up all the inputs + the temporal input if there is one
		 resetInputs();
		 for (int i=0; i < layer_inputs.size(); i++) {
		 	incrInputs(layer_inputs[i]);
			//layer_inputs[i].print("layer_inputs incr");
			//printf("incrInputs, input: %d\n", i);
		 }
		 // add the self-looping if there. 
		 incrInputs(loop_input);
		 //loop_input.print("add loop_input, ");

		 prod = getActivation()(getInputs());
		 //prod.print("processData, output, ");
		 setOutputs(prod);
	}
}
#endif
//----------------------------------------------------------------------
void Layer::processOutputDataFromPreviousLayer(Connection* conn, VF2D_F& prod, int t)
{
	//printf("enter Layer::processOutputDataFromPreviousLayer, t= %d\n", t);
	++nb_hit;

	const VF2D_F& from_outputs = conn->from->getOutputs();
	const WEIGHT& wght = conn->getWeight();  // what if connection operates differently
	VF2D_F& to_inputs = layer_inputs[conn->which_lc];

	//layer_inputs[0].print("layer_input[0]");
	this->printSummary();

		#if 0
		conn->printSummary("conn");
		printf("which_lc= %d\n", conn->which_lc);
		printf("layer_inputs.size= %d\n",layer_inputs.size());
		//layer_inputs[conn->which_lc].print("layer_inputs, which_lc");  // ===> zero
		wght.print("wght");
		U::print(wght, "wght");
		U::print(from_outputs, "from_outputs");
		U::print(prod, "prod");
		conn->from->printSummary("conn->from");
		from_outputs.print("from_outputs");
		#endif

	//U::matmul(prod, wght, from_outputs);  // w * x
	//printf("seq_i= %d\n", seq_i);



	U::matmul(to_inputs, wght, from_outputs, t, t);  // w * x



	//U::print(to_inputs, "to_inputs");
	//to_inputs.print("to_inputs");
	//to_inputs = prod;
	//exit(0);

	//prod.print("enter processData, prod");

	// completeness only happens once per layer and per input value into the predicition module
	if (areIncomingLayerConnectionsComplete()) {
		 // sum up all the inputs + the temporal input if there is one
		 resetInputs(t);   // incorrect answer for a10, correct for a00
		 //resetInputs();  // correct answer (STRANGE)
		//layer_inputs[conn->which_lc].print("** layer_inputs, which_lc");  // ===> zero
		//exit(0);
		 for (int i=0; i < layer_inputs.size(); i++) {
		 	incrInputs(layer_inputs[i], t);
		 }
		 // add the self-looping if there. 
		 //loop_input.print("loop input");  // UNINITIALIZED
		 incrInputs(loop_input); 
		 // Add layer biases. must loop over batch and over sequence size. 
		 //inputs.print("before, inputs, ");
		 //inputs.print("before bias, layer inputs, ");
		 addBiasToInput(t);
		 //inputs.print("after bias, layer inputs, ");
		 //bias.print("bias");
		 //inputs.print("after, inputs, ");
		 //printf("gordon\n");
		 //exit(0);

		 prod = getActivation()(getInputs());
		 setOutputs(prod);
		 //this->printSummary();
		 //getOutputs().print("outputs");
	}
	//exit(0);
}
//----------------------------------------------------------------------
void Layer::processData(Connection* conn, VF2D_F& prod)
{
		++nb_hit;

		VF2D_F& to_inputs = layer_inputs[conn->which_lc];
		to_inputs = prod;
		//prod.print("enter processData, prod");

		// Where are the various inputs added up? So derivatives will work if layer_size=1, but not otherwise. 

		if (areIncomingLayerConnectionsComplete()) {
			 prod = getActivation()(prod);
			 //prod.print("processData, set output");
			 setOutputs(prod);
		}
}
//----------------------------------------------------------------------
void Layer::forwardLoops()
{ }
void Layer::forwardLoops(int seq)
{ }
//----------------------------------------------------------------------
void Layer::resetState()
{
	U::zeros(inputs);
	U::zeros(outputs);
	U::zeros(delta);

	for (int i=0; i < layer_inputs.size(); i++) {
		U::zeros(layer_inputs[i]);
		U::zeros(layer_deltas[i]);
	}
}
//----------------------------------------------------------------------
void Layer::resetDelta()
{
	U::zeros(delta);
	//for (int b=0; b < delta.n_rows; b++) {
		//delta[b].zeros();
	//}
}
//----------------------------------------------------------------------
void Layer::addBiasToInput(int t)
{
	for (int b=0; b < nb_batch; b++) {
		inputs(b).col(t) += bias;
	}
}
//----------------------------------------------------------------------
#if 1
void Layer::gradMulDLda(VF2D_F& prod, const WEIGHT& wght_t, int t_from, int t_to)
{
	const VF2D_F& old_deriv = this->getDelta();

	//printf("act type: %s\n", getActivation().getDerivType().c_str());
	Activation& activation = getActivation();

	if (getActivation().getDerivType() == "decoupled") {   // ** called
		printf("gradMulDLda, decoupled\n");
		const VF2D_F& grad 		= this->getGradient();
		U::rightTriad(prod, wght_t, grad, old_deriv, t_from, t_to);
	} else { // "coupled"
		printf("gradMulDLda, coupled\n");
		for (int b=0; b < nb_batch; b++) {
			const VF1D& x =  inputs(b).col(t_from);
			const VF1D& y = outputs(b).col(t_from);
			const VF2D grad = activation.jacobian(x, y); // not stored (3,3)
			printf("prod(b).col(t_to) = wght_t * (grad * old_deriv[b].col(t_from);)\n");
			U::print(prod, "prod"); // (3,2)
			U::print(wght_t, "wght_t");  // (4,3)
			U::print(grad, "grad");  // (3,3)
			U::print(old_deriv[b], "old_deriv[b]"); //   3,2
			//VF2D gg = grad * old_deriv[b].col(t_from); // orig
			VF2D gg = old_deriv[b].col(t_from).t(); // orig
			//VF2D gg = old_deriv[b].col(t_from) * grad;
			//VF2D hh = grad.t() * wght_t.t();
			VF2D hh = gg * grad;
			VF2D ii = hh * wght_t;
			//VF2D ii = old_deriv[b].col(t_from) * wght_t();
			//U::print(gg, "gg");
			//U::print(hh, "hh");
			U::print(ii, "ii");
			//exit(0);
			U::print(prod(b), "prod(b)");
			prod(b).col(t_to) = ii.t();
			//prod(b).col(t_to) = wght_t * (grad * old_deriv[b].col(t_from));
		}
	}
	//printf("XXX exit\n"); exit(0);
}
#endif
//----------------------------------------------------------------------
void Layer::gradMulDLda(VF2D_F& prod, const Connection& conn, int t_from, int t_to)
{
	assert(this == conn.to);

	const VF2D_F& old_deriv = this->getDelta();  // 3
	const WEIGHT& wght   = conn.getWeight(); // invokes copy constructor, or what?  3 x 4
	Layer* layer_from = conn.from; // 4

	Activation& activation = getActivation();

	if (getActivation().getDerivType() == "decoupled") {   
		printf("gradMulDLda, decoupled\n");
		const VF2D_F& grad 		= this->getGradient();
		for (int b=0; b < prod.n_rows; b++) {
			prod(b) = VF2D(size(layer_from->getDelta()(0)));
		}
		const WEIGHT& wght_t = conn.getWeightTranspose();
		U::rightTriad(prod, wght_t, grad, old_deriv, t_from, t_to); 
	} else { // "coupled"
		printf("gradMulDLda, coupled\n");
		for (int b=0; b < nb_batch; b++) {
			const VF1D& x   =  inputs(b).col(t_from);
			const VF1D& y   = outputs(b).col(t_from);
			const VF2D grad = activation.jacobian(x, y); // not stored (3,3)
			// parentheses required to ensure that left hand multiplication occurs first
			// since old_deriv is a vector and grad/wght are matrices
			const VF2D& gg = (old_deriv[b].col(t_from).t() * grad) * wght; 
			prod(b) = VF2D(size(layer_from->getDelta()(0)));
			prod(b).col(t_to) = gg.t();
		}
	}
}
//----------------------------------------------------------------------
void Layer::dLdaMulGrad(Connection* con, const VF2D_F& out_, int t)
{
	Layer* layer_from = con->from;
	Layer* layer_to   = con->to;
	const VF2D_F& old_deriv = getDelta();
	const VF2D_F& out = layer_from->getOutputs();
	//printf("act type: %s\n", getActivation().getDerivType().c_str());
	Activation& activation = getActivation();
	WEIGHT delta = VF2D(size(con->getWeight()));

	if (getActivation().getDerivType() == "decoupled") {
		printf("dLdaMulGrad, decoupled\n");
		const VF2D_F& grad      = getGradient();

		for (int b=0; b < nb_batch; b++) {
			const VF2D& out_t = out(b).t();
			delta = (old_deriv[b].col(t) % grad[b].col(t)) * out_t.row(t);
			con->incrDelta(delta);
		}
	} else { // "coupled derivatives"


		printf("dLdaMulGrad, coupled\n");
		printf("------------------\n");
		for (int b=0; b < nb_batch; b++) {
			const VF1D& x =  inputs(b).col(t);
			const VF1D& y = outputs(b).col(t);
			const VF2D grad = activation.jacobian(x, y); // not stored
			const VF2D& out_t = out(b).t();
			const VF2D& hh = (old_deriv[b].col(t).t() * grad);
			U::print(hh, "hh");
			U::print(out(b), "out(b)");
			const VF2D& gg = out(b).col(t) * (old_deriv[b].col(t).t() * grad);
           	// Must generalize for when times are not separated by 1, TODO (Need different arguments)
			U::print(old_deriv[b], "old_deriv[b]");
			printf("delta = (old_deriv[b].col(t) * grad) * out_t.row(t);\n");
			printf("t= %d\n", t);
			old_deriv[b].col(t).print("old_deriv[b].col(t)");
			grad.print("grad");
			out_t.row(t).print("out_r.row(t)");
           	//delta = (old_deriv[b].col(t) * grad) * out_t.row(t); //out(b).t();    // ERROR
           	delta = gg.t();
			delta.print("delta");
           	con->incrDelta(delta);
		}
		con->getDelta().print("con->getDelta()");
		//exit(0);
	}
	//printf("Layer::dLdaMulGrad\n"); exit(0);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
