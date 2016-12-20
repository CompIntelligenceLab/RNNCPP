#include "activations.h"
#include <stdio.h>

int Activation::counter = 0;

Activation::Activation(std::string name /* "activation" */) 
{
	char cname[80];
	deriv_type = "decoupled";  // default

	dt = .10;   // Discrete time step
	//printf("RESET dt to a small value!!!\n");

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	//printf("Activation constructor (%s)\n", this->name.c_str());
	counter++;

	// allow for a maximum of 10 parameters
	setNbParams(0);
}
//----------------------------------------------------------------------
Activation::~Activation() 
{
	//printf("Activation destructor (%s)\n", name.c_str());
	;
}
//----------------------------------------------------------------------
void Activation::print(std::string msg /* "" */)
{
	printf("Activation: name= %s\n", this->name.c_str());
	if (msg != "") printf("%s\n", msg.c_str());
}
//----------------------------------------------------------------------
Activation::Activation(const Activation& a)
{
	char cname[80];
	printf("activation copy constructor\n");

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	counter++;
}

const Activation& Activation::operator=(const Activation& a)
{
	printf("activation operator=\n");
	if (this != &a) {
		name = a.name;
	}
	return *this;
}
//----------------------------------------------------------------------
void Activation::setParam(int which, REAL value) 
{ 
	if (which < params.size()) {
		params[which] = value; 
		//printf("set parameter\n");
	} else {
		printf("params argument out of range, which= %d, value= %f\n", which, value);
		//exit(1);
	}
}
//----------------------------------------------------------------------
#if 0
void Activation::resetDelta()
{
}
#endif
//----------------------------------------------------------------------


Tanh::~Tanh() 
{
	printf("Tanh constructor (%s)\n", this->name.c_str());
}

Tanh::Tanh(const Tanh& t) : Activation(t)
{ }

const Tanh& Tanh::operator=(const Tanh& t)
{
	if (this != &t) {
		name = t.name + '=';
	}
	return *this;
}


Sigmoid::~Sigmoid()
{
}
 
Sigmoid::Sigmoid(const Sigmoid& s) : Activation(s)
{
	printf("Sigmoid constructor (%s)\n", this->name.c_str());
}

const Sigmoid& Sigmoid::operator=(const Sigmoid& s)
{
	if (this != &s) {
		name = s.name + '=';
	}
	return *this;
}

//----------------------------------------------------------------------
Identity::~Identity()
{
	printf("identity destructor\n");
}

//Identity::Identity(Identity&& w) = 0;
 
Identity::Identity(const Identity& s) : Activation(s)
{
	printf("Identity copy constructor (%s)\n", this->name.c_str());
}

const Identity& Identity::operator=(const Identity& s)
{
	printf("identity operator=\n");
	if (this != &s) {
		name = s.name + '=';
	}
	return *this;
}

//----------------------------------------------------------------------

VF2D_F Softmax::operator()(const VF2D_F& x) 
{
	VF2D_F y(x);
	// softmax over the dimension index of VF2D (first index)
	for (int b=0; b < x.n_rows; b++) {
		float mx = arma::max(arma::max(x[b]));
	    for (int s=0; s < x[b].n_cols; s++) {
		    y(b).col(s) = arma::exp(y(b).col(s)-mx);
			// trick to avoid overflows
			float ssum = 1. / arma::sum(y[b].col(s)); // = arma::exp(y[b]);
			y[b].col(s) = y[b].col(s) * ssum;  // % is elementwise multiplication (arma)
		}
		return y;
	}
}

//----------------------------------------------------------------------
//f = 1 / (1 + exp(-x)) = 1/D
//f' = -1/D^2 * (-exp(-x)-1 + 1) = -1/D^2 * (-D + 1) = 1/D - 1/D^2 = f (1-f)
VF2D_F Softmax::derivative(const VF2D_F& x)
{
	VF2D_F y(x.n_rows);
	for (int b=0; b < x.n_rows; b++) {
		y[b] = arma::zeros<VF2D>(arma::size(x[b]));
		y[b].elem(find(x[b] > 0.)).ones(); 
	}
	return y;
}

VF1D Softmax::derivative(const VF1D& x) 
{
	VF1D y(x.n_rows);
	for (int i=0; i < x.n_rows; i++) {
		// Use >= or > 
		y[i] = x[i] >= 0. ? 1.0 : 0.0;
	}
	printf("Activation.cpp::derivative of Softmax is not applicable. Use Jacobian\n");
	exit(1);
	return x;
}

// arguments are x and y=softmax(x) (already computed)
// The second argument is used for efficiency
VF2D Softmax::jacobian(const VF1D& x, const VF1D& y)
{
	VF2D jac(x.n_rows, x.n_rows);
	deriv_type = "coupled";  // never tested

	// jac(i,j) = dsoft[i]/dx[j]
	//printf("IN JACOBIAN\n");
	//x.print("x");
	//y.print("y");
	//printf("nr, nc= %d, %d\n", jac.n_rows, jac.n_cols);

	//VF1D soft = (*this)(x);

	for (int i=0; i < jac.n_rows; i++) {
	for (int j=0; j < jac.n_cols; j++) {
		if (i == j) {
			jac(i,j) = y[i] * (1. - y[j]);
		} else {
			jac(i,j) = -y[i] * y[j];
		}
		//printf("i,j= %d, %d, jac(i,j)= %f\n", i, j, jac(i,j));
	}}
	return jac;
}
//----------------------------------------------------------------------
ReLU::~ReLU()
{
}
//----------------------------------------------------------------------
DecayDE::DecayDE(std::string name /*="decayde"*/) : Activation(name)
{
	setNbParams(10);
	//for (int i=0; i < 10; i++) { printf("x params[%d]= %f\n", i, params[i]); }
	//exit(0);
	//printf("params.size= %d\n", params.size());
	//exit(0);
	// freeze all parameters
	for (int i=0; i < params.size(); i++) {
		frozen[i] = true;
	}
	params = arma::randn<VF1D>(arma::size(params)); 

	// unfreeze single parameter (the zeroth one)
	unfreezeParam(0);

	printf("DecayDE constructor (%s)\n", this->name.c_str());
}

DecayDE::~DecayDE()
{ }

const DecayDE& DecayDE::operator=(const DecayDE& t)
{
	if (this != &t) {
		name = t.name + '=';
	}
	return *this;
}
