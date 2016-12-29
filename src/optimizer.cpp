#include "optimizer.h"
#include "model.h"
#include <stdio.h>

class Model;

int Optimizer::counter = 0;

Optimizer::Optimizer(std::string name /* "optimizer" */)
{
	char cname[80];

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("Optimizer constructor (%s)\n", this->name.c_str());

	this->learning_rate = 1.e-5;
}

Optimizer::~Optimizer()
{
	printf("Optimization destructor (%s)\n", name.c_str());
}

Optimizer::Optimizer(const Optimizer& o) : learning_rate(o.learning_rate)
{
	loss = o.loss;
	name = o.name + "c";
	printf("Optimizer copy constructor (%s)\n", name.c_str());
}

const Optimizer& Optimizer::operator=(const Optimizer& o) 
{
	if (this != &o) {
		learning_rate = o.learning_rate;
		name = o.name + "=";
		loss = o.loss;
		printf("Optimizer::operator= (%s)\n", name.c_str());
	}

	//printf("exit optimizer=\n");
	return *this;
}

void Optimizer::print(const std::string msg /*=std::string()*/)
{
	//printf("enter optimizer print\n"); // How can this print more than once? 
  //printf("msg = %s\n", msg.c_str());
	//printf("xxx\n");
	//printf("Optimizer: %s\n", name.c_str());
	//if (msg != "") printf("%s\n", msg.c_str());
	printf("learning_rate: %f\n", learning_rate);
	printf("exit optimizer print\n");
	//exit(0);
}

void Optimizer::update(Model* mo, VF2D& w, VF2D& m, VF2D& v, VF2D& dLdw, int& count) {;}

//----------------------------------------------------------------------

RMSProp::RMSProp(std::string name /* "RMSProp" */) : Optimizer(name) {;}
//----------------------------------------------------------------------

//Adam::Adam(REAL alpha /*=.001*/, REAL beta1 /*=.9*/, REAL beta2 /*=.999*/, REAL eps /*=1.e-8*/)
Adam::Adam(std::string name) : Optimizer(name) 
{
	this->alpha = 0.001;
	this->beta1 = 0.9;
	this->beta2 = 0.999;
	this->eps   = 1.e-8;
	//alphat = alpha;
}
//----------------------------------------------------------------------
Adam::~Adam()
{
}
//----------------------------------------------------------------------
Adam::Adam(const Adam&) 
{
}
//----------------------------------------------------------------------
void Adam::update(Model* mo, VF2D& w, VF2D& m, VF2D& v, VF2D& dLdw, int& count)
{
	printf("enter: norm(dLdw)= %f\n", arma::norm(dLdw));
	printf("norm(m)= %f\n", arma::norm(m));
	printf("norm(v)= %f\n", arma::norm(v));
	printf("norm(w)= %f\n", arma::norm(w));
	count++;
	REAL beta1t = pow(beta1, count);
	REAL beta2t = pow(beta2, count);
	//VF2D gt = dLdw;
	m = beta1 * m + (1.-beta1) * dLdw;
	v = beta2 * v + (1.-beta2) * arma::square(dLdw);
	REAL alphat = alpha * sqrt(1. - beta2t) / (1. - beta1t);
	w = w - alphat * m / (arma::sqrt(v) + eps);
	//dLdw.print("exit: dLdw");
	//m.print("m");
	//v.print("v");
	//w.print("w");
	//arma::sqrt(v).print("sqrt(v)");
	printf("beta1t, beta2t= %f, %f, count= %d, alphat= %f\n", beta1t, beta2t, count, alphat);

	// Why nan's? 
#if 0
t←t+1
gt ← ∇θft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
vt ← β2 · vt−1 + (1 − β2) · gt2 (Update biased second raw moment estimate)
m􏰨t ←mt/(1−β1t)(Computebias-correctedfirstmomentestimate)
v􏰨t ← vt /(1 − β2t ) (Compute bias-corrected second raw moment estimate) √
θt ←θt−1 −α·m􏰨t/( v􏰨t +ε)(Updateparameters)

αt=α·􏰜(1−β2t)/(1−β1t) and  θt←θt−1 −αt ·mt/(√vt +εˆ).
#endif
}
//----------------------------------------------------------------------
Adagrad::Adagrad(std::string name) : Optimizer(name) 
{
	this->alpha = 0.001;
	this->beta1 = 0.9;
	this->beta2 = 0.999;
	this->eps   = 1.e-8;
	//alphat = alpha;
}
//----------------------------------------------------------------------
Adagrad::~Adagrad()
{
}
//----------------------------------------------------------------------
Adagrad::Adagrad(const Adagrad&) 
{
}
//----------------------------------------------------------------------
void Adagrad::update(Model* mo, VF2D& w, VF2D& m, VF2D& v, VF2D& dLdw, int& count)
{
	count++;
	REAL lr = mo->getLearningRate();
	m += dLdw % dLdw;
	//printf("lr= %f\n", lr);
	//printf("enter: norm(dLdw)= %f\n", arma::norm(dLdw));
	//printf("norm(m)= %f\n", arma::norm(m));
	//printf("norm(w)= %f\n", arma::norm(w));
	VF2D dw = dLdw / arma::sqrt(m+1.e-8);
	//dw.print("dw");
	//printf("norm(dw)= %f\n", arma::norm(dw));
	w -= lr * dLdw / arma::sqrt(m + 1.e-8);
}
//----------------------------------------------------------------------
