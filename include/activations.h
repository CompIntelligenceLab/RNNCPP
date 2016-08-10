#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__

#include <vector>
//#include <math.h>
#include <Eigen/Core>
#include "typedefs.h"


class Activation
{
protected:
	std::string name;
	static int counter;

public:
	Activation(std::string name="activation");
	virtual ~Activation();
	Activation(const Activation&); 
	const Activation& operator=(const Activation&); 
	/** Derivative f'(x) of activation function f(x) */
	/** x has dimensionality equal to the previous layer size */
	/** the return value has the dimensionality of the new layer size */
	virtual VF2D_F derivative(const VF2D_F& x) = 0; // derivative of activation function evaluated at x
	virtual VF2D_F operator()(const VF2D_F& x) = 0;
	virtual void print(std::string name= "");
};

//----------------------------------------------------------------------
class Tanh : public Activation
{
public:
	Tanh(std::string name="tanh") : Activation(name) {;}
	~Tanh();
    Tanh(const Tanh&);
    const Tanh& operator=(const Tanh&);
 
	VF2D_F operator()(const VF2D_F& x) {
#ifdef ARMADILLO
		VF2D_F y(x.n_rows);
		for (int i=0; i < x.n_rows; i++) {
			y[i] = tanh(x[i]);
		}
		return y;
#else
		AF ex = x;
		ex = (2.*ex).exp(); //exp(x);
		return (ex-1.) / (ex + 1.);
#endif
	}

	VF2D_F derivative(const VF2D_F& x)
	{
#ifdef ARMADILLO
		VF2D_F y(x.n_rows);
		for (int i=0; i < x.n_rows; i++) {
			y[i] = 1.-x[i]*x[i];
		}
		return y;
#else
		VF s = this->operator()(x);
		return (1.-s*s);
#endif
	}
};


//----------------------------------------------------------------------
class Sigmoid : public Activation
{
public:
	Sigmoid(std::string name="sigmoid") : Activation(name) {;}
	~Sigmoid();
    Sigmoid(const Sigmoid&);
    const Sigmoid& operator=(const Sigmoid&);

	VF2D_F operator()(const VF2D_F& x) {
#ifdef ARMADILLO
		VF2D_F y(x.n_rows);
		for (int i=0; i < x.n_rows; i++) {
			y[i] = 1. / (1. + exp(-x[i]));
		}
		return y;
#else
		AF ex = x;
		return 1. / (1. + (-ex).exp());
#endif
	}

	//f = 1 / (1 + exp(-x)) = 1/D
	//f' = -1/D^2 * (-exp(-x)-1 + 1) = -1/D^2 * (-D + 1) = 1/D - 1/D^2 = f (1-f)
	VF2D_F derivative(const VF2D_F& x) 
	{
#ifdef ARMADILLO
		VF2D_F y(x.n_rows);
		for (int i=0; i < x.n_rows; i++) {
			y[i] = x[i]*(1.-x[i]);
		}
		return y;
#else
		AF s = this->operator()(x);
		return s*(1-s);
#endif
	}
};
//----------------------------------------------------------------------

#endif
