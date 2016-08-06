#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__

#include <vector>
//#include <math.h>
#include <Eigen/Core>
#include "typedefs.h"


class Activation
{
private:
	std::string name;

public:
	Activation();
	virtual ~Activation();
	/** Gradient f'(x) of activation function f(x) */
	/** x has dimensionality equal to the previous layer size */
	/** the return value has the dimensionality of the new layer size */
	virtual VF gradient(VF x) = 0; // gradients of activation function evaluated at x
	virtual VF operator()(VF x) = 0;
	virtual void print();
};
//----------------------------------------------------------------------
class Tanh : public Activation
{
public:
	Tanh(std::string name="tanh");
	VF operator()(VF x) {
		AF ex = x;
		ex = (2.*ex).exp(); //exp(x);
		return (ex-1.) / (ex + 1.);
	}

	VF gradient(VF x)
	{
		AF s = this->operator()(x);
		return (1.-s*s);
	}
};
//----------------------------------------------------------------------
class Sigmoid : public Activation
{
public:
	Sigmoid(std::string name="sigmoid");
	VF operator()(VF x) {
		AF ex = x;
		return 1. / (1. + (-ex).exp());
	}

	//f = 1 / (1 + exp(-x)) = 1/D
	//f' = -1/D^2 * (-exp(-x)-1 + 1) = -1/D^2 * (-D + 1) = 1/D - 1/D^2 = f (1-f)
	VF gradient(VF x) 
	{
		AF s = this->operator()(x);
		return s*(1-s);
	}
};
//----------------------------------------------------------------------

/*
class Sigmoid : public class Activation {
	float operator()(VF x);
};
*/

#endif
