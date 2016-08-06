#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__

#include <vector>
#include <math.h>
#include <Eigen/Core>
#include "typedefs.h"


class Activation
{
public:
	Activation();
	~Activation();
	/** Gradient f'(x) of activation function f(x) */
	/** x has dimensionality equal to the previous layer size */
	/** the return value has the dimensionality of the new layer size */
	virtual VF gradient(VF x) = 0; // gradients of activation function evaluated at x
	virtual VF operator()(VF x) = 0;
};
//----------------------------------------------------------------------
class Tanh : public Activation
{
public:
	VF operator()(VF x) {
		AF ex = x;
		ex = 2.*ex.exp(); //exp(x);
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
	VF operator()(VF x) {
		VF y; //x.size);
		for (int i=0; i < x.size(); i++) {
			y[i] = 1. / (1.+exp(-x[i]));
		}
		return y;
	}
};
//----------------------------------------------------------------------

/*
class Sigmoid : public class Activation {
	float operator()(VF x);
};
*/

#endif
