#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__

#include <vector>
#include <math.h>
#include "typedefs.h"


class Activation
{
public:
	Activation();
	~Activation();
	virtual VF operator()(VF x) = 0;
};
//----------------------------------------------------------------------
class Tanh : public Activation
{
public:
	VF operator()(VF x) {
		VF y; //x.size);
		for (int i=0; i < x.size(); i++) {
			y[i] = tanh(x[i]);
		}
		return y;
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
