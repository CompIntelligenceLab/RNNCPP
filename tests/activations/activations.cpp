#include <stdio.h>
#include <math.h>
#include "print_utils.h"
#include "activations.h"

int main() 
{
	printf("\n\n\n");
	printf("=============== BEGIN Activations =======================\n");
	Tanh* ttanh = new Tanh();
	Sigmoid* ssigmoid = new Sigmoid();
	bool success = true;

	VF2D_F c(2), tt(2), ttexact(2);        // number of fields
	VF1D norms(c.size());
	for (int f=0; f < c.size(); f++) {
		c[f] = VF2D(3,4);   // single field
		for (int i=0; i < c[f].size(); i++) {
			c[f](i) = i*.1;
		}
	}

	tt = (*ttanh)(c);   // ==> operator= not accepting a field as input
	for (int f=0; f < c.size(); f++) {
		ttexact[f] = VF2D(3,4);   // single field
		for (int i=0; i < c[f].size(); i++) {
			ttexact[f][i] = tanh(c[f][i]);
		}
		norms[f] = arma::norm(ttexact[f] - tt[f]);
		if (norms[f] > 1.e-5) success = false;
		printf("norms(Tanh::tanh, tanh) = %f\n", norms[f]);
	}

	U::print(tt, "Tanh::operator(VF2D_F) vs tanh(float x) function"); 

//------------------------------------------
	printf("\n\n---------- END TEST TANH ----------------\n");
	U::print(tt, "Sigmoid::operator(VF2D_F) vs sigmoid(float x) function"); 

	tt = (*ssigmoid)(c);
	for (int f=0; f < tt.size(); f++) {
		for (int i=0; i < tt[f].size(); i++) {
			ttexact[f][i] = 1. / (1. + exp(-i*.1));
		}
		norms[f] = arma::norm(ttexact[f] - tt[f]);
		if (norms[f] > 1.e-5) success = false;
		printf("norms(Sigmoid::sigmoid, sigmoid)= %f\n", norms[f]);
	}

	printf("\n\n---------- END TEST SIGMOID ----------------\n");

    if (success) {
  	  printf("\n\n--- Activations Test Successful! ---\n\n");
    } else {
  	  printf("\n\n--- Activations Test Failed! ---\n\n");
    }

//-----------------------------------------------------
	printf("\n\n: testing of activation parameter in DecayDE\n");
	DecayDE* decay = new DecayDE();
    {
		VF2D_F c(2), tt(2), ttexact(2);        // number of fields
		for (int f=0; f < c.size(); f++) {
			c[f] = VF2D(3,4);   // single field
			for (int i=0; i < c[f].size(); i++) {
				c[f](i) = i*.1;
			}
		}

		printf("before setParam\n");
		decay->setNbParams(1);
		decay->setParam(0, .15);
		printf("after setParam\n");
		VF2D_F val = (*decay)(c);
		printf("after val\n");
		c.print("c");
		val.print("decay(c)");

		VF2D_F deriv = decay->derivative(c);
		deriv.print("deriv");
	
  		return 0;
	}
}
