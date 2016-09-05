#include <stdio.h>
#include <math.h>
#include "print_utils.h"
#include "activations.h"

int main() {
	printf("\n\n\n");
	printf("=============== BEGIN Activations =======================\n");
	Tanh* ttanh = new Tanh();
	Sigmoid* ssigmoid = new Sigmoid();

	VF2D_F c(1);
	c[0] = VF2D(5,7);
	for (int i=0; i < c.size(); i++) {
		c[0](i) = i*.1;
	}

	VF2D_F tt(1);
	tt = (*ttanh)(c);   // ==> operator= not accepting a field as input
	U::print(tt, "tt"); 

	for (int i=0; i < tt[0].size(); i++) {
		printf("%f, %f\n", tt[0][i], tanh(c[0][i]));
	}

	tt = (*ssigmoid)(c);
	for (int i=0; i < tt[0].size(); i++) {
		c[0][i] = 1. / (1. + exp(-i*.1));
	}

	printf("\n");
	for (int i=0; i < tt[0].size(); i++) {
		printf("%f, %f\n", tt[0][i], c[0][i]);
	}

  printf("\n\n---Activations Test Successful---\n\n");
  return 0;
}
