#include <stdio.h>
#include <math.h>
#include "activations.h"

int main() {
	Tanh* ttanh = new Tanh();
	Sigmoid* ssigmoid = new Sigmoid();

	VF c(10);
	for (int i=0; i < 10; i++) {
		c[i] = i*.1;
	}

	VF tt = (*ttanh)(c);
	for (int i=0; i < tt.size(); i++) {
		printf("%f, %f\n", tt[i], tanh(c[i]));
	}

	tt = (*ssigmoid)(c);
	for (int i=0; i < 10; i++) {
		c[i] = 1. / (1. + exp(-i*.1));
	}

	printf("\n");
	for (int i=0; i < tt.size(); i++) {
		printf("%f, %f\n", tt[i], c[i]);
	}
}
