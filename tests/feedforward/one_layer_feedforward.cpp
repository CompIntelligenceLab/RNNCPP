#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "layers.h"
#include "dense_layer.h"
#include "activations.h"

// MUST REWRITE

int main()
{
#if 0
	Model* m  = new Model(1, "feedforward");
	Layer* l1 = new DenseLayer(5, "dense");
	m->add(l1);
	m->print("data on model");

	Optimizer* opt = new RMSProp("myrmsprop");
	opt->print();
  //m->predict(1.0); // Uncommenting this will break things!

  printf("\n\n--- Feed Forward Test Successful ---\n\n");
	exit(0);
#endif
}
