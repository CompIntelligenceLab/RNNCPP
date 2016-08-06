#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "layers.h"
#include "dense_layer.h"
#include "activations.h"


int main()
{
	Model* m  = new Model("feedforward");
	Layer* l1 = new DenseLayer("dense");
	m->add(l1);
	m->print("data on model");

	//Optimizer* opt = new RMSProp("myrmsprop");
	//opt->print();

	exit(0);
}
