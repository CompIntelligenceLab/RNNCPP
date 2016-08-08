#include <stdio.h>
#include <math.h>
#include <string>
#include "model.h"
#include "activations.h"
#include "optimizer.h"
#include "layers.h"
#include "dense_layer.h"
#include "lstm_layer.h"
#include "gmm_layer.h"

int main() {
	Model* m  = new Model(1);
	Layer* dense = new DenseLayer(1, "dense");
	m->add(dense);
	Sigmoid* sig = new Sigmoid();

	printf("before init\n");
	m->initializeWeights();
	printf("after init\n");

	Optimizer* opt = new RMSProp("myrmsprop");
	opt->print();
	exit(1);

	m->print();
	printf("-----------------\n");

	Model n = *m;
	n.setName("model n");
	n.print();
	exit(0);
}
