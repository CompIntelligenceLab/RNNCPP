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
	Model* m  = new Model();
	Layer* dense = new DenseLayer("dense");
	//Layer* l2 = new LSTMLayer("lstm");
	//Layer* l3 = new GMMLayer("gmm");
	m->add(dense);
	//m->add(l2);
	//m->add(l3);
	Sigmoid* sig = new Sigmoid();

	Optimizer* opt = new RMSProp("myrmsprop");
	opt->print();

	m->print();
}
