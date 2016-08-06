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
	Layer* l1 = new DenseLayer("dense");
	Layer* l2 = new LSTMLayer("lstm");
	Layer* l3 = new GMMLayer("gmm");
	m->add(l1);
	m->add(l2);
	m->add(l3);

	Activation* ttanh = new Tanh("tanh");
	Sigmoid* ssigmoid = new Sigmoid("sigmoid");
	l2->setActivation(ssigmoid);

	Optimizer* opt = new RMSProp("myrmsprop");
	opt->print();

	m->print();
}
