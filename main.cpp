#include <stdio.h>
#include <math.h>
#include <string>
#include "model.h"
#include "activations.h"
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

	//Tanh* ttanh = new Tanh();
	//Sigmoid* ssigmoid = new Sigmoid();

	m->print();
}
