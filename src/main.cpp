#include <stdio.h>
#include "model.h"
#include "activations.h"
#include "layers.h"
#include "dense_layer.h"
#include "lstm_layer.h"
#include "gmm_layer.h"

int main() {
	Model* m  = new Model();
	Layer* l1 = new DenseLayer();
	Layer* l2 = new LSTMLayer();
	Layer* l3 = new GMMLayer();
	m->add(l1);
	m->add(l2);
	m->add(l3);

	Tanh* a = new Tanh();
}
