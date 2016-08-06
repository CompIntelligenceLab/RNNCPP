#include "layers.h"

Layer::Layer(int layer_size) {
	this->layer_size = layer_size;
	int batch_size = 1;  // default value
	int seq_len   = 1; // default value
	int dim   = 1; // default: scalar
}

Layer::~Layer() {
}

Layer::Layer(Layer&) {
}

