#include "layers.h"

Layer::Layer(int layer_size, std::string name)
{
	this->name = name;
	this->layer_size = layer_size;
	int batch_size = 1;  // default value
	int seq_len   = 1; // default value
	int dim   = 1; // default: scalar
	weights = 0;
	activation = 0;
}

Layer::~Layer()
{
	delete weights;
	delete activation;
}

Layer::Layer(Layer&)
{
}

void Layer::print() {
	printf("  -- layer: %s ---\n", name.c_str());
	printf("seq len: %d\n", seq_len);
	printf("batch_size: %d\n", batch_size);
	printf("dim: %d\n", dim);
	printf("layer size: %d\n", layer_size);
	printf("input_dim: ???\n");
	printf("inputs size: %ld\n", inputs.size());
	printf("outputs size: %ld\n", outputs.size());
	if (weights) {
		weights->print();
	}
	if (activation) {
		activation->print();
	}
}
