
#include <stdio.h>
#include <math.h>
#include "activations.h"
#include "optimizer.h"
#include "connection.h"
#include "layers.h"
#include "dense_layer.h"

int main() 
{
	printf("\n\n\n");
	printf("=============== BEGIN copy_constructorssss  =======================\n");
	Optimizer* opt = new Optimizer("optGE");
	Optimizer opt1(*opt);
	Optimizer opt2 = *opt;

	opt->print("\n--> print opt\n");
	opt1.print("\n--> print opt1\n");
	opt2.print("\n--> print opt2\n");

	printf("=================================\n");
	Connection* weight = new Connection(1,1,"wghtGE");
	Connection w1(*weight);
	Connection w2 = *weight;

	weight->print("\n --> print wght\n");
	w1.print("\n --> print w1\n");
	w2.print("\n --> print w2\n");

	printf("=================================\n");
	int layer_size = 5;
	Layer* layer = new DenseLayer(layer_size, "layerGE");

	// Both these statements lead to to identical objects being deleted. Only one of these
	// statements leads to correct code. 
	DenseLayer layer3(layer_size, "layer3GE");
	DenseLayer layer2;
	DenseLayer layer4 = layer3;

  printf("\n\n---Copy Constructor Test Successful---\n\n");
}

