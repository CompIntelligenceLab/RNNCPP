
#include <stdio.h>
#include <math.h>
#include "activations.h"
#include "optimizer.h"
#include "weights.h"
#include "layers.h"

int main() 
{
	printf("=================================\n");
	Optimizer* opt = new Optimizer("optGE");
	Optimizer opt1(*opt);
	Optimizer opt2 = *opt;

	opt->print("\n--> print opt\n");
	opt1.print("\n--> print opt1\n");
	opt2.print("\n--> print opt2\n");

	printf("=================================\n");
	Weights* weight = new Weights(1,1,"wghtGE");
	Weights w1(*weight);
	Weights w2 = *weight;

	weight->print("\n --> print wght\n");
	w1.print("\n --> print w1\n");
	w2.print("\n --> print w2\n");

	printf("=================================\n");
	int layer_size = 5;
	Layer* layer = new Layer(layer_size, "layerGE");

	// Both these statements lead to to identical objects being deleted. Only one of these
	// statements leads to correct code. 
	Layer layer3(layer_size, "layer3GE");
	Layer layer2;
	layer2 = layer3;

	//Layer layer1(*layer);
	#if 0

	//layer->print("\n--> print layer\n");
	layer1.print("\n--> print layer1\n");
	exit(0);
	layer2.print("\n--> print layer2\n");
	//exit(0);
	#endif
  printf("\n\n---Copy Constructor Test Successful---\n\n");
}

