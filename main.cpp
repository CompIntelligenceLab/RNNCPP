#include <stdio.h>
#include <math.h>
#include <string>
#include "model.h"
#include "activations.h"
#include "optimizer.h"
#include "objective.h"
#include "layers.h"
#include "dense_layer.h"
#include "lstm_layer.h"
#include "gmm_layer.h"

int main() {
	Model* m  = new Model(1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	Layer* dense = new DenseLayer(5, "dense");
	m->add(dense);
	Layer* dense1 = new DenseLayer(3, "dense");
	m->add(dense1);

	printf("layer 0: input dim: %d\n", dense->getInputDim());
	printf("layer 0: output dim: %d\n", dense->getLayerSize());
	printf("layer 1: input dim: %d\n", dense1->getInputDim());
	printf("layer 1: output dim: %d\n", dense1->getLayerSize());

	Sigmoid* sig = new Sigmoid();

	printf("before init\n");
	m->initializeWeights();
	printf("after init\n");

	Optimizer* opt = new RMSProp("myrmsprop");
	m->setOptimizer(opt);
	opt->print();

	m->print();
	printf("-----------------\n");

	Model n = *m;
	n.setName("model n");
	n.print();

	// prediction
	VF3D x(1,1,1);
	x(0,0,0) = 0.5;
	m->predict(x);

	printf("--------------------\n");
	Objective* obj1 = new MeanSquareError("mse gordon");
	//Objective* obj1 = new Objective("mse gordon");
	//Objective* obj1 = new Objective();
	MeanSquareError mse1("mse_one");;
	MeanSquareError mse2("mse_two");;
	printf("ob1 name: %s\n", obj1->getName().c_str());
	printf("mse1 name: %s\n", mse1.getName().c_str());
	mse2 = mse1;
	printf("mse2 name: %s\n", mse2.getName().c_str());
	MeanSquareError mse3("xxx");
	MeanSquareError mse4("xxx"); 
	printf("mse4 name: %s\n", mse4.getName().c_str());
}
