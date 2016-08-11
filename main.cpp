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
	VF3D cub1(3,4,5);
	VF3D cub2(size(cub1));
	VF3D cub3(cub1);
	cub3.randu();
	for (int i=0; i < cub3.n_rows; i++) {
	for (int j=0; j < cub3.n_cols; j++) {
	for (int k=0; k < cub3.n_slices; k++) {
		printf("cub3(%d,%d,%d)= %f\n", i,j,k, cub3(i,j,k));
	}}}
	printf("*** cub3(59)= %f\n", cub3(59));
	for (int i=0; i < cub3.size(); i++) {
		printf("cub3(%d)= %f\n", i, cub3(i));
	}
	VF2D cub5 = cub3.slice(2);
	printf("cub5.size()= %d\n", cub5.size());

	printf("cub2: %d\n", cub2.n_rows);
	printf("cub2: %d\n", cub2.n_cols);
	printf("cube2 size: %d\n", cub2.size());
	printf("cub3: %d\n", cub3.n_rows);
	printf("cub3: %d\n", cub3.n_cols);
	printf("cub32 size: %d\n", cub3.size());
	printf("cub3 size: %d\n", arma::size(cub3));
	//exit(0);

	VF3D cub20(3,4,5);
	cub20.randu();
	VF3D cub21(cub20); // copy constructor
	VF3D cub22(size(cub20)); // constructor that creates 3D array, same size as cub20)

	printf("cub20(1,1,1)= %f\n", cub20(1,1,1));
	printf("cub21(1,1,1)= %f\n", cub21(1,1,1));
	exit(0);

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

	VF2D_F xf(3);
	VF2D y; y.randu(1,1);

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(1,1);
	}
	m->predict(xf);

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
