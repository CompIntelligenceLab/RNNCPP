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
	//printf("cub3 size: %d\n", arma::size(cub3)); // only works in C++11
	//exit(0);

	VF3D cub20(3,4,5);
	cub20.randu();
	VF3D cub21(cub20); // copy constructor
	VF3D cub22(size(cub20)); // constructor that creates 3D array, same size as cub20)

	printf("cub20(1,1,1)= %f\n", cub20(1,1,1));
	printf("cub21(1,1,1)= %f\n", cub21(1,1,1));

	printf("----------- Model -----------------\n");
	WEIGHTS w1, w2;
	int input_dim = 2;
	Model* m  = new Model(input_dim); // argument is input_dim of model

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	Layer* dense = new DenseLayer(5, "dense");
	m->add(dense);
	Layer* dense1 = new DenseLayer(3, "dense");
	m->add(dense1); // weights should be defined when add is done

	printf("...........\n");
	w1 = dense->getWeights();
	printf("   layer 0: weights: %d, %d\n", w1.n_rows, w1.n_cols);
	w2 = dense1->getWeights();
	printf("   layer 1: weights: %d, %d\n", w2.n_rows, w2.n_cols);

	printf("before init\n");
	m->initializeWeights();
	printf("after init\n");

	printf("...........\n");
	w1 = dense->getWeights();
	printf("   layer 0: weights: %d, %d\n", w1.n_rows, w1.n_cols);
	w2 = dense1->getWeights();
	printf("   layer 1: weights: %d, %d\n", w2.n_rows, w2.n_cols);
	exit(0);

	printf("check 1 --------------\n");
	printf("layer 0: input dim: %d\n", dense->getInputDim());
	printf("layer 0: output dim: %d\n", dense->getLayerSize());
	printf("layer 1: input dim: %d\n", dense1->getInputDim());
	printf("layer 1: output dim: %d\n", dense1->getLayerSize());

	Sigmoid* sig = new Sigmoid();

	exit(0);

	Optimizer* opt = new RMSProp("myrmsprop");
	m->setOptimizer(opt);
	opt->print();

	m->print();
	printf("-----------------\n");

	#if 0
	Model n = *m;
	n.setName("model n");
	n.print();
	#endif

	#if 0
	printf("check 1 --------------\n");
	printf("layer 0: input dim: %d\n", dense->getInputDim());
	printf("layer 0: output dim: %d\n", dense->getLayerSize());
	printf("layer 1: input dim: %d\n", dense1->getInputDim());
	printf("layer 1: output dim: %d\n", dense1->getLayerSize());
	dense->print("dense layer");
	dense1->print("dense1 layer");
	#endif

	// prediction
	#if 0
	VF3D x(1,1,1);
	x(0,0,0) = 0.5;
	m->predict(x);
	#endif

	VF2D_F xf(3);
	VF2D y; y.randu(1,1);

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(1,1);
	}
	m->predict(xf);

	exit(0);

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
