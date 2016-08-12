#include <stdio.h>
#include <math.h>
#include <string>
#include <assert.h>
//#include <iostream>
//#include <fstream>
#include "model.h"
#include "activations.h"
#include "optimizer.h"
#include "objective.h"
#include "layers.h"
#include "dense_layer.h"
#include "lstm_layer.h"
#include "gmm_layer.h"
#include "input_layer.h"

using namespace arma;

//----------------------------------------------------------------------
void testCube()
{
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
}

//----------------------------------------------------------------------
void testPredict()
{
	printf("----------- Test Predict -----------------\n");
	WEIGHTS w1, w2;
	int input_dim = 2; 
	Model* m  = new Model(input_dim); // argument is input_dim of model
	assert(m->getBatchSize() == 1);
	m->setBatchSize(2); 
	assert(m->getBatchSize() == 2);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	Layer* input = new InputLayer(2, "input_layer");
	m->add(input);
	Layer* dense = new DenseLayer(5, "dense");
	m->add(dense);
	Layer* dense1 = new DenseLayer(3, "dense");
	m->add(dense1); // weights should be defined when add is done

	m->initializeWeights();
	Optimizer* opt = new RMSProp("myrmsprop");
	m->setOptimizer(opt);

	int batch_size = m->getBatchSize();
	VF2D_F xf(batch_size);
	VF2D_F yf(batch_size); 

	input_dim = m->getInputDim();

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(input_dim,1);
		yf[i].randu(input_dim,1);
	}
	printf("sizeof(xf)= %d\n", sizeof(xf));

	// Compute prediction through the network
	m->initializeWeights();
	printf("after initialize\n");
	w1 = dense->getWeights();
	printf("w1= %d, %d\n", w1.n_rows, w1.n_cols);
	w2 = dense1->getWeights();

	// must initialize weights before predicting anything
	VF2D_F pred_calc = m->predict(xf);
	pred_calc.print("predict");

	// computation of x1 = w1*x
	// computation of x2 = w2*x1 = w2 * w1 * x

	VF2D_F x1(xf.n_rows);
	VF2D_F x2(xf.n_rows);

	printf("=============================================\n");
	printf("------ BEGIN CHECK PREDICT --------------\n");
	w1.print("w1");
	xf.print("xf");
	printf("xf= fields: %d, rows: %d, cols: %d, size: %d\n", xf.n_rows, xf[0].n_rows, xf[0].n_cols, xf.size());

	for (int b=0; b < xf.size(); b++) {
		x1[b] = arma::Mat<float>(w1.n_rows, xf[b].n_cols);
		for (int i=0; i < xf[b].n_cols; i++) {
			for (int l=0; l < w1.n_rows; l++) {
				float xx = 0;
				for (int j=0; j < xf[b].n_rows; j++) {
					xx += w1(l,j) * xf[b](j,i);
				//printf("xx= %f, b,i,l,j= %d, %d, %d, %d\n", xx, b,i,l,j);
				}
				x1[b](l,i) = xx;
			}
		}
	}

	x1 = dense->getActivation()(x1);

	w2.print("w2");
	x1.print("x1 = w1*xf");
	xf = x1;
	printf("xf= fields: %d, rows: %d, cols: %d, size: %d\n", xf.n_rows, xf[0].n_rows, xf[0].n_cols, xf.size());

	for (int b=0; b < xf.size(); b++) {
		x1[b] = arma::Mat<float>(w2.n_rows, xf[b].n_cols);
		for (int i=0; i < xf[b].n_cols; i++) {
			for (int l=0; l < w2.n_rows; l++) {
				float xx = 0;
				for (int j=0; j < xf[b].n_rows; j++) {
					xx += w2(l,j) * xf[b](j,i);
				//printf("xx= %f, b,i,l,j= %d, %d, %d, %d\n", xx, b,i,l,j);
				}
				x1[b](l,i) = xx;
			}
		}
	}

	x1 = dense1->getActivation()(x1);

	x1.print("x1 = w2*(w1*xf)");
	VF2D_F pred_exact = x1;

	for (int b=0; b < pred_calc.n_rows; b++) {
		bool bb = APPROX_EQUAL(pred_calc(b), pred_exact(b)); assert(bb);
		// For some reason, I cannot combine both in one line. 
		//assert(APPROX_EQUAL(pred_calc(b), pred_exact(b)));
	}

	printf("------ END CHECK PREDICT --------------\n");
}
//----------------------------------------------------------------------
void testModel()
{
	printf("----------- Model -----------------\n");
	WEIGHTS w1, w2;
	int input_dim = 2;
	Model* m  = new Model(input_dim); // argument is input_dim of model

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	Layer* input = new InputLayer(2, "input_layer");
	m->add(input);
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

	Sigmoid* sig = new Sigmoid();

	Optimizer* opt = new RMSProp("myrmsprop");
	m->setOptimizer(opt);
	opt->print();

	m->print();
	printf("-----------------\n");

	int batch_size = m->getBatchSize();
	VF2D_F xf(batch_size);
	VF2D_F yf(batch_size); 

	input_dim = m->getInputDim();
	printf("input_dim= %d\n", input_dim);
	//exit(0);

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(1,1);
		yf[i].randu(1,1);
	}
	VF2D_F pred = m->predict(xf);
	printf("sizeof(pred)= %d\n", sizeof(pred));
	pred.print("prediction: ");

	m->train(xf,yf);

	// Of course, I must check the prediction manually. 


	exit(0);
}

//----------------------------------------------------------------------
void testObjective()
{
	printf("--------------------\n");
	Objective* obj1 = new MeanSquareError("mse gordon");
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

//----------------------------------------------------------------------
int main() 
{
	//testCube();
	//testModel();
	testPredict();
	//testObjective();
}
