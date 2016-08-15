#include <stdio.h>
#include <math.h>
#include <string>
#include <assert.h>
#include <iostream>
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
#include "print_utils.h"

using namespace arma;
using namespace std;

//----------------------------------------------------------------------
void testCube()
{
	VF3D cub1(3,4,5);
	VF3D cub2(size(cub1));
	VF3D cub3(cub1);
	cub3.randu();

	U::print(cub3, "cub3");
	cub3.print("cube");

	#if 0
	for (int i=0; i < cub3.n_rows; i++) {
	for (int j=0; j < cub3.n_cols; j++) {
	for (int k=0; k < cub3.n_slices; k++) {
		printf("cub3(%d,%d,%d)= %f\n", i,j,k, cub3(i,j,k));
	}}}
	#endif

	printf("*** cub3(59)= %f\n", cub3(59));
	for (int i=0; i < cub3.size(); i++) {
		printf("cub3(%d)= %f\n", i, cub3(i));
	}

	VF2D cub5 = cub3.slice(2);
	printf("cub5.size()= %d\n", (int) cub5.size());

	U::print(cub2,"cub2");

	VF3D cub20(3,4,5);
	cub20.randu();
	VF3D cub21(cub20); // copy constructor
	VF3D cub22(size(cub20)); // constructor that creates 3D array, same size as cub20)

	printf("cub20(1,1,1)= %f\n", cub20(1,1,1));
	printf("cub21(1,1,1)= %f\n", cub21(1,1,1));
}

//----------------------------------------------------------------------
void testPredict()
// MUST BE RETESTED
{
#if 0
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

	// Compute prediction through the network
	//m->initializeWeights();
	w1 = dense->getWeights();
	w2 = dense1->getWeights();

	//Optimizer* opt = new RMSProp("myrmsprop");
	//m->setOptimizer(opt);

	int batch_size = m->getBatchSize();
	VF2D_F xf(batch_size);
	VF2D_F yf(batch_size); 

	input_dim = m->getInputDim();

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(input_dim,1);
		yf[i].randu(input_dim,1);
	}
	printf("sizeof(xf)= %d\n", sizeof(xf));

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
#endif
}
//----------------------------------------------------------------------
void testModel()
{
#if 0
	WEIGHTS w1, w2;
	int input_dim = 2;
	Model* m  = new Model(input_dim); // argument is input_dim of model
    m->setBatchSize(2);
	assert(m->getBatchSize() == 2);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	Layer* input = new InputLayer(2, "input_layer");
	m->add(input);
	Layer* dense = new DenseLayer(5, "dense");
	m->add(dense);
	Layer* dense1 = new DenseLayer(3, "dense");
	m->add(dense1); // weights should be defined when add is done
	Layer* dense2 = new DenseLayer(input_dim, "dense");
	m->add(dense2); // weights should be defined when add is done

	w1 = dense->getWeights();
	w2 = dense1->getWeights();
	w1.print("w1");
	w2.print("w2");

	Optimizer* opt = new RMSProp("myrmsprop");
	m->setOptimizer(opt);
	opt->print();

	//m->print();

	int batch_size = m->getBatchSize();
	VF2D_F xf(batch_size);
	VF2D_F yf(batch_size); 

	input_dim = m->getInputDim();

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(input_dim, 1);
		yf[i].randu(input_dim, 1);
	}

	xf.print("xf");
	VF2D_F pred = m->predict(xf);
	pred.print("prediction: ");

	m->train(xf,yf);


	exit(0);
#endif
}
//----------------------------------------------------------------------
// TEST MODELS for structure
void testModel1()
{
	printf("\n --- testModel1 ---\n");
	int input_dim = 1;
	Model* m  = new Model(); // argument is input_dim of model
	assert(m->getBatchSize() == 1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input   = new InputLayer(2, "input_layer");
	Layer* dense0  = new DenseLayer(5, "dense0");
	Layer* dense1  = new DenseLayer(3, "dense1");
	Layer* dense2  = new DenseLayer(4, "dense2");
	Layer* dense3  = new DenseLayer(6, "dense3");

	m->add(input, dense0);
	m->add(dense0, dense1);
	m->add(dense1, dense2);
	m->add(dense2, dense3);

	m->checkIntegrity();
	m->printSummary();
}
//----------------------------------------------------------------------
// TEST MODELS for structure
void testModel2()
{
	printf("\n --- testModel2 ---\n");
	Model* m  = new Model(); // argument is input_dim of model
	assert(m->getBatchSize() == 1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	int input_dim = 2;
	Layer* input   = new InputLayer(input_dim, "input_layer");
	Layer* dense1  = new DenseLayer(5, "dense");
	Layer* dense2  = new DenseLayer(3, "dense");
	Layer* dense3  = new DenseLayer(4, "dense");
	Layer* dense4  = new DenseLayer(6, "dense");

	/*  S: Spatial, T: Temporal

	          S
	   input ---> dense1
         \          | T
          \         | 
           \        v    S           S
	        ---> dense2 ---> dense3 ---> dense4
	*/

	m->add(0, input);
	m->add(input, dense1);
	m->add(input, dense2);
	m->add(dense2, dense3);
	m->add(dense1, dense2);
	m->add(dense3, dense4);

	m->checkIntegrity();
	m->printSummary();
}
//----------------------------------------------------------------------
void testFuncModel1()
{
	printf("\n --- testFuncModel1 ---\n");

	// In reality, the model should not have an input_dim. 
	Model* m  = new Model(); 
	assert(m->getBatchSize() == 1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	// Must make sure that input_dim of input layer is the same as model->input_dim
	int input_dim = 2;
	Layer* input   = new InputLayer(input_dim, "input_layer");  
	Layer* dense0  = new DenseLayer(5, "dense0");
	Layer* dense1  = new DenseLayer(3, "dense1");
	Layer* dense2  = new DenseLayer(4, "dense2");
	Layer* dense3  = new DenseLayer(6, "dense3");

	m->add(0, input);
	m->add(input, dense0);
	m->add(dense0, dense1);
	m->add(dense1, dense2);
	m->add(dense2, dense3);

	m->checkIntegrity();
	m->printSummary();
	//----------

	int batch_size = m->getBatchSize();
	VF2D_F xf(batch_size);
	VF2D_F yf(batch_size); 

	input_dim = input->getLayerSize();

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(input_dim, 1);
		yf[i].randu(input_dim, 1);
	}

	printf("   nlayer layer_size: %d\n", m->getLayers()[0]->getLayerSize());
	printf("   input layer_size: %d\n", input->getLayerSize());
	
	VF2D_F pred = m->predict(xf);
	pred.print("funcModel, predict:");
}
//----------------------------------------------------------------------
void testFuncModel2()
{
	printf("\n --- testModel2 ---\n");
	Model* m  = new Model(); 
	assert(m->getBatchSize() == 1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	int input_dim = 1;
	Layer* input   = new InputLayer(input_dim, "input_layer");
	Layer* dense1  = new DenseLayer(5, "dense");
	Layer* dense2  = new DenseLayer(3, "dense");
	Layer* dense3  = new DenseLayer(4, "dense");
	Layer* dense4  = new DenseLayer(6, "dense");

	/*  S: Spatial, T: Temporal

	          S
	   input ---> dense1
         \          | T
          \         | 
           \        v    S           S
	        ---> dense2 ---> dense3 ---> dense4
	*/

	m->add(0, input); // changs input_dim to zero. Why? 

	// Currently, the layer list will contain a layer name twice. 
	m->add(input, dense1);
	m->add(input, dense2);
	m->add(dense2, dense3);
	m->add(dense1, dense2);
	m->add(dense3, dense4);

	input_dim = input->getInputDim();
	printf("input_dim= %d\n", input_dim);


// 
    //for (int i=0; i < m->getLayers().size(); i++) {
		//m->getLayers()[i]->printSummary();
	//}
	//exit(0);


	m->checkIntegrity();
	m->printSummary();

	int batch_size = m->getBatchSize();
	VF2D_F xf(batch_size);
	VF2D_F yf(batch_size); 

	printf("batch_size= %d\n", batch_size);
	input_dim = input->getInputDim();
	printf("input_dim= %d\n", input_dim);
	printf("xf.size= %llu", xf.n_rows);

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(input_dim, 1);
		yf[i].randu(input_dim, 1);
	}

	printf("   nlayer layer_size: %d\n", m->getLayers()[0]->getLayerSize());
	printf("   input layer_size: %d\n", input->getLayerSize());

	xf.print("xf");
	
	VF2D_F pred = m->predictComplexMaybeWorks(xf);
	pred.print("funcModel, predict:");
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
	testCube();
	//testModel();
	//testFuncModel1();
	testFuncModel2();
	//testModel1();
	//testModel2();
	//testPredict();
	//testObjective();
}
