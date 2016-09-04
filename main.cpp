#include <stdio.h>
#include <math.h>
#include <string>
#include <assert.h>
#include <iostream>
//#include <fstream>
#include "model.h"
#include "activations.h"
#include "connection.h"
#include "optimizer.h"
#include "objective.h"
#include "layers.h"
#include "recurrent_layer.h"
#include "dense_layer.h"
#include "out_layer.h"
#include "lstm_layer.h"
#include "gmm_layer.h"
#include "input_layer.h"
#include "print_utils.h"

using namespace arma;
using namespace std;

void testData(Model& m, VF2D_F& xf, VF2D_F& yf, VF2D_F&);
WEIGHT weightDerivative(Model* m, Connection& con, float inc, VF2D_F& xf, VF2D_F& exact);
BIAS biasDerivative(Model* m, Layer& layer, float inc, VF2D_F& xf, VF2D_F& exact);
void testRecurrentModel1(int nb_batch);
void testRecurrentModel2(int nb_batch);
void testRecurrentModel3(int nb_batch); // testRecurrentModel2 with no recurrence
void testRecurrentModel4(int nb_batch); // testRecurrentModel2 with single recurrent node
WEIGHT dLdw(1,1);
BIAS dLdb(1);

//----------------------------------------------------------------------
float runModel(Model* m)
{
	m->printSummary();
	m->connectionOrderClean(); // no print statements

	VF2D_F xf, yf, exact;
	testData(*m, xf, yf, exact);

	Layer* outLayer = m->getOutputLayers()[0];
	int output_dim = outLayer->getOutputDim();
	printf("output_dim = %d\n", output_dim);

	CONNECTIONS connections = m->getConnections();

	for (int b=0; b < m->getBatchSize(); b++) {
		xf(b) = .3;
		yf(b) = .4;
		exact(b) = arma::Mat<float>(output_dim,1);
		exact(b).ones();
		exact(b) *= .5;
	}
	//exact.print("exact");
	//exit(0);
	float w =  m->getConnections()[0]->getWeight()[0];
	printf("w = %f\n", w);

	printf("*** connections.size() = %d\n", connections.size());
	for (int c=0; c < connections.size(); c++) {
		connections[c]->printSummary();
	}
	// xf = .3
	// yf = w * .3;
	w =  m->getConnections()[0]->getWeight()(0,0);
	printf("w[0] = %f\n", w);
	printf("w[0]*xf = %f\n", w*xf(0)(0,0));
	w =  m->getConnections()[1]->getWeight()(0,0);
	printf("w[1] = %f\n", w);
	printf("w[1]*xf = %f\n", w*xf(0)(0,0));

	//const WEIGHT& wght =  m->getConnections()[1]->getWeight();
	//VF2D exact_prediction = wght % xf(0);
	// WRONG RESULT for Model1a

	VF2D_F pred = m->predictViaConnectionsBias(xf);

	pred.print("predicted value");
	printf("+++++++++++\n");
	//exact_prediction.print("exact predicted");
	#if 0
	VF2D err = (pred(0) - exact_prediction); 
	err.print("absolute error on prediction");
	err = err / exact_prediction;
	err.print("relative error on prediction");
	printf("----------------------------\n");
	#endif

	float inc = .0001;
	WEIGHT fd_dLdw;
	// First connection is between 0 and input (does not count)
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary();
		fd_dLdw = weightDerivative(m, *connections[c], inc, xf, exact);
	}

	// Exact dL/dw
	VF2D dLdw_analytical = 2.*(exact(0) - pred(0)) * xf(0);
	printf("t=0, Analytical dLdw: = %f\n", dLdw(0));
	printf("t=0, F-D  derivative: = %f\n", fd_dLdw(0));
exit(0);

/*********************
storeGradientsInLayers, Layer (input_layer0), layer_size: 1
layer outputs, 0.3000
layer gradient, 1.0000
layer Delta, [matrix size: 0x0]

storeGradientsInLayers, Layer (dense1), layer_size: 1
layer outputs, 0.1373
layer gradient, 1.0000
layer Delta, 0.7248
********* ENTER storeDactivationDoutputInLayers() ************** connectionConnection (weight1), weight(1, 1), layers: (input_layer0, dense1), type: spatial
layer_toLayer (dense1), layer_size: 1
layer_fromLayer (input_layer0), layer_size: 1
grad[0], 1.0000
old_deriv[0], 0.7248
wght, 0.4577
prod[0], 0.3317
********* EXIT storeDactivationDoutputInLayers() **************
********** ENTER storeDLossDweightInConnections ***********
Connection, Connection (weight1), weight(1, 1), layers: (input_layer0, dense1), type: spatial
layer_to->getGradient, grad, 1.0000
layer_to->getDelta, old_deriv, 0.7248
storeDLossDweightInConnections, prod 0.0995
********** EXIT storeDLossDweightInConnections ***********
***************** EXIT BACKPROPVIACONNECTIONS <<<<<<<<<<<<<<<<<<<<<<
*******************/

	m->backPropagationViaConnections(exact, pred);
	printf("BackProp derivatives\n");
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary("Connection (backprop)");
		connections[c]->getDelta().print("delta");
	}

	// Go through connections and print out weight derivatives
	//printf("gordon\n"); exit(0);
}
//----------------------------------------------------------------------
WEIGHT weightDerivative(Model* m, Connection& con, float inc, VF2D_F& xf, VF2D_F& exact)
{
	// I'd expect the code to work with nb_batch=1 
	//printf("********** ENTER weightDerivative *************, \n");

	WEIGHT w0 = con.getWeight();
	int rrows = w0.n_rows;
	int ccols = w0.n_cols;
	dLdw = arma::Mat<float>(size(w0));
	dLdw.zeros();
	Objective* mse = new MeanSquareError();

	for (int rr=0; rr < rrows; rr++) {
	for (int cc=0; cc < ccols; cc++) {

		WEIGHT& wp = con.getWeight(); 
		wp(rr,cc) += inc;
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);

		WEIGHT& wm = con.getWeight(); 
		wm(rr,cc) -= (2.*inc);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of floats
		//U::print(pred_p, "pred_p"); 
		//U::print(loss_p, "loss_p"); 
		//loss_p.print("loss_p");
		//exit(0);
		LOSS loss_n = (*mse)(exact, pred_n);

		//loss_n(0) = arma::sum(loss_n(0), 1);
		//loss_p(0) = arma::sum(loss_p(0), 1);
		//U::print(loss_n, "loss_n");
		//loss_n.print("loss_n");
		//loss_n(0).print("loss_n(0)");
		//exit(0);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdw(rr, cc) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*inc);
	}}
	//con.printSummary("weightDerivative");
	//dLdw.print("dLdw");
	//printf("********** EXIT weightDerivative *************, \n");
	return dLdw;
}
//----------------------------------------------------------------------
BIAS biasDerivative(Model* m, Layer& layer, float inc, VF2D_F& xf, VF2D_F& exact)
{
	// I'd expect the code to work with nb_batch=1 
	//printf("********** ENTER biasDerivative *************, \n");

	BIAS bias = layer.getBias();
	int layer_size = layer.getLayerSize();
	dLdb = BIAS(size(bias));
	dLdb.zeros();
	Objective* mse = new MeanSquareError();

	for (int rr=0; rr < layer_size; rr++) {

		BIAS& biasp = layer.getBias();
		biasp(rr) += inc;
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);

		BIAS& biasm = layer.getBias(); 
		biasm(rr) -= (2.*inc);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of floats
		LOSS loss_n = (*mse)(exact, pred_n);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdb(rr) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*inc);
	}
	//printf("********** EXIT biasDerivative *************, \n");
	return dLdb;
}
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
float runModelRecurrent(Model* m)
{
#if 0
	m->printSummary();
	m->connectionOrderClean(); // no print statements

	VF2D_F xf, yf, exact;
	testData(*m, xf, yf, exact);

	Layer* outLayer = m->getOutputLayers()[0];
	int output_dim = outLayer->getOutputDim();
	printf("output_dim = %d\n", output_dim);

	CONNECTIONS connections = m->getConnections();

	//U::print(xf, "xf"); exit(0);
	for (int b=0; b < m->getBatchSize(); b++) {
		xf(b).fill(.3);
		yf(b).fill(.4);
		exact(b) = arma::Mat<float>(output_dim, m->getSeqLen());
		exact(b).fill(.5);
	}
	//U::print(xf, "xf"); exit(0);

	/*** Analytical solution with two time steps (seq_len=2)
	 Activation: Identity 
	 x = .3;   w = .2; exact: .5
	 input to dense0: x = .3
	 output to dense0: x = .3
	 input to dense1: w*x = .06
	 output to dense1: w*x = .06
	 Loss function: (.5-.06)**2 = .44^2 = .1936

	 2nd prediction. Recursion kicks in
	 Input/output to dense0: x = 0.3
	 Input to dense1: w*x + wloop*layer1->output = 0.06 + 0.1315*.06 = 0.06 + 0.00789
	 Output to dense1: 0.06789
	***/

	//exact.print("exact");
	WEIGHT w0(1,1), w1(1,1);
	w0(0,0) = .2;
	w1(0,0) = .1315;
	m->getConnections()[0]->setWeight(w0);
	m->getConnections()[1]->setWeight(w0);
	m->getLayers()[1]->recurrent_conn->setWeight(w1);

	m->getConnections()[0]->getWeight().print("weight0");
	m->getConnections()[1]->getWeight().print("weight1");
	m->getLayers()[1]->recurrent_conn->getWeight().print("weight_recurrent");

	printf("*** connections.size() = %d\n", m->getConnections().size());
	for (int c=0; c < connections.size(); c++) {
		connections[c]->printSummary();
	}
	// xf = .3
	// yf = w * .3;
	float w;
	w =  m->getConnections()[0]->getWeight()(0,0);
	printf("w[0] = %f\n", w);
	printf("w[0]*xf = %f\n", w*xf(0)(0,0));
	w =  m->getConnections()[1]->getWeight()(0,0);
	printf("w[1] = %f\n", w);
	printf("w[1]*xf = %f\n", w*xf(0)(0,0));

	VF2D_F pred;

	for (int i=0; i < 1; i++) {
		U::print(xf, "xf");
		pred = m->predictViaConnections(xf);
		U::print(pred, "pred");
	}
	U::print(pred, "pred");
	U::print(exact, "exact");
	pred.print("pred");
	exact.print("exact");
	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary("Connection (backprop)");
		connections[c]->getDelta().print("delta");
	}
	testRecurrentModel1(1);
	exit(0);




	printf("-------------\n");
	pred.print("first prediction\n");
	//exit(0);
	pred = m->predictViaConnections(xf);
	pred.print("second prediction\n");
	exit(0);

	float inc = .0001;
	WEIGHT fd_dLdw;
	// First connection is between 0 and input (does not count)
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary();
		fd_dLdw = weightDerivative(m, *connections[c], inc, xf, exact);
		fd_dLdw.print("weightDerivative");
	}

	// Exact dL/dw
	VF2D dLdw_analytical = 2.*(exact(0) - pred(0)) * xf(0);
	printf("Analytical dLdw: = %f\n", dLdw(0));
	printf("F-D  derivative: = %f\n", fd_dLdw(0));
exit(0);

	m->backPropagationViaConnections(exact, pred);
	printf("BackProp derivatives\n");
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary("Connection (backprop)");
		connections[c]->getDelta().print("delta");
	}

	// Go through connections and print out weight derivatives
	//printf("gordon\n"); exit(0);
#endif
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
// TEST MODELS for structure
void testModel1a(int nb_batch)
{
/***
	Simplest possible network: two nodes with the identity activation. 
	seq_len = nb_batch = 1
	This allows testing via simple matrix-multiplication

                 w1
	    input ---------> dense --> loss    (loss is attached to the output layer)
***/
	printf("\n --- testModel1a ---\n");
	int input_dim = 1;
	Model* m  = new Model(); // argument is input_dim of model

	// I am not sure that batchSize and nb_batch are the same thing
	m->setBatchSize(nb_batch);
	assert(m->getBatchSize() == nb_batch);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input  = new InputLayer(1, "input_layer");
	Layer* dense  = new DenseLayer(1, "dense");
	Layer* out    = new OutLayer(1, "out");

	m->add(0,      input);
	m->add(input,  dense);
	m->add(dense,  out);

	dense->setActivation(new Identity());
	input->setActivation(new Identity());
	out->setActivation(new Identity());

	m->addInputLayer(input);
	m->addOutputLayer(dense);
	runModel(m);
}
//----------------------------------------------------------------------
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
	Layer* dense0  = new DenseLayer(5, "dense");
	Layer* dense1  = new DenseLayer(3, "dense");
	Layer* dense2  = new DenseLayer(4, "dense");
	Layer* dense3  = new DenseLayer(6, "dense");

	m->add(input, dense0);
	m->add(dense0, dense1);
	m->add(dense1, dense2);
	m->add(dense2, dense3);

	m->addInputLayer(input);
	m->addOutputLayer(dense3);
	runModel(m);
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
	m->add(dense3, dense4);
	m->add(input, dense1);
	m->add(input, dense2);
	m->add(dense2, dense3);
	m->add(dense1, dense2);

	m->addInputLayer(input);
	m->addOutputLayer(dense4);

	runModel(m);

	#if 0
	//m->checkIntegrity();  // seg error
	m->printSummary();
	m->connectionOrderClean(); // no print statements

	VF2D_F xf, yf, exact;
	testData(*m, xf, yf, exact);
	m->predictViaConnections(xf);
	exit(0);
	#endif
}
//----------------------------------------------------------------------
void testFuncModel1()
{
	printf("\n --- testFuncModel1 ---\n");

	// In reality, the model should not have an input_dim. 
	Model* m  = new Model(); 
    m->setBatchSize(1);
	assert(m->getBatchSize() == 1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	// Must make sure that input_dim of input layer is the same as model->input_dim
	int input_dim = 1;
	int layer_size = 1;
	Layer* input   = new InputLayer(input_dim, "input_layer");  
	Layer* dense0  = new DenseLayer(layer_size, "dense");  // weights between dense0 and dense1
	Layer* dense1  = new DenseLayer(layer_size, "dense");  // weights btween dense1 and dense2
	Layer* dense2  = new DenseLayer(layer_size, "dense");
	Layer* dense3  = new DenseLayer(layer_size, "dense");

	input->setActivation(new Identity());
	dense0->setActivation(new Identity());
	dense1->setActivation(new Identity());
	dense2->setActivation(new Identity());
	dense3->setActivation(new Identity());

	m->add(0, input);
	m->add(input, dense0);
	m->add(dense0, dense1);
	m->add(dense1, dense2);
	m->add(dense2, dense3);

	/*
	    input --> dense0 --> dense1 --> dense2 --> dense3
	*/

	m->addInputLayer(input);
	m->addOutputLayer(dense3);
	m->addProbeLayer(dense0);
	m->addProbeLayer(dense1);
	m->addLossLayer(dense1);

	runModel(m);

	// Backprop works
	return;
}
//----------------------------------------------------------------------
void testFuncModel2()
{
	printf("\n --- testFuncModel2 ---\n");
	Model* m  = new Model(); 
	m->setBatchSize(1);
	assert(m->getBatchSize() == 1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	int input_dim = 1;
	Layer* input   = new InputLayer(input_dim, "input_layer");
	//Layer* dense1  = new DenseLayer(5, "dense");
	//Layer* dense2  = new DenseLayer(3, "dense");
	//Layer* dense3  = new DenseLayer(4, "dense");
	//Layer* dense4  = new DenseLayer(6, "dense");
	Layer* dense1  = new DenseLayer(1, "dense");
	Layer* dense2  = new DenseLayer(1, "dense");
	Layer* dense3  = new DenseLayer(1, "dense");
	Layer* dense4  = new DenseLayer(1, "dense");

	/*  S: Spatial, T: Temporal

	          S
	   input ---> dense1
         \          | T
          \         | 
           \        v    S           S
	        ---> dense2 ---> dense3 ---> dense4
	*/

	m->add(0, input); // changs input_dim to zero. Why? 

	m->add(input, dense1);
	//m->add(input, dense2);
	//m->add(dense2, dense3);
	m->add(dense1, dense2);
	//m->add(dense3, dense4);

	m->addInputLayer(input);
	m->addOutputLayer(dense2);

	runModel(m);
}
//----------------------------------------------------------------------
void testFuncModel3()
{
	printf("\n --- testModel2 ---\n");
	Model* m  = new Model(); 
	m->setBatchSize(1);
	assert(m->getBatchSize() == 1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	int input_dim = 1;
	Layer* input   = new InputLayer(input_dim, "input_layer");
	Layer* dense1  = new DenseLayer(5, "dense");
	Layer* dense2  = new DenseLayer(3, "dense");
	Layer* dense3  = new DenseLayer(4, "dense");

	/*  S: Spatial, T: Temporal

	          S
	   input ---> dense1 -------> dense3 --> loss
         \                 ^ 
          \                |
           \               |
	        ---> dense2 ---|
	*/

	m->add(0, input); // changs input_dim to zero. Why? 
	m->add(input, dense1);
	m->add(input, dense2);
	m->add(dense1, dense3);
	m->add(dense2, dense3);

	m->addInputLayer(input);
	m->addOutputLayer(dense3);

	runModel(m);
}
//----------------------------------------------------------------------
void testData(Model& m, VF2D_F& xf, VF2D_F& yf, VF2D_F& exact)
{
	int batch_size = m.getBatchSize();
	xf.set_size(batch_size);
	yf.set_size(batch_size);
	exact.set_size(batch_size);

	Layer* input = m.getInputLayers()[0];
	int input_dim = input->getInputDim();
	printf("input_dim= %d\n", input_dim);

	int output_dim = m.getOutputLayers()[0]->getOutputDim();
	printf("output_dim= %d\n", output_dim);
	int seq_len = m.getSeqLen();

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(input_dim, seq_len); // uniform random numbers
		yf[i].randu(input_dim, seq_len);
		exact[i].randu(output_dim, seq_len);
	}
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
void testMatMulSequences()
{
	printf("\n --- testMatMulSequences ---\n");
	int input_dim = 3;
	Model* m  = new Model(); // argument is input_dim of model

	// I am not sure that batchSize and nb_batch are the same thing
	int nb_batch = 4;
	m->setBatchSize(nb_batch);
    int seq_len = 5;
	m->setSeqLen(seq_len);

	assert(m->getBatchSize() == nb_batch);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input  = new InputLayer(2, "input_layer");
	Layer* dense  = new DenseLayer(3, "dense");
	Layer* out    = new OutLayer(1, "out");

	m->add(0,      input);
	m->add(input,  dense);
	m->add(dense,  out);

	dense->setActivation(new Identity());
	input->setActivation(new Identity());
	out->setActivation(new Identity());

	m->addInputLayer(input);
	m->addOutputLayer(dense);  // Notice: I did not specify "out" as output layer
	
	Connection* conn = input->next[0].second;
	WEIGHT& wght = conn->getWeight();
	m->printSummary();
	conn->printSummary("connection");
	wght.print("weight");

	VF2D_F xf, yf, exact;
	testData(*m, xf, yf, exact);
	printf("xf batch: %d\n", xf.n_rows);

	VF2D_F prod;
	U::createMat(prod, nb_batch, wght.n_rows, seq_len); 

	int seq = 2;
	prod(3).col(1) = wght * xf(0).col(2);
	// I can assign each batch. But not each sequence, unless memory is 
	// preallocated (I think)
	prod.print("prod");
	//U::matmul(wght, xf);
	exit(0);

	U::print(wght, "wght");
	U::print(xf, "xf");
	U::print(prod, "prod");

	xf.print("xf");
	prod.print("prod=wght*xf");
	U::print(prod, "prod");
	exit(0);
	//xf[0].col(0) = prod[0].col(0);
	//printf("xf batch: %d\n", xf.n_rows);
}
//----------------------------------------------------------------------
int main() 
{
	VF2D_F a;
	printf("sizeof(VF2D_F)= %d\n", sizeof(a));
	a.set_size(10);
	a[0] = VF2D(100,100);
	printf("sizeof(VF2D_F)(10)(100,100)= %d\n", sizeof(a));

	VF2D b;
	printf("sizeof(VF2D)= %d\n", sizeof(b));
	VF2D c(100,100);
	printf("sizeof(VF2D(100,100))= %d\n", sizeof(c));
	b.set_size(100,100);
	printf("sizeof(VF2D(100,100))= %d\n", sizeof(b));
}
