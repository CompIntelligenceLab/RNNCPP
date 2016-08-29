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
void testRecurrentModel1(int nb_batch);
void testRecurrentModel2(int nb_batch);
WEIGHT dLdw(1,1);

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

	VF2D_F pred = m->predictViaConnections(xf);

	#if 0
	pred.print("predicted value");
	exact_prediction.print("exact predicted");
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
	printf("Analytical dLdw: = %f\n", dLdw(0));
	printf("F-D  derivative: = %f\n", fd_dLdw(0));
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
		VF2D_F pred_n = m->predictViaConnections(xf);

		WEIGHT& wm = con.getWeight(); 
		wm(rr,cc) -= (2.*inc);
		VF2D_F pred_p = m->predictViaConnections(xf);

		LOSS loss_p = (*mse)(exact, pred_p);
		LOSS loss_n = (*mse)(exact, pred_n);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdw(rr, cc) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*inc);
	}}
	return dLdw;
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
void testRecurrentModel2(int nb_batch=1)
{
	printf("\n --- testRecurrentModel2 ---\n");

	//================================
	Model* m  = new Model(); // argument is input_dim of model
	m->setSeqLen(2); // runs (but who knows whether correct) with seq_len > 1

	// I am not sure that batchSize and nb_batch are the same thing
	m->setBatchSize(nb_batch);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(1, "input_layer");
	Layer* d1 = new RecurrentLayer(1, "rdense");
	Layer* d2 = new RecurrentLayer(1, "rdense");
	Layer* out   = new OutLayer(1, "out");  // Dimension of out_layer must be 1.
	                                       // Automate this at a later time

	m->add(0,     input);
	m->add(input, d1);
	m->add(d1, d2);

	input->setActivation(new Identity());
	d1->setActivation(new Identity());
	d2->setActivation(new Identity());

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	m->printSummary();
	m->connectionOrderClean(); // no print statements
	//===========================================
/***
Check the sequences: prediction and back prop.

1) dimension = 1, identity activation functions
   seq=2

 l=0    l=1    l=2
  In --> d1 --> d2 --> loss0    (t=0)
         |      |
         |      |
         v      v
  In --> d1 --> d2 --> loss1    (t=1)


Inputs to nodes: z(l,t), a(l,t-1)
Output to nodes: a(l,t)
Weights: In -- d1 : w1
Weights: d1 -- d2 : w12
Weights: d1 -- d1 : w11
Weights: d2 -- d2 : w22
d1: l=1
d2: l=2
exact(t): exact results at time t
loss0(a(2,0), exact(0))
loss1(a(2,1), exact(1))
Input at t=0: x0
Input at t=1: x1

Loss = L = loss0 + loss1
Forward: 
 a(1,-1) = 0, z(1,-1) = w11 * a(1,-1)
 a(2,-1) = 0, z(1,-1) = w22 * a(2,-1)
 a(1,0) = z(1,0) = w1*x0     + w12 * z(1,-1)
 a(2,0) = z(2,0) = w2*z(1,0) + w12 * z(2,-1)
 -------
 z(1,0) = w11 * a(1,0)
 z(2,0) = w22 * a(2,0)
 a(1,1) = z(2,1) = w1*x1     + w12 * z(1,0)
 a(2,1) = z(2,1) = w2*z(1,1) + w12 * z(2,0)
***/

	float w1       = .4;
	float w12      = .5;
	float w11      = .6;
	float w22      = .7;
	float x0       = .45;
	float x1       = .75;
	float ex0      = .75; // exact value
	float ex1      = .85; // exact value
	int seq_len    = 2;
	int input_dim  = 1;
	int nb_layers  = 2;  // in addition to input
	VF2D a(nb_layers+1, seq_len); // assume al dimensions = 1
	VF2D z(nb_layers+1, seq_len); // assume al dimensions = 1

	z(0,0) = a(0,0) = x0;
	z(0,1) = a(0,1) = x1;
	a(1,0) = z(1,0) = w1  * z(0,0);
	a(2,0) = z(2,0) = w12 * z(1,0);


	//a(1,0) = w1  * a(0,0);
	//a(2,0) = w12 * a(1,0);

	z(1,0) = w11 * a(1,0);
	z(2,0) = w22 * a(2,0);
	a(1,1) = z(1,1) = w1  * z(0,1) + z(1,0);
	a(2,1) = z(2,1) = w12 * z(1,1) + z(2,0);

	//a(1,1) = w1  * a(0,1) + w11 * a(1,0)
	//a(2,1) = w12 * a(1,1) + w22 * a(2,0)

	// loss = loss0 + loss1
	//      = (a(2,0)-ex0)^2 + (a(2,1)-ex1)^2
	//      = (w12*a(1,0)-ex0)^2 + (w12*a(1,1)+w22*a(2,0)-ex1)^2
	//      = (w12*w1*a(0,0)-ex0)^2 + (w12*(w1*a(0,1)+w11*a(1,0)) + w22*w12*w1*x0 -ex1)^2
	//      = (w12*w1*x0-ex0)^2 + (w12*(w1*x1+w11*w1*x0) + w22*w12*w1*x0 -ex1)^2
	//      = (l1-ex0)^2 + (l2+l3-ex1)^2
	//
	// d(loss)/dw1  = 2*(l1-ex0)*w12*x0 + 2*(l2+l3-ex1)*(w12*x1) + 2*(l2+l3-ex1)*w22*w12*x0
	// d(loss)/dw12 = 2*(l1-ex0)*w1*x0 + 2*(l2+l3-ex1)*(w1*x1+w11*w1*x0) + 2*(l2+l3-ex1)*w22*w1*x0
	// d(loss)/dw11 = 2*(l2+l3-ex1)*(w12*w1*x0) 
	// d(loss)/dw22 = 2*(l2+l3-ex1)*(w12*w1*x0) 
	//
	float L0 = 2.*(a(2,0)-ex0);
	float L1 = 2.*(a(2,1)-ex1);
	float dldw1  = L0*w12*x0 + L1*(w12*x1+w12*w11*x0+w22*w12*x0); // CORRECT
	float dldw12 = L0*w1*x0 + L1*(w1*x1+w11*w1*x0 + w22*w1*x0); // CORRECT
	float dldw11 = L1*(w12*w1*x0); // CORRECT
	float dldw22 = L1*(w12*w1*x0); // CORRECT

	//a11 = w1 *a01 + w11*a10;
	//a21 = w12*a11 + w22*a20;
	//a10 = w1*a00
	//a20 = w1*w11*a10;

	float da11da01 = w1;
	float da11da10 = w11;
	float da21da11 = w12;
	float da21da20 = w22;
	float da10da00 = w1;
	float da20da10 = w1*w11;

	printf("dLda20= %f\n", L0);
	printf("dLda21= %f\n", L1);
	printf("da11da01= %f\n", w1);
	printf("da11da10= %f\n", w11);
	printf("da21da11= %f\n", w12);
	printf("da21da20= %f\n", w22);
	printf("da10da00= %f\n", w1);
	printf("da20da10= %f\n", w1*w11);
	printf("\n\n");

	float dLda20 = L0;
	float dLda21 = L1;
	float dLda11 = L1 * da21da11;
	float dLda10 = L0*w1*w11;
	float dLda01 = w1*w12*L1;
	float dLda00 = w1*(L1*w11*w12 + L0*w1*w11);
	//float dLda00 = da10da00*(L1*da11da10*da21d11 + L0*da20da10) = w1*(L1*w11*w12 + L0*w1*w11);

	printf("Calculation of weight derivatives by hand\n");
	printf("dldw1= %f\n", dldw1);
	printf("dldw12= %f\n", dldw12);
	printf("dldw11= %f\n", dldw11);
	printf("dldw22= %f\n", dldw22);
	printf("\n");

	printf("dLda by hand\n");
	printf("dLda20= %f\n", L0);
	printf("dLda21= %f\n", L1);
	printf("dLda11= %f\n", dLda11);
	printf("dLda10= %f\n", dLda10);
	printf("dLda01= %f\n", dLda01);
	printf("dLda00= %f\n", dLda00);


	printf("a(1,1)= %f,  a(2,1)= %f\n", a(1,1), a(2,1));

	float loss0 = (ex0-a(2,0))*(ex0-a(2,0));  // same as predict routine
	float loss1 = (ex1-a(2,1))*(ex1-a(2,1));  // DIFFERENT FROM PREDICT ROUTINE
	// Is error hand-solution or in predict? 
	printf("loss0= %f, loss1= %f\n", loss0, loss1);

	int output_dim = m->getOutputLayers()[0]->getOutputDim();

	VF2D_F xf(nb_batch), exact(nb_batch);
	for (int b=0; b < nb_batch; b++) {
		xf(b) = VF2D(input_dim, seq_len);
		for (int i=0; i < input_dim; i++) {
			xf(b)(i,0) = x0; 
			xf(b)(i,1) = x1;
		}
		exact(b) = VF2D(output_dim, seq_len);
		for (int i=0; i < output_dim; i++) {
			exact(b)(i,0) = ex0; 
			exact(b)(i,1) = ex1;
		}
	}

	Connection* conn;
	{
		conn = m->getConnection(input, d1);
		WEIGHT& w1 = conn->getWeight();
		w1(0,0) = 0.4;
	}

	{
		conn = m->getConnection(d1, d2);
		WEIGHT& w12 = conn->getWeight();
		w12(0,0) = .5;
	}
	
	{
		conn = d1->getConnection();
		WEIGHT& w11 = conn->getWeight();
		w11(0,0) = .6;
	}
	
	{
		conn = d2->getConnection();
		WEIGHT& w22 = conn->getWeight();
		w22(0,0) = .7;
	}


	//================================

	VF2D_F yf;
	//testData(*m, xf, yf, exact);

	Layer* outLayer = m->getOutputLayers()[0];
	printf("output_dim = %d\n", output_dim);

	CONNECTIONS connections = m->getConnections();

	VF2D_F pred;

	for (int i=0; i < 1; i++) {
		U::print(xf, "xf");
		pred = m->predictViaConnections(xf);
		U::print(pred, "Prediction: pred");
	}
	Objective* obj = m->getObjective();
	LOSS loss = (*obj)(exact, pred);
	loss.print("loss");
	//exit(0);   // PREDICTIONS ARE WRONG

	#if 0
	U::print(pred, "pred");
	U::print(exact, "exact");
	pred.print("pred");
	exact.print("exact");
	#endif
	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 
	
	printf("\n*** deltas from back propagation ***\n");
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary("Connection (backprop)");
		connections[c]->getDelta().print("delta");
	}
	d1->getConnection()->printSummary("Connection d1-d1");
	d1->getConnection()->getDelta().print("delta(d1,d1)");
	d2->getConnection()->printSummary("Connection d2-d2");
	d2->getConnection()->getDelta().print("delta(d2,d2)");

	//============================================
	// Finite-Difference weights
	float inc = .0001;
	printf("\n*** deltas from finite-difference weight derivative ***\n");
	WEIGHT fd_dLdw;
	// First connection is between 0 and input (does not count)
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary();
		fd_dLdw = weightDerivative(m, *connections[c], inc, xf, exact);
		fd_dLdw.print("weight derivative, spatial connections");
	}
	fd_dLdw = weightDerivative(m, *d1->getConnection(), inc, xf, exact);
	fd_dLdw.print("weight derivative, temporal d1");
	fd_dLdw = weightDerivative(m, *d2->getConnection(), inc, xf, exact);
	fd_dLdw.print("weight derivative, temporal d2");
	exit(0);

	// Exact dL/dw
	//U::print(exact, "exact");
	//U::print(pred, "pred");
	//U::print(xf, "xf");
	VF2D dLdw_analytical = 2.*(exact(0) - pred(0)) % xf(0);
	printf("Analytical dLdw: = %f\n", dLdw(0));
	printf("F-D  derivative: = %f\n", fd_dLdw(0));
	//testRecurrentModel1(1);
	exit(0);
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
void testRecurrentModel1(int nb_batch=1)
{
/***
	Simplest possible network: two nodes with the identity activation. 
	seq_len = 2
	nb_batch = 1
	This allows testing via simple matrix-multiplication

                 w1
	    input ---------> rdense --> loss    (loss is attached to the output layer)
***/
	printf("\n --- testRecurrentModel1 ---\n");
	int input_dim = 1; // predict works with input_dim > 1
	Model* m  = new Model(); // argument is input_dim of model
	m->setSeqLen(2); // runs (but who knows whether correct) with seq_len > 1

	// I am not sure that batchSize and nb_batch are the same thing
	m->setBatchSize(nb_batch);
	assert(m->getBatchSize() == nb_batch);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(1, "input_layer");
	Layer* dense = new RecurrentLayer(1, "rdense");
	Layer* out   = new OutLayer(1, "out");  // Dimension of out_layer must be 1.
	                                       // Automate this at a later time

	m->add(0,     input);
	m->add(input, dense);
	//printf("m->layers size: %d\n", m->getLayers().size()); exit(0);
	//m->add(dense, out); 
	//m->add(dense,  out); // No weights and no recursion. It is only there to connect with the loss function. 
	                     // There is a connection from dense to out. 
						 // Waste of memory if dimensionality is high (even if using identity matrix). 

	dense->setActivation(new Identity());
	input->setActivation(new Identity());
	//out->setActivation(new Identity());

	m->addInputLayer(input);
	m->addOutputLayer(dense);
	runModelRecurrent(m);
}
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
	//exit(0);

	//testMatMulSequences();
	//exit(0);
	


	//testCube();
	//testModel();

	//testModel1a(1);
	//exit(0);

	#if 0
	//testModel1a(2);
	testModel1a(5);
	testFuncModel1();
	testFuncModel2();
	#endif

	//testRecurrentModel1(1);
	testRecurrentModel2(1);
	exit(0);

	testFuncModel3();
	exit(0);
	testModel1();
	testModel2();
	exit(0);

	//testPredict();
	//testObjective();
}
