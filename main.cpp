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
#include "dense_layer.h"
#include "lstm_layer.h"
#include "gmm_layer.h"
#include "input_layer.h"
#include "print_utils.h"

using namespace arma;
using namespace std;

VF2D_F testBackprop(Model* m);
void testData(Model& m, VF2D_F& xf, VF2D_F& yf, VF2D_F&);


float weightDerivative(Model* m, Connection& con, float inc, VF2D_F& xf, VF2D_F& exact)
{
	WEIGHT w0 = con.getWeight();
	int rrows = w0.n_rows;
	int ccols = w0.n_cols;
	float dLdw = 0;

	for (int rr=0; rr < rrows; rr++) {
	for (int cc=0; cc < ccols; cc++) {

		WEIGHT& wp = con.getWeight(); 
		wp(rr,cc) += inc;
		VF2D_F pred_n = m->predictComplex(xf);

		WEIGHT& wm = con.getWeight(); 
		wm(rr,cc) -= (2.*inc);
		VF2D_F pred_p = m->predictComplex(xf);

		VF2D_F sub(pred_n.n_rows);
		for (int i=0; i < pred_n.size(); i++) {
			sub(i) = pred_n(i) - pred_p(i);
		}

		//U::print(exact, "exact"); 
		//printf("exact.nrows= %d\n", exact.n_rows); 

		Objective* mse = new MeanSquareError();
		VF1D_F loss_p = (*mse)(exact, pred_p);
		VF1D_F loss_n = (*mse)(exact, pred_n);
		//U::print(loss_p, "loss_p"); exit(0);
		//U::print(loss_n, "loss_n");
		//loss_n.print("loss_n");
		//loss_p.print("loss_p");
		dLdw = (loss_n(0)(0) - loss_p(0)(0)) / (2.*inc);
		con.setWeight(w0);
		//printf("loss: %f, %f\n", loss_n(0)(0), loss_p(0)(0));
		printf("...> Finite-Difference, dLdw(%d, %d)= %f\n", rr, cc, dLdw);
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
    m->setBatchSize(1);
	assert(m->getBatchSize() == 1);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	// Must make sure that input_dim of input layer is the same as model->input_dim
	int input_dim = 1;
	int layer_size = 1;
	Layer* input   = new InputLayer(input_dim, "input_layer");  
	Layer* dense0  = new DenseLayer(layer_size, "dense0");
	Layer* dense1  = new DenseLayer(layer_size, "dense1");
	Layer* dense2  = new DenseLayer(layer_size, "dense2");
	Layer* dense3  = new DenseLayer(layer_size, "dense3");

	m->add(0, input);
	m->add(input, dense0);
	//m->add(dense0, dense1);
	//m->add(dense1, dense2);
	//m->add(dense2, dense3);

	m->addInputLayer(input);
	m->addOutputLayer(dense0);

	m->checkIntegrity();
	m->printSummary();
	//----------

	VF2D_F xf, yf, exact;
	testData(*m, xf, yf, exact);
	exact.print("exact"); // .0470

// xxxxxxxxx

	//xf = testBackprop(m);
	xf.print("xf");

	printf("\n===== PREDICT ===============================================================================================\n");
	VF2D_F pred = m->predictComplexMaybeWorks(xf);  // for testing while Nathan works with predict

	xf.print("xf"); //    0.5328
	pred.print("pred"); //    0.2396  (matches analytical)
	// Output to dens0: tanh(w*xf) = tanh(.4587*.5328) = tanh(.2444) = 0.2396
	// objective: (.239643-.0470)**2 = .037094
	// tanh gradient: (1-.239643^2) = .9426
	WEIGHT w =  m->getConnections()[1]->getWeight();
	m->getConnections()[1]->getWeight().print("weight"); //    0.4587
	(*m->getObjective())(exact, pred).print("objective");  // .0371 (ok)

	// dL/dw = (dL/dz) (dz/da) (da/dw) = 2.*(pred-exact)*tanh'(xf*w) * xf
	//       = 2.*(pred-exact)*(1-(xf*w)**2) * xf
    for (int b=0; b < pred.size(); b++) {
		VF1D dLdw_exact = 2.*(pred(b)-exact(b))*(1.-arma::tanh(xf(b)*w(0,0))%tanh(xf(b)*w(0,0))) * xf(b);
		dLdw_exact.print("==> dLdw_exact");
	}

	printf("\n===== BACK PROPAGATION =================================================================================\n");
	m->backPropagation(exact, pred);

	for (int c=1; c < m->getConnections().size(); c++) {
		Connection* con = m->getConnections()[c];
		WEIGHT delta = con->getDelta();
		delta.print("==> delta"); //   -0.2052
	}
	printf("\n===== END BACK PROPAGATION =================================================================================\n");

	// Test derivative calculations via finite-differences

	printf("\n===== EXACT DERIVATIVES dL/dw ============================================================================\n");

	float inc = .011;
	int rr = 0;
	int cc = 0;

	CONNECTIONS connections = m->getConnections();
	WEIGHT w0 = connections[1]->getWeight(); 
	w0.print("w0");

	// calculate derivative with respect to all elements of weight matri
	float dLdw = weightDerivative(m, *connections[1], inc, xf, exact);
	// dLdw(0, 0)= 0.128712  (weight: 1x1)
	printf("\n===== END EXACT DERIVATIVES dL/dw ============================================================================\n");

}
//----------------------------------------------------------------------
void testFuncModel2()
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

	m->add(input, dense1);
	m->add(input, dense2);
	m->add(dense2, dense3);
	m->add(dense1, dense2);
	m->add(dense3, dense4);

	m->addInputLayer(input);
	m->addOutputLayer(dense4);

	input_dim = input->getInputDim();
	printf("input_dim= %d\n", input_dim);

	m->checkIntegrity();
	m->printSummary();

	testBackprop(m);
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

	input_dim = input->getInputDim();
	printf("input_dim= %d\n", input_dim);

	m->checkIntegrity();
	m->printSummary();

	testBackprop(m);
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

	int output_dim = m.getOutputLayers()[0]->getOutputDim();
	printf("output_dim= %d\n", output_dim);
	int seq_len = 1;

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(input_dim, seq_len); // uniform random numbers
		yf[i].randu(input_dim, seq_len);
		exact[i].randu(output_dim, seq_len);
	}
}
//----------------------------------------------------------------------
VF2D_F testBackprop(Model* m)
{
	int batch_size = m->getBatchSize();
	VF2D_F xf(batch_size);
	VF2D_F yf(batch_size); 
	VF2D_F exact(batch_size);

	//printf("batch_size= %d\n", batch_size);
	Layer* input = m->getInputLayers()[0];
	int input_dim = input->getInputDim();
	//printf("input_dim= %d\n", input_dim);
	//printf("xf.size= %llu", xf.n_rows);

	int output_dim = m->getOutputLayers()[0]->getOutputDim();
	int seq_len = 1;
	//printf("output_dim= %d\n", output_dim);

	for (int i=0; i < xf.size(); i++) {
		xf[i].randu(input_dim, seq_len); // uniform random numbers
		yf[i].randu(input_dim, seq_len);
		exact[i].randu(output_dim, seq_len);
	}

	//printf("   nlayer layer_size: %d\n", m->getLayers()[0]->getLayerSize());
	//printf("   input layer_size: %d\n", input->getLayerSize());

	//xf.print("xf");
	//exact.print("exact");
	
	//printf("\n=====================================\n");
	//printf("\n\n --- Prediciton --- \n\n");
	VF2D_F pred = m->predictComplex(xf);
	//pred.print("funcModel, predict:");
	
	//U::print(pred, "pred");
	//U::print(exact, "exact");

	//m->train(xf);

	for (int i=0; i < 1; i++) {
		m->backPropagationComplex(exact, pred);
		//printf("i= %d\n", i);
	}

	return xf;
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


	//testCube();
	//testModel();
	testFuncModel1();
	//testFuncModel2();
	//testFuncModel3();
	//testModel1();
	//testModel2();
	//testPredict();
	//testObjective();
}
