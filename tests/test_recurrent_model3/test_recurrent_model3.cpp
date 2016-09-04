#include "../common.h"

//----------------------------------------------------------------------
void testRecurrentModel3(int nb_batch=1)
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
	Layer* d1 = new DenseLayer(1, "rdense");
	Layer* d2 = new DenseLayer(1, "rdense");
	Layer* out   = new OutLayer(1, "out");  // Dimension of out_layer must be 1.
	                                       // Automate this at a later time
	m->add(0,     input);
	m->add(input, d1);
	m->add(d1,    d2);

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
   (no links between times)
  In --> d1 --> d2 --> loss1    (t=1)

Inputs to nodes: z(l,t), a(l,t-1)
Output to nodes: a(l,t)
Weights: In -- d1 : w01
Weights: d1 -- d2 : w12
d1: l=1 (layer 1)
d2: l=2
exact(t): exact results at time t
loss0(a(2,0), exact(0))
loss1(a(2,1), exact(1))
Input at t=0: x0
Input at t=1: x1

Loss = L = loss0 + loss1
Forward: 
 z(1,0) = w01*x0   
 z(2,0) = w12*a(1,0)
 a(1,0) = z(1,0)
 a(2,0) = z(2,0)
 -------
 z(1,1) = w01*x1    (2nd arg is time; 1st argument is layer)
 z(2,1) = w12*a(1,1) 
 a(1,1) = z(1,1)
 a(1,2) = z(1,2)
***/

// ANALYTICAL DERIVATION when activation functions are the identity

	// set weights to 1 for testing
	float w01      = .4;
	float w12      = .5;
	float x0       = .45;
	float x1       = .75;
	float ex0      = .75; // exact value
	float ex1      = .85; // exact value
	int seq_len    = 2;
	int input_dim  = 1;
	int nb_layers  = 2;  // in addition to input

	// nb_layers+1 accommodates nb_layers hidden layers and an input layer
	VF2D a(nb_layers+1, seq_len); // assume all dimensions = 1
	VF2D z(nb_layers+1, seq_len); // assume all dimensions = 1

	a(0,0) = x0;
	a(0,1) = x1;
	z(0,0) = a(0,0);
	z(0,1) = a(0,1);

	z(1,0) = w01  * a(0,0);
	z(1,1) = w01  * a(0,1);
	a(1,0) = z(1,0);
	a(1,1) = z(1,1);

	z(2,0) = w12  * a(1,0);
	z(2,1) = w12  * a(1,1);
	a(2,0) = z(2,0);
	a(2,1) = z(2,1);

	// loss = loss0 + loss1
	//      = (a(2,0)-ex0)^2 + (a(2,1)-ex1)^2
	//      = (w12*a(1,0)-ex0)^2 + (w12*a(1,1)-ex1)^2
	//      = (w12*w01*x0 -ex0)^2 + (w12*w01*x1 -ex1)^2
	//      = (l0-ex0)^2 + (l1-ex1)^2
	//
	// d(loss)/dw01 = 2*(l0-ex0)*w12*x0 + 2*(l1-ex1)*(w12*x1) 
	// d(loss)/dw12 = 2*(l0-ex0)*w01*x0 + 2*(l1-ex1)*(w01*x1) 

	float loss0 = (ex0-a(2,0))*(ex0-a(2,0));  // same as predict routine
	float loss1 = (ex1-a(2,1))*(ex1-a(2,1));  // DIFFERENT FROM PREDICT ROUTINE
	float loss_tot  = loss0 + loss1;   // total loss for a sequence of length 2

	float     L0 = 2.*(a(2,0)-ex0);
	float     L1 = 2.*(a(2,1)-ex1);
	float dlda20 = L0;
	float dlda21 = L1;
	float dlda10 = dlda20 * w12;
	float dlda11 = dlda21 * w12;
	float dlda00 = dlda10 * w01;
	float dlda01 = dlda11 * w01;

	printf("\n ============== Layer Outputs =======================\n");
	printf("a00= %f\n", a(0,0));
	printf("a10= %f\n", a(1,0));
	printf("a20= %f\n", a(2,0));
	printf("a01= %f\n", a(0,1));
	printf("a11= %f\n", a(1,1));
	printf("a21= %f\n", a(2,1));

	printf("dlda20= %f, dlda21= %f\n", dlda20, dlda21);
	printf("dlda10= %f, dlda11= %f\n", dlda10, dlda11);
	printf("dlda00= %f, dlda01= %f\n", dlda00, dlda01);

	// Derivative of loss with respect to weights
	// dL/dw01 = sum_t dL/da1t * da1t/dw01  = sum_t dL/da1t * (f'=1) * a0t
	//         = dL/da10 * a00 + dL/da11 * a01
	//         = dlda10  * a00 + dlda11  * a01
	// dL/dw12 = sum_t dL/da2t * da2t/dw12  = sum_t dL/da2t * (f'=1) * a1t
	//         = dL/da20 * a10 + dL/da21 * a11
	//         = dlda20  * a10 + dlda21  * a11
    
	float dldw01 = dlda10*a(0,0) + dlda11*a(0,1);
	float dldw12 = dlda20*a(1,0) + dlda21*a(1,1);

	printf(".... Calculation of weight derivatives by hand\n");
	printf("loss0= %f, loss1= %f\n", loss0, loss1);
	printf("total loss: %f\n", loss_tot);
	printf("dldw01= %f\n", dldw01);
	printf("dldw12= %f\n", dldw12);
	printf("\n");


	printf(" ================== END dL/da's =========================\n\n");



	//============================================

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
		WEIGHT& w_01 = conn->getWeight();
		w_01(0,0) = w01;
		conn->computeWeightTranspose();
	}

	{
		conn = m->getConnection(d1, d2);
		WEIGHT& w_12 = conn->getWeight();
		w_12(0,0) = w12;
		conn->computeWeightTranspose();
	}

	// ================  BEGIN F-D weight derivatives ======================
	float inc = .0001;
	printf("\n*** deltas from finite-difference weight derivative ***\n");
	WEIGHT fd_dLdw;
	// First connection is between 0 and input (does not count)
	CONNECTIONS connections = m->getConnections();
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary();
		fd_dLdw = weightDerivative(m, *connections[c], inc, xf, exact);
		fd_dLdw.print("weight derivative, spatial connections, recurrent3");
	}
	// ================  END F-D weight derivatives ======================
	

	//================================

	VF2D_F yf;
	Layer* outLayer = m->getOutputLayers()[0];
	printf("output_dim = %d\n", output_dim);

	VF2D_F pred;

	for (int i=0; i < 1; i++) {
		U::print(xf, "xf");
		pred = m->predictViaConnectionsBias(xf);
		U::print(xf, "+++++++++++++ Prediction: xf");
		U::print(pred, "+++++++++++++ Prediction: pred");
		xf.print("xf");
		pred.print("pred");
	}
	Objective* obj = m->getObjective();
	const LOSS& loss = (*obj)(exact, pred);
	loss.print("prediction loss, ");

	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 
	

	printf("\n*** deltas from back propagation ***\n");
	for (int c=1; c < connections.size(); c++) {
		connections[c]->printSummary("Connection (backprop), ");
		connections[c]->getDelta().print("delta");
	}

	VF2D dLdw_analytical = 2.*(exact(0) - pred(0)) % xf(0);
	printf("Analytical dLdw: = %f\n", dLdw(0));
	printf("F-D  derivative: = %f\n", fd_dLdw(0));
	exit(0);

	// Three checks: 
	// 1) analytical calculation (using recursion to compute derivatives of loss wrt layer output. 
	// 2) Finite-Difference derivative
	// 3) Actual back-propagation
	// If all three give the same answer to within some tolerance, it is pretty certain that the results are correct (although not a proof)
}
//----------------------------------------------------------------------
int main()
{
	testRecurrentModel3(1);
}
