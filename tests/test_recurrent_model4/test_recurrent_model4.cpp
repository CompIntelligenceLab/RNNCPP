#include "../common.h"


void testRecurrentModel4(int nb_batch=1)
{
	printf("\n\n\n");
	printf("=============== BEGIN test_recurrent_model4  =======================\n");

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
		WEIGHT& w_1 = conn->getWeight();
		w_1(0,0) = w1;
		conn->computeWeightTranspose();
	}

	{
		conn = m->getConnection(d1, d2);
		WEIGHT& w_12 = conn->getWeight();
		w_12(0,0) = w12;
		conn->computeWeightTranspose();
	}
	
	{
		conn = d1->getConnection();
		WEIGHT& w_11 = conn->getWeight();
		w_11(0,0) = w11;
		conn->computeWeightTranspose();
	}
	
	{
		conn = d2->getConnection();
		WEIGHT& w_22 = conn->getWeight();
		w_22(0,0) = w22;
		conn->computeWeightTranspose();
	}


	//================================
	float inc = .001;
	runTest(m, inc, xf, exact);
	exit(0);
}
//----------------------------------------------------------------------
int main()
{
	testRecurrentModel4(1);
}
