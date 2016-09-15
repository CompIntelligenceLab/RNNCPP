#include "../common.h"

/***
Single recurrent node. Test back-propagation. 
Built based on a working version of test_current_model3.cpp
***/
//----------------------------------------------------------------------
void testRecurrentModel5(int nb_batch=1)
{
	//printf("\n\n\n");
	//printf("=============== BEGIN test_recurrent_model5  =======================\n");

	//================================
	Model* m  = new Model(); // argument is input_dim of model
	m->setSeqLen(2); // runs (but who knows whether correct) with seq_len > 1

	// I am not sure that batchSize and nb_batch are the same thing
	m->setBatchSize(nb_batch);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(1, "input_layer");
	Layer* d1 = new RecurrentLayer(1, "recurrent");
	Layer* out   = new OutLayer(1, "out");  // Dimension of out_layer must be 1.
	                                       // Automate this at a later time
	m->add(0,     input);
	m->add(input, d1);

	input->setActivation(new Identity());
	d1->setActivation(new Identity());

	m->addInputLayer(input);
	m->addOutputLayer(d1);

	//m->printSummary();
	m->connectionOrderClean(); // no print statements
	//===========================================
/***
Check the sequences: prediction and back prop.

1) dimension = 1, identity activation functions
   seq=2

 l=0    l=1    l=2
  In --> d1 --> loss0    (t=0)
         |
         |
         v
  In --> d1 --> loss1    (t=1)

Inputs to nodes: z(l,t), a(l,t-1)
Output to nodes: a(l,t)
Weights: In -- d1 : w01
d1: l=1 (layer 1)
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
	REAL w01      = .4;
	REAL w11      = .6;  // recurrence weights for layer 1
	//w01      = 1.;
	//w11      = 1.;  // recurrence weights for layer 1
	REAL x0       = .45;
	REAL x1       = .65;
	REAL ex0      = .75; // exact value
	REAL ex1      = .85; // exact value
	int seq_len    = 2;
	int input_dim  = 1;
	int nb_layers  = 1;  // in addition to input

	// nb_layers+1 accommodates nb_layers hidden layers and an input layer
	VF2D a(nb_layers+1, seq_len); // assume all dimensions = 1
	VF2D z(nb_layers+1, seq_len); // assume all dimensions = 1

	z(0,0) = x0;
	a(0,0) = z(0,0);
	//a(1,-1) is retrieved via  layer->getConnection()->from->getOutputs();
	z(1,0) = w01  * a(0,0); // + w11 * a(1,-1); // a(1,-1) is initially zero
	a(1,0) = z(1,0);

	z(0,1) = x1;
	a(0,1) = z(0,1); // input <-- output  (set input to output)
	z(1,1) = w01  * a(0,1) + w11 * a(1,0);   // for delay of 2, the 2nd term would be: w11 * a(1,-1)
	a(1,1) = z(1,1);

	// loss = loss0 + loss1
	//      = (a(1,0)-ex0)^2  + (a(1,1)-ex1)^2
	//      = (w01*a00-ex0)^2 + (w01*a01+w11*a10-ex1)^2
	//      = (w01*a00-ex0)^2 + (w01*a01+w11*w01*a00-ex1)^2
	//      = (w01*x0-ex0)^2  + (w01*x1+w11*w01*x0-ex1)^2
	//      = (l0-ex0)^2      + (l1-ex1)^2
	//
	// Direct calculation of weight derivatives
	// d(loss)/dw01 = 2*(l0-ex0)*x0 + 2*(l1-ex1)*x1 + 2*(l1-ex1)*w11*x0;
	// d(loss)/dw11 = 2*(l1-ex1)*(w01*x0) 

	REAL loss0 = (ex0-a(1,0))*(ex0-a(1,0));  // same as predict routine
	REAL loss1 = (ex1-a(1,1))*(ex1-a(1,1));  // DIFFERENT FROM PREDICT ROUTINE
	REAL loss_tot = loss0 + loss1;   // total loss for a sequence of length 2
	REAL     L0 = 2.*(a(1,0)-ex0); // dlda10
	REAL     L1 = 2.*(a(1,1)-ex1); // dlda11

	REAL dldw01_direct = L0 * x0  +  L1 * (x1+w11*x0);
	REAL dldw11_direct = L1 * w01 * x0;

	// dlda00 = sum_t dla1t * da1t/da00 
	//   = L0 * w01 + L1 * da11/da10 * da10/da00
	//   = L0 * w01 + L1 * w11 * w01;
	// dlda01 = sum_t dla1t * da1t/da01 
	//   = L0 * da10/da01 + L1 * da11/da01
	//   = L0 * da10/da01 + L1 * da11/da01
	//   = L1 * da11/da10 * da10/da00
	//   = L1 * w11 * w01;
	REAL dlda10 = L0;
	REAL dlda11 = L1;
	REAL dlda00 = L0 * w01  +  L1 * w11 * w01;
	REAL dlda01 = L1 * w11 * w01;

	#if 0
	printf("\n ============== Layer Outputs =======================\n");
	printf("a00= %f\n", a(0,0));
	printf("a10= %f\n", a(1,0));
	printf("a01= %f\n", a(0,1));
	printf("a11= %f\n", a(1,1));

	printf(".... Calculation of weight derivatives by hand\n");
	printf("loss0= %f, loss1= %f\n", loss0, loss1);
	printf("total loss: %f\n", loss_tot);
	printf("dldw01_direct= %f\n", dldw01_direct);
	printf("dldw11_direct= %f\n", dldw11_direct);

	printf("\n");
	printf("dlda00, dlda01= %f, %f\n", dlda00, dlda01);
	printf("dlda10, dlda11= %f, %f\n", dlda10, dlda11);
	#endif

	// Derivative of loss with respect to weights
	// dL/dw01 = sum_t dL/da1t * da1t/dw01  = sum_t dL/da1t * (f'=1) * a0t
	//         = dL/da10 * a00 + dL/da11 * a01
	//         = dlda10  * a00 + dlda11  * a01
	// dL/dw11 = sum_t dL/da2t * da2t/dw11  = sum_t dL/da2t * (f'=1) * a1t
	//         = dL/da20 * a10 + dL/da21 * a11
	//         = dlda20  * a10 + dlda21  * a11
    
	#if 0
	REAL dldw01 = dlda10*x0 + dlda11*x1;
	REAL dldw11 = dlda20*a(1,0) + dlda21*a(1,1);

	printf(".... Calculation of weight derivatives by hand\n");
	printf("dldw01= %f\n", dldw01);
	printf("dldw12= %f\n", dldw12);
	printf("\n");
	#endif


	printf(" ================== END dL/da's =========================\n\n");


	//============================================

	int output_dim = m->getOutputLayers()[0]->getOutputDim();
	VF2D_F xf(nb_batch), exact(nb_batch);

	for (int b=0; b < nb_batch; b++) {
		xf(b) = VF2D(input_dim, seq_len);
		for (int i=0; i < input_dim; i++) {
			xf(b)(i,0) = x0;  // later, use different values in different dimensions
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
		conn = d1->getConnection(); // recurrent layer
		WEIGHT& w_11 = conn->getWeight();
		w_11(0,0) = w11;
		conn->computeWeightTranspose();
	}

	//=======================================
	REAL inc = .0001;
	runTest(m, inc, xf, exact);
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
	testRecurrentModel5(4);
}
