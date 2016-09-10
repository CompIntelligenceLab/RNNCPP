#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>

void testRecurrentModelBias2(int nb_batch, int seq_len, int layer_size)
{
	printf("\n\n\n");
	printf("=============== BEGIN test_recurrent_model_bias2  =======================\n");

	//================================
	Model* m  = new Model(); // argument is input_dim of model
	int input_dim  = 1;
	m->setSeqLen(seq_len); // runs (but who knows whether correct) with seq_len > 1

	// I am not sure that batchSize and nb_batch are the same thing
	m->setBatchSize(nb_batch);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer* d1    = new RecurrentLayer(layer_size, "rdense");
	Layer* d2    = new RecurrentLayer(layer_size, "rdense");

	m->add(0,     input);
	m->add(input, d1);
	m->add(d1, d2);

	input->setActivation(new Identity());
	d1->setActivation(   new Identity());
	d2->setActivation(   new Identity());

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
	z(0,0) = x0;
	a(0,0) = z(0,0);
	//a(1,-1) is retrieved via  layer->getConnection()->from->getOutputs();
	z(1,0) = w01  * a(0,0); // + w11 * a(1,-1); // a(1,-1) is initially zero
	a(1,0) = z(1,0);

	z(0,1) = x1;
	a(0,1) = z(0,1); // input <-- output  (set input to output)
	z(1,1) = w01  * a(0,1) + w11 * a(1,0);   // for delay of 2, the 2nd term would be: w11 * a(1,-1)
	a(1,1) = z(1,1);

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

	float w01 = .4;
	float w12 = .5;
	float w11 = .6;
	float w22 = .7;
	float bias1 = -.7;  // single layer of size 1 ==> single bias
	float bias2 = -.45; // single layer of size 1 ==> single bias
	//w01       = 1.;
	//w12       = 1.;
	//w11       = 1.;
	//w22       = 1.;
	float x0       = .45;
	float x1       = .75;
	float ex0      = .75; // exact value
	float ex1      = .85; // exact value
	VF2D a(layer_size+1, seq_len); // assume al dimensions = 1
	VF2D z(layer_size+1, seq_len); // assume al dimensions = 1

#if 0
	// t=0
	z(0,0) = x0;
	a(0,0) = z(0,0);
	//a(1,-1) is retrieved via  layer->getConnection()->from->getOutputs();
	z(1,0) = w01 * a(0,0) + bias1; // + w11 * a(1,-1); // a(1,-1) is initially zero
	a(1,0) = (z(1,0));
	a.print("a");
	z(2,0) = w12 * a(1,0) + bias2;
	exit(0);
	a(2,0) = (z(2,0));

	// t=1
	z(0,1) = x1;
	a(0,1) = z(0,1);
	z(1,1) = w01 * a(0,1) + w11 * a(1,0) + bias1;
	a(1,1) = (z(1,1));
	z(2,1) = w12 * a(1,1) + w22 * a(2,0) + bias2;
	a(2,1) = (z(2,1));

	// loss = loss0 + loss1
	//      = (a(2,0)-ex0)^2 + (a(2,1)-ex1)^2
	//      = (w12*a(1,0)-ex0)^2 + (w12*a(1,1)+w22*a(2,0)-ex1)^2
	//      = (w12*w01*a(0,0)-ex0)^2 + (w12*(w01*a(0,1)+w11*a(1,0)) + w22*w12*w01*x0 -ex1)^2
	//      = (w12*w01*x0-ex0)^2 + (w12*(w01*x1+w11*w01*x0) + w22*w12*w01*x0 -ex1)^2
	//      = (l1-ex0)^2 + (l2+l3-ex1)^2
	//
	// d(loss)/dw1  = 2*(l1-ex0)*w12*x0 + 2*(l2+l3-ex1)*(w12*x1) + 2*(l2+l3-ex1)*w22*w12*x0
	// d(loss)/dw12 = 2*(l1-ex0)*w01*x0 + 2*(l2+l3-ex1)*(w01*x1+w11*w01*x0) + 2*(l2+l3-ex1)*w22*w01*x0
	// d(loss)/dw11 = 2*(l2+l3-ex1)*(w12*w01*x0) 
	// d(loss)/dw22 = 2*(l2+l3-ex1)*(w12*w01*x0) 
	//

	float loss0 = (a(2,0)-ex0)*(a(2,0)-ex0);  // same as predict routine
	float loss1 = (a(2,1)-ex1)*(a(2,1)-ex1);  // same as predict routine
	float L0 = 2.*(a(2,0)-ex0);
	float L1 = 2.*(a(2,1)-ex1);
	float dldw1  = L0*w12*x0 + L1*(w12*x1+w12*w11*x0+w22*w12*x0); // CORRECT
	float dldw12 = L0*w01*x0 + L1*(w01*x1+w11*w01*x0 + w22*w01*x0); // CORRECT
	float dldw11 = L1*(w12*w01*x0); // CORRECT
	float dldw22 = L1*(w12*w01*x0); // CORRECT


	#if 0
	printf("\n ============== Layer Outputs =======================\n");
	printf("a00, a01= %f, %f\n", a(0,0), a(0,1));
	printf("z10, z11= %f, %f\n", z(1,0), z(1,1));
	printf("a10, a11= %f, %f\n", a(1,0), a(1,1));
	printf("z20, z21= %f, %f\n", z(2,0), z(2,1));
	printf("a20, a21= %f, %f\n", a(2,0), a(2,1));
	#endif

	//a11 = w01 *a01 + w11*a10;
	//a21 = w12*a11 + w22*a20;
	//a10 = w01*a00
	//a20 = w01*w11*a10;

	float da11da01 = w01;
	float da11da10 = w11;
	float da21da11 = w12;
	float da21da20 = w22;
	float da10da00 = w01;
	float da20da10 = w01*w11;

	#if 0
	printf("dLda20= %f\n", L0);
	printf("dLda21= %f\n", L1);
	printf("da11da01= %f\n", w01);
	printf("da11da10= %f\n", w11);
	printf("da21da11= %f\n", w12);
	printf("da21da20= %f\n", w22);
	printf("da10da00= %f\n", w01);
	printf("da20da10= %f\n", w01*w11);
	printf("\n\n");
	#endif

	float dLda20 = L0;
	float dLda21 = L1;
	float dLda11 = L1 * da21da11;
	float dLda10 = L0*w01*w11;
	float dLda01 = w01*w12*L1;
	float dLda00 = w01*(L1*w11*w12 + L0*w01*w11);
	//float dLda00 = da10da00*(L1*da11da10*da21d11 + L0*da20da10) = w01*(L1*w11*w12 + L0*w01*w11);

	#if 0
	printf("Calculation of weight derivatives by hand\n");
	printf("loss0= %f, loss1= %f\n", loss0, loss1);
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
	#endif
#endif

	int output_dim = m->getOutputLayers()[0]->getOutputDim();

	VF2D_F xf(nb_batch), exact(nb_batch);
	for (int b=0; b < nb_batch; b++) {
		xf(b) = VF2D(input_dim, seq_len);
		for (int i=0; i < input_dim; i++) {
			for (int s=0; s < seq_len; s++) {
				xf(b)(i,s) = x0; 
			}
		}
		exact(b) = VF2D(output_dim, seq_len);
		for (int i=0; i < output_dim; i++) {
			for (int s=0; s < seq_len; s++) {
				exact(b)(i,s) = ex0; 
			}
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

	{
		BIAS& bias_1 = d1->getBias();
		bias_1(0) = bias1;
	}

	{
		BIAS& bias_2 = d2->getBias();
		bias_2(0) = bias2;
	}


	//================================
	float inc = 0.001;
	runTest(m, inc, xf, exact); exit(0);

	predictAndBackProp(m, xf, exact);

	exit(0);
}
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len

    int nb_batch = 1;
    int layer_size = 1;
    int seq_len = 1;

	argv++; 
	argc--; 

	while (argc > 1) {
		std::string arg = std::string(argv[0]);
		if (arg == "-b") {
			nb_batch = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-s") {
			seq_len = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-l") {
			layer_size = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-h") {
			printf("Argument usage: \n");
			printf("  -b <nb_batch>  -s <seq_len> -l <layer_size>\n");
		}
	}

	testRecurrentModelBias2(nb_batch, seq_len, layer_size);
}
