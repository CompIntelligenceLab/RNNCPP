#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>

REAL derivLoss(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws);
REAL predict(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws);

//----------------------------------------------------------------------
//void testRecurrentModelBias1(Model* m, int layer_size, int is_recurrent, Activation* activation, REAL inc) 
void testDiffEq1(Model* m)
{
	//testRecurrentModelBias1(m, layer_size, is_recurrent, activation, inc);
	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;
	REAL inc = m->inc;
	int nb_layers = m->nb_layers;


	printf("\n\n\n");
	printf("=============== BEGIN test_recurrent_model_bias2  =======================\n");

	//================================
	int seq_len = m->getSeqLen();
	int nb_batch = m->getBatchSize();
	int input_dim  = 1;

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer *d1;
	std::vector<Layer*> internal_layers;

	is_recurrent = 0;

	for (int i=0; i < nb_layers; i++) {
		if (is_recurrent) {
			d1    = new RecurrentLayer(layer_size, "rdense");
		} else {
			d1    = new DenseLayer(layer_size, "rdense");
		}
		internal_layers.push_back(d1);
	}

	m->add(0,     input);
	input->setActivation(new Identity()); 

	m->activations[0]->setParam(0, .8);
    internal_layers[0]->setActivation(m->activations[0]); 
	m->add(input, internal_layers[0]);

	for (int i=1; i < internal_layers.size(); i++) {
		m->activations[i]->setParam(0, i*.1);
    	internal_layers[i]->setActivation(m->activations[i]); 
		m->add(internal_layers[i-1], internal_layers[i]);
	}

	// input should always be identity activation

	m->addInputLayer(input);
	m->addOutputLayer(internal_layers[nb_layers-1]);

	m->printSummary();
	m->connectionOrderClean(); // no print statements

	m->initializeWeights();

	// Initialize xf and exact
	VF2D_F xf, exact;
	int input_size = input->getLayerSize();
	U::createMat(xf, nb_batch, input_size, seq_len);
	U::createMat(exact, nb_batch, layer_size, seq_len);
	U::print(xf, "xf"); //exit(0);
	U::print(exact, "exact"); //exit(0);
	//exit(0);

	for (int b=0; b < xf.n_rows; b++) {
		xf[b] = arma::randu<VF2D>(input_size, seq_len); //size(xf[b]));
		exact[b] = arma::randu<VF2D>(layer_size, seq_len); //size(xf[b]));
	}
	xf.print("xf"); 
	exact.print("exact"); 
	U::print(xf, "xf"); //exit(0);
	U::print(exact, "exact"); //exit(0);
	//exit(0);

	// SOME KIND OF MATRIX INCOMPATIBILITY. That is because exact has the wrong dimensions. 
	runTest(m, inc, xf, exact);
	printf("gordon\n");
	exit(0);

	//===========================================

	REAL w01 = .4;
	REAL w11 = .6;
	REAL bias1 = 0.; //-.7;  // single layer of size 1 ==> single bias
	//w01       = 1.;
	//w11       = 1.;
	//w22       = 1.;
	REAL x0       = .45;
	REAL x1       = .75;
	REAL ex0      = .75; // exact value
	REAL ex1      = .85; // exact value
	VF2D a(layer_size+1, seq_len); // assume al dimensions = 1
	VF2D z(layer_size+1, seq_len); // assume al dimensions = 1

	int output_dim = m->getOutputLayers()[0]->getOutputDim();

	//VF2D_F xf(nb_batch), exact(nb_batch);
	for (int b=0; b < nb_batch; b++) {
		xf(b) = VF2D(input_dim, seq_len);
		for (int i=0; i < input_dim; i++) {
			for (int s=0; s < seq_len; s++) {
				if (s == 0) {
					xf(b)(i,s) = x0; 
				} else {
					xf(b)(i,s) = 0.;  // compare with analytics
				}
			}
		}
		printf("*** xf:   ONLY HAS INITIAL COMPONENT at t=0, else ZERO\n");

		exact(b) = VF2D(output_dim, seq_len);
		for (int i=0; i < output_dim; i++) {
			for (int s=0; s < seq_len; s++) {
				exact(b)(i,s) = ex0; 
			}
		}
	}


	// Set to 1 to run the test with diagonal recurrent matrix
	//diagRecurrenceTest(m, input, d1, xf, exact, inc);
	exit(0);

	#if 1
	{
		BIAS& bias_1 = d1->getBias();
		bias_1(0) = 0.; //bias1;
	}
	#endif

	//================================
	std::vector<WEIGHT> wss;
	wss = runTest(m, inc, xf, exact); 
	exit(0);

	predictAndBackProp(m, xf, exact);
	exit(0);
}
//----------------------------------------------------------------------
REAL predict(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws)
{
	VF1D x = z0;
	printf("iteration: 0, x= %14.7f\n", x(0));  //x.print("x: ");

	for (int i=0; i < k; i++) {
		x = w11*x;
		printf("iteration: %d, x= %14.7f\n", i+1, x(0));  //x.print("x: ");
	}
}
//----------------------------------------------------------------------
REAL derivLoss(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws)
{
	VF2D x(1,1);
	#if 0
	printf("INSIDE derivloss\n");
	printf("alpha= %f\n", alpha);
	printf("m0, n0= %d, %d\n", m0, n0);
	printf("w11= %f\n", w11(0,0));
	printf("e= %f\n", e(0));
	printf("z0= %f\n", z0(0));
	printf("k= %d\n", k);
	printf("\n");
	#endif
	REAL alphak = pow(alpha,k);
	VF1D l = 2.*(alphak*z0-e); //.t();
	std::cout.precision(11);
	//std::cout.setf(ios::fixed);
	l.raw_print("\n derivLoss, l.. (loss)");

	REAL Lprime = l[m0] * k * pow(alpha, k-1) * z0[n0];
	//printf("2*(w11*z0-e0)= %f\n",  2.*(w11(0,0)*z0(0)-e(0)));


	//REAL dLdw11 = 2.*(w11(0,0)*z0(0)-e(0)) * z0(0);
	//printf("*****************************\n");
	//printf("analytical dLdw11= %f\n", dLdw11);
	//printf("derivLoss dLdw11= %f\n", Lprime);
	//printf("*****************************\n");
	return Lprime;
}
	//	REAL dLdw11 = 2.*(w11*w01*x0 - e1) * w01*x0;
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

	Model* m = processArguments(argc, argv);
	testDiffEq1(m);
}

