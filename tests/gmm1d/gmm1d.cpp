#include <math.h>
#include <unordered_map>
#include <string>
#include <cstdlib>
#include <random>
#include <set>

// for discrete distributions
#include <iostream>
#include <map>

#include "globals.h"
#include "../common.h"

std::vector<VF2D> vpi;
std::vector<VF2D> vmu;
std::vector<VF2D> vsig;

REAL dt = .5;

//----------------------------------------------------------------------
void saveGMM1(const VF2D& predict)
{
// First compute softmax of prediction to transform the to probabilities

	// predict[batch][inputs, seq]
	// predict[batch][0:inputs/3, :] ==> amplitudes
	// predict[batch][inputs/3:2*inputs/3, :] ==> means
	// predict[batch][2*inputs/3:3*inputs/3, :] ==> standard deviations

	int seq_len = predict.n_cols;
	int input_dim = predict.n_rows;

	int ia1   = 0;
	int ia2   = input_dim / 3;
	int imu1  = input_dim / 3;
	int imu2  = 2 * input_dim / 3;
	int isig1 = 2 * input_dim / 3;
	int isig2 = 3 * input_dim / 3;

	VF2D pi(predict.rows(ia1,ia2-1));

	// softmax over the dimension index of VF2D (first index)
	REAL mx = arma::max(arma::max(pi));
	for (int s=0; s < seq_len; s++) {
	   pi.col(s) = arma::exp(pi.col(s)-mx);
	   // trick to avoid overflows
	   REAL ssum = 1. / arma::sum(pi.col(s)); // = arma::exp(y[b]);
	   pi.col(s) = pi.col(s) * ssum;  // % is elementwise multiplication (arma)
	}

	// standard deviations: exp(output) => sig

	VF2D sig(predict.rows(isig1, isig2-1));
	sig = arma::exp(sig);
	VF2D mu(predict.rows(imu1, imu2-1));
	vpi.push_back(pi);
	vmu.push_back(mu);
	vsig.push_back(sig);
}
//----------------------------------------------------------------------
VF2D discrete_sample_gmm1d(VF2D predict);

/* implementation of Karpathy's char-rnn. 
   Need on-hot vectors. 

  Probably 65 or so characters + punctuation. 
  Need a translator from characters to hotvec. 

  E.g.  'a' ==> 100000....00000
  E.g.  'b' ==> 010000....00000
    ..........
	    ',' ==> 000......000001

How to transfer weights from model M1 to model M2?
  
     M2.setWeightsAndBiases(M1);
*/

void check_gmm_sampling()
{
	// Fake predict vector with 3 pdfs
	VF1D pi = {.8, .8, .8};
	VF1D mu = {-5., 3., 12.};
	VF1D sig = {1., 1., 1.};
	VF2D pre(9,1);
	pre.rows(0,2) = pi;
	pre.rows(3,5) = mu;
	pre.rows(6,8) = sig;
	pre.print("pre");
	VF2D vv;
	int sz = 1000;
	VF1D ve(sz);
	std::vector<REAL> prob;
	for (int i=0; i < sz; i++) {
		vv = discrete_sample_gmm1d(pre);
		vv.print("discrete gmm sample");
		prob.push_back(vv(0,0));
		ve(i) = vv(0,0);
	}
	printf("min/max prob: %f, %f\n", arma::min(ve), arma::max(ve));
	//for (int i=0; i < prob.size(); i++) {
		//printf("prob[%d]= %f\n", i, prob[i]);
	//}
	int nb_centers = 60;
	VF1D centers(nb_centers);
	for (int i=0; i < nb_centers; i++) {
		centers[i] = -nb_centers/2. + i; 
	}
	uvec hh = arma::hist(ve,centers);
	printf("centers, histogram\n");
	for (int i=0; i < centers.size(); i++) {
		printf("%f, %d\n", centers[i], hh[i]);
	}
	//hh.print("hh");
	exit(0);
}
//----------------------------------------------------------------------
int discrete_sample(VF2D& x)
{   
// code modified from code obtained on the net
// Input has already been passed through a softmax

    // ideally, these (rd and gen) should be called only once
    std::random_device rd;
    std::mt19937 gen(rd());

    // Setup the weights (in this case linearly weighted)
    int sz = x.n_rows;
//printf("discrete_sample: distribution size: %d\n", sz);

    std::vector<REAL> weights(sz); 
    for(int i=0; i < sz; i++) {
        weights[i] = x(i,0);
    }

    // Create the distribution with those weights
    std::discrete_distribution<> d(weights.begin(), weights.end());

    // use the distribution and print the results.
    int ix = d(gen);
    return ix;
}
//----------------------------------------------------------------------

VF2D discrete_sample_gmm1d(VF2D predict)
{
// code modified from code obtained on the net

    // Setup the random bits
	Softmax soft; 

	// ideally, these (rd and gen) should be called only once
    std::random_device rd;
    std::mt19937 gen(rd()); 

// First compute softmax of prediction to transform the to probabilities

    // predict[batch][inputs, seq]
    // predict[batch][0:inputs/3, :] ==> amplitudes
    // predict[batch][inputs/3:2*inputs/3, :] ==> means
    // predict[batch][2*inputs/3:3*inputs/3, :] ==> standard deviations

    int seq_len = predict.n_cols;
    int input_dim = predict.n_rows;
	if (seq_len > 1) {
		printf("discrete_sample_gmm1d, I expect seq_len to be 1\n");
		exit(1);
	}

	//printf("input_dim= %d\n", input_dim);

    int ia1   = 0;
    int ia2   = input_dim / 3;
    int imu1  = input_dim / 3;
    int imu2  = 2* input_dim / 3;
    int isig1 = 2* input_dim / 3;
    int isig2 = 3* input_dim / 3;

    int Npi = ia2; // number of distributions
    VF2D pi(predict.rows(ia1,ia2-1));
	//pi.print("pi before\n"); // is nan

    // softmax over the dimension index of VF2D (first index)
    REAL mx = arma::max(arma::max(pi));
    for (int s=0; s < seq_len; s++) {
       pi.col(s) = arma::exp(pi.col(s)-mx);
       // trick to avoid overflows
       REAL ssum = 1. / arma::sum(pi.col(s)); // = arma::exp(y[b]);
       pi.col(s) = pi.col(s) * ssum;  // % is elementwise multiplication (arma)
    }

	// Sample from pi
	VF2D sig(predict.rows(isig1, isig2-1));
	sig = arma::exp(sig);
	VF2D mu(predict.rows(imu1, imu2-1));
	//mu.print("mu");
	//sig.print("sig");
	//pi.print("pi");

	int ix = discrete_sample(pi);

	// Sample the ix^{th} normal distribution
	REAL sigma = sig[ix];
	REAL mean  = mu[ix];
	//printf("discrete sample: mean, sigma= %f, %f\n", mean, sigma);

	//sig.print("sig");
	//mu.print("mu");
	//pi.print("pi");
	//printf("ix= %d\n", ix);
	//printf("ix= %d\n", ix);

	VF1D ss = arma::randn<VF1D>(1);

	//U::print(ss, "ss");
	//printf("mu[%d] = %f\n", ix, mean);
	//printf("sig[%d]= %f\n", ix, sigma);
	//printf("ss= %f\n", ss[0]);

	REAL sample = mean + sigma * ss[0]; //arma::randn<VF1D>(1)[0];
	//printf("final discrete sample: %f\n", sample);

	//printf("sample= %f\n", sample);
	//exit(0);

	VF2D res(1,1);
	res(0,0) = sample;
	return res;
}
//----------------------------------------------------------------------
void sample(Model* mi, REAL which_char) 
{
	// Sample the GMM1D

	VF2D_F x(1), y(1);
	std::vector<REAL> message;

	//printf("ENTER SAMPLE, which_char= %d\n", which_char);
	x(0) = 3.;
	//U::print(x, "x");
	message.push_back(which_char);

	mi->resetState();
	//mi->printSummary();
	//exit(0);

	for (int i=0; i < 10; i++) {
		REAL xx = i * dt;
		REAL f = sin(xx);
		//VF2D_F x(1); x(0)(0,0) = f;
		x(0)(0,0) = f;
		y = mi->predictViaConnectionsBias(x);
		message.push_back(x[0](0,0));
		x[0] = discrete_sample_gmm1d(y[0]);
	}

	int nb_points = 300;
	for (int i=0; i < nb_points; i++) {
		//printf("\nsample: i= %d\n", i);
		//printf(".... before ...\n"); U::printOutputs(mi);
		// the output is prior to softmax

		// Retrieve x as output
		// Sample the probability distribution
		// feed the results back into predict

		y = mi->predictViaConnectionsBias(x);
		//printf(".... after ...\n"); U::printOutputs(mi);
		for (int l=0; l < mi->getLayers().size(); l++) {
			mi->getLayers()[l]->setPreviousState();
		}
		x[0] = discrete_sample_gmm1d(y[0]);
		message.push_back(x[0](0,0));
	}
	//exit(0);

	printf("message (100 points): ");
	for (int i=0; i < message.size(); i++) {
		printf("%f %f\n,  ", dt*i, message[i]);
	}
	//printf("\n\n");
}
//----------------------------------------------------------------------
// returns REAL(signal element processed)

REAL getNextGroupOfChars(Model* m, bool& reset, std::vector<REAL>& input_data,
        VF2D_F& net_inputs, VF2D_F& net_exact)
{
	static int base; // keep value across invocations
	if (reset) base = 0;

	int batch_size = m->getBatchSize();
	int input_dim = m->getInputDim();
	int seq_len = m->getSeqLen();

	VF2D_F vf2d; 
	VF2D_F vf2d_exact;

	int nb_chars = input_dim;

	// Assume batch_size = 1
	//if (batch_size != 1) { printf("batch_size should be 1! batch_size=1 is untested! \n"); }

	REAL which_char;

	// Check that we won't go beyond character string
	if (((base + seq_len*batch_size + 2)) >= input_data.size()) {
		return -1;
	}

	for (int b=0; b < batch_size; b++) {
		for (int s=0; s < seq_len; s++) {
			which_char = input_data[base + s];
			net_inputs[b](0, s)       = which_char;

			which_char = input_data[base + (s+1)];
			net_exact[b](0, s) = which_char;
		}
	}
	base += seq_len * batch_size;
	return 1.;
	//return which_char;
}
//----------------------------------------------------------------------
Model* createModel(Globals* g, int batch_size, int seq_len, int input_dim, int layer_size) 
{
	printf("*********** enter createModel ***********\n");
	// Not working

	//Model* m = new Model(*m_old);
	Model* m = new Model();
	m->setSeqLen(seq_len); // ignore seq_len in Globals
	m->setInputDim(input_dim);
	m->setBatchSize(batch_size);
	m->nb_epochs = g->nb_epochs;
	m->setInitializationType(g->initialization_type);
	m->setLearningRate(g->learning_rate);
	m->layer_size = g->layer_size;
	m->init_weight_rms = g->init_weight_rms;

	m->setObjective(new GMM1D()); 
	m->setStateful(true);

	Layer* input = new InputLayer(m->getInputDim(), "input_layer");
	Layer* d1    = new DenseLayer(m->layer_size,    "rdense");
	Layer* d12   = new DenseLayer(m->layer_size,    "rdense");
	int nb_gmms = 10;
	Layer* d2    = new DenseLayer(3*nb_gmms, "gmm"); // layer_size must be multiple of 3 for GMM

	// Softmax is included in the calculation of the cross-entropy

	m->add(0, input);
	m->add(input, d1);
	m->add(d1, d1, true);  // recursive
	m->add(d1, d12);  
	m->add(d12, d12, true);  // recursive
	m->add(d12, d2);

	input->setActivation(new Identity());// Original
	d1->setActivation(new Tanh());
	d12->setActivation(new Tanh());
	//d1->setActivation(new ReLU());
	//d12->setActivation(new ReLU());
	d2->setActivation(new Identity()); // original

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	// create clist
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing
	BIAS& b1  = d1->getBias();
	BIAS& b2  = d2->getBias();
	BIAS& b12 = d12->getBias();

	// NEED A ROUTINE TO SET ALL TRANSPOSES

	// b1 = 0.1 generates a single scalar. Do not know why. 
	b1  = 0.3 * arma::ones<BIAS>(size(b1));
	b2  = 0.3 * arma::ones<BIAS>(size(b2));
	b12 = 0.3 * arma::ones<BIAS>(size(b12));

	// COMPUTE ALL WEIGHT TRANSPOSES

	CONNECTIONS& conns = m->getSpatialConnections();
	for (int c=0; c < conns.size(); c++) {
		conns[c]->computeWeightTranspose();
		printf("%d\n", conns[c]->getTemporal());
	}
	CONNECTIONS& connt = m->getTemporalConnections();
	for (int c=0; c < connt.size(); c++) {
		connt[c]->computeWeightTranspose();
		printf("%d\n", connt[c]->getTemporal());
	}

	m->printSummary();
	printf("====   after printSummary ====\n");

	//U::printWeights(m);
	//U::printInputs(m);
	//U::printOutputs(m);

	return m;
}
//----------------------------------------------------------------------
Model* createModelFF(Globals* g, int batch_size, int seq_len, int input_dim, int layer_size) 
{
	// Feedforward model. Single layer. 
	printf("*********** enter createModelFF ***********\n");
	// Not working

	//Model* m = new Model(*m_old);
	Model* m = new Model();
	m->setSeqLen(seq_len); // ignore seq_len in Globals
	m->setInputDim(input_dim);
	m->setBatchSize(batch_size);
	m->nb_epochs = g->nb_epochs;
	m->setInitializationType(g->initialization_type);
	m->setLearningRate(g->learning_rate);
	m->layer_size = g->layer_size;
	m->init_weight_rms = g->init_weight_rms;

	m->setObjective(new GMM1D()); 
	m->setStateful(true);

	Layer* input = new InputLayer(m->getInputDim(), "input_layer");
	//Layer* d1    = new DenseLayer(m->layer_size,    "rdense");
	int nb_gmms = 2;
	Layer* d2    = new DenseLayer(3*nb_gmms, "gmm"); // layer_size must be multiple of 3 for GMM

	// Softmax is included in the calculation of the cross-entropy

	m->add(0, input);
	m->add(input, d2);
	//m->add(d1, d12);  
	//m->add(d12, d12, true);  // recursive
	//m->add(d12, d2);

	input->setActivation(new Identity());// Original
	//d1->setActivation(new Tanh());
	//d12->setActivation(new Tanh());
	//d1->setActivation(new ReLU());
	//d12->setActivation(new ReLU());
	d2->setActivation(new Identity()); // original

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	// create clist
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing
	//BIAS& b1  = d1->getBias();
	BIAS& b2  = d2->getBias();
	//BIAS& b12 = d12->getBias();

	// NEED A ROUTINE TO SET ALL TRANSPOSES

	// b1 = 0.1 generates a single scalar. Do not know why. 
	//b1  = 0.3 * arma::ones<BIAS>(size(b1));
	b2  = 0.3 * arma::ones<BIAS>(size(b2));
	//b12 = 0.3 * arma::ones<BIAS>(size(b12));

	// COMPUTE ALL WEIGHT TRANSPOSES

	CONNECTIONS& conns = m->getSpatialConnections();
	for (int c=0; c < conns.size(); c++) {
		conns[c]->computeWeightTranspose();
		printf("%d\n", conns[c]->getTemporal());
	}

	m->printSummary();
	printf("====   after printSummary ====\n");

	//U::printWeights(m);
	//U::printInputs(m);
	//U::printOutputs(m);

	return m;
}
//----------------------------------------------------------------------

void gmm1d(Globals* g) 
{
	int layer_size = g->layer_size;
	int is_recurrent = g->is_recurrent;
	int nb_epochs = g->nb_epochs;

// Read Input Data (stored in input_data)
    // Data is continuous: a signal (use a sine() for testing)

	std::vector<REAL> input_data;

	for (int i=0; i < 10000; i++) {
		REAL x = i * dt;
		//REAL f = .7 + .01 * sin(x);
		//REAL f = 1.2 + .1 * sin(x);
		VF1D rn = arma::randn<VF1D>(1);
		REAL f = sin(x) + .00 * rn[0];
		printf("f[%d]= %f\n", i, f);
		input_data.push_back(f);
	}
	//exit(0);

	//--------------------------------------

	// CONSTRUCT MODEL
	int input_dim = 1; // a real signal
	#if 1
	Model* m_train = createModel(g, g->batch_size, g->seq_len, input_dim, g->layer_size);
	Model* m_pred  = createModel(g,             1,          1, input_dim, g->layer_size);
	#else
	Model* m_train = createModelFF(g, g->batch_size, g->seq_len, input_dim, g->layer_size);
	Model* m_pred  = createModelFF(g,             1,          1, input_dim, g->layer_size);
	#endif
	Model* m = m_train;

	//check_gmm_sampling();


	// End of model
	// -----------------------------
	// Run model

	bool reset;
	int nb_samples;
	int seq_len = m->getSeqLen();
	int batch_size = m->getBatchSize();
	nb_samples = input_data.size() / (batch_size * seq_len);
	printf("nb_samples= %d\n", nb_samples); 
	//exit(0);
	Objective* objective = m->getObjective();

	VF2D_F net_inputs, net_exact;
	U::createMat(net_inputs, batch_size, input_dim, seq_len);
	U::createMat(net_exact, batch_size, input_dim, seq_len);

	int count = 1;
	REAL which_char;

	for (int e=0; e < nb_epochs; e++) {
		which_char = input_data[0]; // need better way to initialize
		printf("*** epoch %d ****\n", e);
		m->resetState();
		reset = true;

		for (int i=0; i < nb_samples; i++) {
			//printf("==============================\n");
			#if 1
			// ADD BACK WHEN CODE WORKS
			//if ((count+1) % 1 == 0) {
			printf("count= %d\n", count);
			if (!(count % 100)) {
				printf("TEST, nb_epochs: %d, iter: %d, \n", e, count);
				m_pred->setWeightsAndBiases(m);
				sample(m_pred, which_char);
			}
			#endif

    		REAL wh = getNextGroupOfChars(m, reset, input_data, net_inputs, net_exact);
    		//net_inputs.print("net_inputs");
    		//net_exact.print("net_exact");
			//if (which_char < 0) break;
			// Need a way to exit getNext... when all characters are processed
			reset = false;

	//-----------------------------------------
	#if 0
	// check derivatives via finite-differences
//WEIGHT weightDerivative(Model* m, Connection& con, REAL fd_inc, VF2D_F& xf, VF2D_F& exact)
	//std::vector<WEIGHT> runTest(Model* m, REAL inc, VF2D_F& xf, VF2D_F& exact)
	runTest(m, 0.001, net_inputs, net_exact);
	printf("gordon\n");
	exit(0);
	Connection* con = m->getSpatialConnections()[1];
	con->printSummary();
	WEIGHT fd = weightDerivative(m, *con, .001, net_inputs, net_exact);
	fd.print("fd");
	exit(0);
	#endif
	//-----------------------------------------

			//printf("TRAIN, \n");
			//net_inputs.print("net_inputs");
			//net_exact.print("net_exact");
			VF2D_F predict = m->trainOneBatch(net_inputs, net_exact);
			//U::print(predict, "x predict");
			saveGMM1(predict[0]);
			predict.reset();
			count++;
		}
		printf("\n\n");
	}

	// save data
	int nb_gmm = vpi[0].n_rows;
	FILE *fd = fopen("stats.out", "w");
	for (int i=0; i < vpi.size(); i++) {
	   for (int j=0; j < seq_len; j++) {
	   	 for (int k=0; k < nb_gmm; k++)
			fprintf(fd, "%f ", vpi[i](k,j));
	   	 for (int k=0; k < nb_gmm; k++)
			fprintf(fd, "%f ", vmu[i](k,j));
	   	 for (int k=0; k < nb_gmm; k++)
			fprintf(fd, "%f ", vsig[i](k,j));
		fprintf(fd, "\n");
	}
	}
	#if 0
			#if 1
	   		fprintf(fd, "%f  %f  %f\n",
	   		vpi[i](0,j), 
	   		vmu[i](0,j),
	   		vsig[i](0,j));
			#else
	   		fprintf(fd, "%f  %f  %f  %f  %f  %f  %f  %f  %f\n",
	   		vpi[i](0,j), vpi[i](1,j), vpi[i](2,j),
	   		vmu[i](0,j), vmu[i](1,j), vmu[i](2,j),
	   		vsig[i](0,j), vsig[i](1,j), vsig[i](2,j));
			#endif
		}
	}
	#endif
	fclose(fd);

	//delete input;
	//delete d1;
	//delete d2;
	//delete m_train;
	//delete m_pred;
}

//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -r is_recursive

	Globals* g = processArguments(argc, argv);
	gmm1d(g);
}
//----------------------------------------------------------------------
