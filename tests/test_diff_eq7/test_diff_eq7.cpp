#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>

/**
z1 = a1*xT(n) + a2*x(n) + a3*y(n)
z1 = a1*xT(n) + a2*x(n) + a3*y(n)
x1(n+1) = a1 * [x(n) + dt * (-alpha1*x(n))] + a2 * x2
x2 = x(n) + dt * (-alpha2*x(n))
x(n+1) = w1*x1 + w2*x2
**/


//----------------------------------------------------------------------
void testDiffEq7(Model* m)
{
// Two DecayEQ nodes in parallel, and cross-linked

	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;
	int nb_epochs = m->nb_epochs;

	printf("\n\n\n");
	printf("=============== BEGIN test_recurrent_model_bias2  =======================\n");

	//********************** BEGIN MODEL *****************************
	int seq_len = m->getSeqLen();
	int nb_batch = m->getBatchSize();
	int input_dim  = 1;

	m->setObjective(new MeanSquareError()); 
	m->getObjective()->setErrorType(m->obj_err_type);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer* d1    = new DenseLayer(layer_size, "rdense");
	Layer* d2    = new DenseLayer(layer_size, "rdense");
	Layer* dsum    = new DenseLayer(1, "rdense");
	m->add(0, input);
	m->add(input, d1);
	m->add(input, d2);
	//m->add(d1, d1, true); // temporal link
	//m->add(d2, d2, true); // temporal link
	m->add(d2, d1, true); // temporal link  (link between parallel nodes)
	m->add(d1, d2, true); // temporal link
	m->add(d1, dsum);
	m->add(d2, dsum);
	input->setActivation(new Identity()); 
	d1->setActivation(new DecayDE());
	d2->setActivation(new DecayDE());
	dsum->setActivation(new Identity());

	m->addInputLayer(input);
	m->addOutputLayer(dsum);

	printf("total nb layers: %d\n", m->getLayers().size());
	m->printSummary();
	// Code crashes if not called
	// compute clist datastructure (list of connections)
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing
	m->freezeBiases();
	m->freezeWeights();

	// Unfreeze weights (d1,dsum) and (d2,dsum)
	//m->getConnection(d1,dsum)->unfreeze();
	//m->getConnection(d2,dsum)->unfreeze();
	//m->getConnection(d3,dsum)->unfreeze();

	//********************** END MODEL *****************************

	m->initializeWeights();

	BIAS& bias0 = input->getBias(); 	bias0.zeros();
	BIAS& bias1 =    d1->getBias(); 	bias1.zeros();
	BIAS& bias2 =    d2->getBias(); 	bias2.zeros();
	BIAS& biasdsum =    dsum->getBias(); 	biasdsum.zeros();  // should biasdsum be frozen? 

	// Set the weights of the two connection that input into Layer d1 to 1/2
	// This should create a stable numerical scheme
	WEIGHT& w1  = m->getConnection(input, d1)->getWeight();	w1[0,0]  *= 0.5;
	//WEIGHT& w11 = m->getConnection(d1, d1)->getWeight();    w11[0,0] *= 0.35;
	WEIGHT& w21 = m->getConnection(d2, d1)->getWeight();    w21[0,0] *= 0.5;

	WEIGHT& w2  = m->getConnection(input, d2)->getWeight();	w2[0,0]  *= 0.5; // introduce slight asymmetry
	//WEIGHT& w22 = m->getConnection(d2, d2)->getWeight();	w22[0,0] *= 0.35;
	WEIGHT& w12 = m->getConnection(d1, d2)->getWeight();	w12[0,0] *= 0.5;

	// Ideally, these weights should be able to vary and sum to 1
	// One way to do this is to have three weights in the network, connect to a softmax, and make the three output weights
	// the softmax coefficients. Since the weights could be negative, we do not need a softmax: w1 / (w1 + w2 + w3). 
	// If they initially sum to one, it is unlikely they will ever sum to zero if learning rate is sufficiently small
	WEIGHT& wrs1 = m->getConnection(d1, dsum)->getWeight();	wrs1[0,0] *= 0.45;
	WEIGHT& wrs2 = m->getConnection(d2, dsum)->getWeight();	wrs2[0,0] *= 0.55;

	//------------------------------------------------------------------
	// SOLVE ODE  dy/dt = -alpha y
	// Determine alpha, given a curve YT (y target) = exp(-2 t)
	// Initial condition on alpha: alpha(0) = 1.0
	// I expect the neural net to return alpha=2 when the loss function is minimized. 

	std::vector<VF2D_F> net_inputs, net_exact;
	VF1D xabsc, ytarget;
	int nb_samples = getData(m, net_inputs, net_exact, xabsc, ytarget);

	m->setStateful(true);
	m->resetState();

	//d1->getActivation().setParam(0,  .15); // 1st parameter
	//d2->getActivation().setParam(0, -.15); // 1st parameter
	d1->getActivation().setParam(0,  .75); // 1st parameter
	d2->getActivation().setParam(0, -.75); // 1st parameter

	//------------------------------------------------------
	for (int e=0; e < nb_epochs; e++) {
		m->resetState();

		// First iteration, make effective weight from input to d1 equal to one
		// PROBABLY INCORRECT WITH MORE THAN ONE NODE PER LAYER. NEED SOME SORT OF INTELLIGENT SPLITTER
		net_inputs[0][0][0,0] *= 2.;  // only once per epoch

		printf("**** Epoch %d ****\n", e);

		for (int i=0; i < nb_samples-1; i++) {
			if (m->getStateful() == false) m->resetState();
			m->trainOneBatch(net_inputs[i][0], net_exact[i][0]);
			updateWeightsSumConstraint(m, input, d1, d2, d1);
			updateWeightsSumConstraint(m, input, d2, d1, d2);
			//updateWeightsSumConstraint(m, input, d1, d1, d1, d2, d1);
			//updateWeightsSumConstraint(m, input, d2, d2, d2, d1, d2);
			updateWeightsSumConstraint(m, d1, dsum, d2, dsum);
			// The sum of the weights between d1-dsum, d2-dsum, d3-dsum should add to 1. Not implemented. So freeze these weights
		}

		// Must reset net_inputs to original value
		net_inputs[0][0][0,0] /= 2.;
	}
	//------------------------------------------------------------------
	// Starting from initial condition, recreate the solution. 
	{
		VF2D_F x; 
		m->setSeqLen(1); 
		U::createMat(x, nb_batch, 1, m->getSeqLen());
		x[0][0,0] = net_inputs[0][0][0,0];

		// Done since weights are 1/2 (due to recursion and consistency) and there is no data
		// on the recurrent weight initially. Somewhat artificial. 
		x[0][0,0] *= 2.;

		m->resetState();

		for (int e=0; e < 500; e++) {
			m->x_in_history.push_back(ytarget[e]);
			m->x_out_history.push_back(x[0][0,0]);
			x = m->predictViaConnectionsBias(x);
		}
	}
	//------------------------------------------------------------------
	m->addWeightHistory(input, d1);
	//m->addWeightHistory(d1, d1);
	m->addWeightHistory(d2, d1);
	m->addWeightHistory(input, d2);
	//m->addWeightHistory(d2, d2);
	m->addWeightHistory(d1, d2);
	m->addWeightHistory(d1, dsum);
	m->addWeightHistory(d2, dsum);

	//printf("YYY\n"); exit(0);
	m->addParamsHistory(d1);
	m->addParamsHistory(d2);

	std::vector<std::vector<REAL> > h1 = d1->params_history;
	std::vector<std::vector<REAL> > h2 = d2->params_history;
	for (int i=0; i < 5; i++) {
		printf("param history: %f, %f\n", h1[i][0], h2[i][0]);
	}

	m->printHistories();
	m->printWeightHistories();
	//------------------------------------------------------------------

	printf("XXX gordon XXX\n");

	exit(0);
}
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -r is_recursive

	Model* m = processArguments(argc, argv);
	testDiffEq7(m);
}
//----------------------------------------------------------------------
