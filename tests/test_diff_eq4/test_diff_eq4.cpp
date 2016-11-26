#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>


//#include "testDiffEq4.h"
//#include "testDiffEq6.h"

//----------------------------------------------------------------------
void testDiffEq4(Model* m)
{
	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;
	int nb_epochs = m->nb_epochs;

	printf("\n\n\n");
	printf("=============== BEGIN test_recurrent_model_bias2  =======================\n");

	//********************** BEGIN MODEL *****************************
	int seq_len = m->getSeqLen();
	int nb_batch = m->getBatchSize();
	int input_dim  = 1;

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer* d1    = new DenseLayer(layer_size, "rdense");
	Layer* d2    = new DenseLayer(layer_size, "rdense");
	m->add(0, input);
	m->add(input, d1);
	m->add(d1, d2);
	m->add(d1, d1, true); // temporal link
	m->add(d2, d2, true); // temporal link
	input->setActivation(new Identity()); 
	d1->setActivation(new DecayDE());
	d2->setActivation(new DecayDE());

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	printf("total nb layers: %d\n", m->getLayers().size());
	m->printSummary();
	// Code crashes if not called
	// compute clist datastructure (list of connections)
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing
	m->freezeBiases();
	m->freezeWeights();

	//********************** END MODEL *****************************

	m->initializeWeights();

	BIAS& bias1 = input->getBias(); 	bias1.zeros();
	BIAS& bias2 =    d1->getBias(); 	bias2.zeros();
	BIAS& bias3 =    d2->getBias(); 	bias3.zeros();

	// Set the weights of the two connection that input into Layer d1 to 1/2
	// This should create a stable numerical scheme
	WEIGHT& w1  = m->getConnection(input, d1)->getWeight();	w1[0,0]    *= 0.5;
	WEIGHT& wr1 = m->getConnection(d1, d1)->getWeight();    wr1[0,0]   *= 0.5;

	WEIGHT& w2  = m->getConnection(d1, d2)->getWeight();	w2[0,0]    *= 0.5;
	WEIGHT& wr2 = m->getConnection(d2, d2)->getWeight();	 wr2[0,0]   *= 0.5;

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

	//------------------------------------------------------
	for (int e=0; e < nb_epochs; e++) {
		m->resetState();
		// First iteration, make effective weight from input to d1 equal to one
		net_inputs[0][0][0,0] *= 2.;  // only once per epoch

		printf("**** Epoch %d ****\n", e);

		for (int i=0; i < nb_samples-1; i++) {
			if (m->getStateful() == false) m->resetState();
			m->trainOneBatch(net_inputs[i][0], net_exact[i][0]);
			updateWeightsSumConstraint(m, input, d1, d1, d1);
			updateWeightsSumConstraint(m, d1, d2, d2, d2);
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
	m->printHistories();
	m->addWeightHistory(input, d1);
	m->addWeightHistory(d1, d1);
	m->addWeightHistory(d1, d2);
	m->addWeightHistory(d2, d2);
	m->printWeightHistories();
	//------------------------------------------------------------------

	printf("XXX gordon XXX\n");

	exit(0);
}
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

	Model* m = processArguments(argc, argv);

	// two equation nodes in series
	testDiffEq4(m);

	// two equation nodes in parallel
	//testDiffEq6(m);
}

