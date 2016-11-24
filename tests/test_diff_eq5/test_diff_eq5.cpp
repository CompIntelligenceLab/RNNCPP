#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>


//----------------------------------------------------------------------
void testDiffEq5(Model* m)
{
	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;
	int nb_epochs = m->nb_epochs;

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
	m->add(d1, d2); // spatial
	m->add(d2, d1, true); // temporal link
	input->setActivation(new Identity()); 
	d1->setActivation(new DecayDE());
	//d2->setActivation(new Identity()); // Identity should give same result as no node (works)
	d2->setActivation(new Tanh());

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	printf("total nb layers: %d\n", m->getLayers().size());
	m->printSummary();
	// Code crashes if not called. 
	// compute "clist" datastructure (list of connections)
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing
	m->freezeBiases();
	m->freezeWeights();

	//********************** END MODEL *****************************

	BIAS& bias1 = input->getBias(); 	bias1.zeros();
	BIAS& bias2 =    d1->getBias(); 	bias2.zeros();
	BIAS& bias3 =    d2->getBias(); 	bias3.zeros();

	// Set the weights of the two connection that input into Layer d1 to 1/2
	// This should create a stable, consistent numerical scheme
	WEIGHT& w1  = m->getConnection(input, d1)->getWeight();
	w1[0,0]    *= 0.5;
	WEIGHT& wr2 = m->getConnection(d2, d1)->getWeight();
	wr2[0,0]   *= 0.5;

    //------------------------------------------------------------------
    // SOLVE ODE  dy/dt = -alpha y
    // Determine alpha, given a curve YT (y target) = exp(-2 t)
    // Initial condition on alpha: alpha(0) = 1.0
    // I expect the neural net to return alpha=2 when the loss function is minimized.

	std::vector<VF2D_F> net_inputs, net_exact;
	int nb_samples = getData(m, net_inputs, net_exact);

	m->setStateful(true);
	m->resetState();

	for (int e=0; e < nb_epochs; e++) {
		m->resetState();
		// First iteration, make effective weight from input to d1 equal to one
		net_inputs[0][0][0,0] *= 2.;  // only once per epoch

		printf("**** Epoch %d ****\n", e);

		for (int i=0; i < nb_samples-1; i++) {
			if (m->getStateful() == false) m->resetState();
			m->trainOneBatch(net_inputs[i][0], net_exact[i][0]);
			updateWeightsSumConstraint(m, input, d1, d2, d1);
		}

		// Must reset net_inputs to original value
		net_inputs[0][0][0,0] /= 2.;
	}

	//****************
	m->printHistories();
	m->addWeightHistory(input, d1);
	m->addWeightHistory(d2, d1);
	m->printWeightHistories();

	// Run prediction. How to do prediction: stateful with seq_len=1. Wonder what I'll get. 

	printf("XXX END PROGRAM\n");

	exit(0);
}
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -r is_recursive

	Model* m = processArguments(argc, argv);
	testDiffEq5(m);
}
//----------------------------------------------------------------------
