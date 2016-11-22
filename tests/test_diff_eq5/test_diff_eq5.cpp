#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>


//----------------------------------------------------------------------
void testDiffEq5(Model* m)
{
	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;

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
	// Code crashes if not called
	// compute clist datastructure (list of connections)
	m->connectionOrderClean(); // no print statements

	CONNECTIONS& conns = m->getConnections();
	for (int c=0; c < conns.size(); c++) {
		conns[c]->printSummary("connections, ");
	}
	CONNECTIONS& tconns = m->getTemporalConnections();
	for (int c=0; c < tconns.size(); c++) {
		tconns[c]->printSummary("temporal connections, ");
	}

	#if 0
	// FREEEZE weights  (if unfrozen, code does not run. Nans arise.)
    CONNECTIONS& cons = m->getConnections();
	for (int i=0; i < cons.size(); i++) {
		Connection* con = cons[i];
		//con->printSummary();
		con->freeze();
	}

    CONNECTIONS& tcons = m->getTemporalConnections();
	for (int i=0; i < tcons.size(); i++) {
		Connection* con = tcons[i];
		//con->printSummary();
		con->freeze();
	}
	#endif

	#if 1
	// FREEZE Biases
	input->setIsBiasFrozen(true);
	d1->setIsBiasFrozen(true);
	d2->setIsBiasFrozen(true);
	#endif
	//********************** END MODEL *****************************

	m->initializeWeights();

	BIAS& bias1 = input->getBias(); 	bias1.zeros();
	BIAS& bias2 =    d1->getBias(); 	bias2.zeros();
	BIAS& bias3 =    d2->getBias(); 	bias3.zeros();

	// Set the weights of the two connection that input into Layer d1 to 1/2
	// This should create a stable, consistent numerical scheme
	WEIGHT& w1  = m->getConnection(input, d1)->getWeight();
	w1[0,0]    *= 0.5;
	WEIGHT& wr2 = m->getConnection(d2, d1)->getWeight();
	wr2[0,0]   *= 0.5;

	m->setLearningRate(20.);
	m->setLearningRate(5.);
	//m->setLearningRate(.01);
	//m->setLearningRate(2.);

	std::vector<VF2D_F> net_inputs, net_exact;
	int nb_samples = getData(m, net_inputs, net_exact);

	m->setStateful(false);
	m->setStateful(true);
	m->resetState();

	int nb_epochs;
	nb_epochs = 20;
	nb_epochs = 2000;

	// allow these weights 
	m->getConnection(input, d1)->freeze();
	m->getConnection(d2, d1)->freeze();

	for (int e=0; e < nb_epochs; e++) {

		// First iteration, make effective weight from input to d1 equal to one
		net_inputs[0][0][0,0] *= 2.;

		printf("**** Epoch %d ****\n", e);
		for (int i=0; i < nb_samples-1; i++) {
			if (m->getStateful() == false) {
				m->resetState();
			}
			net_inputs[i][0].print("net_inputs");
			m->trainOneBatch(net_inputs[i][0], net_exact[i][0]);
			updateWeightsSumConstraint(m, input, d1, d2, d1);
		}
		m->resetState();

		// Must reset net_inputs to original value
		net_inputs[0][0][0,0] /= 2.;
	}
	//------------------------------------------------------------------

	U::printWeights(m);
	U::printLayerBiases(m);
	printf("XXX gordon XXX\n");

	exit(0);
}
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

	Model* m = processArguments(argc, argv);
	testDiffEq5(m);
}

