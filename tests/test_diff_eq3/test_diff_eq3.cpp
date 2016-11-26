#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>


//----------------------------------------------------------------------
void testDiffEq3(Model* m)
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
	//m->setObjective(new LogMeanSquareError()); // NEW
	m->setObjective(new MeanSquareError()); 

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer* d1    = new DenseLayer(layer_size, "rdense");
	m->add(0, input);
	m->add(input, d1);
	m->add(d1, d1, true); // temporal link
	input->setActivation(new Identity()); 
	d1->setActivation(new DecayDE());

	m->addInputLayer(input);
	m->addOutputLayer(d1);

	m->printSummary();
	// Code crashes if not called
	// compute clist datastructure (list of connections)
	m->connectionOrderClean(); // no print statements

    m->initializeWeights();
	m->freezeBiases();
    m->freezeWeights();

	//********************** END MODEL *****************************

	BIAS& bias1 = input->getBias();
	BIAS& bias2 =    d1->getBias();
	bias1.zeros();
	bias2.zeros();

	// Set the weights of the two connection that input into Layer d1 to 1/2
	// This should create a stable numerical scheme
	WEIGHT& w = m->getConnection(input, d1)->getWeight();
	w[0,0] *= 0.5;
	WEIGHT& wr = m->getConnection(d1, d1)->getWeight();
	wr[0,0] *= 0.5;

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

	for (int e=0; e < nb_epochs; e++) {

		// First iteration, make effective weight from input to d1 equal to one
		net_inputs[0][0][0,0] *= 2.;

		printf("**** Epoch %d ****\n", e);

		for (int i=0; i < nb_samples-1; i++) {
			if (m->getStateful() == false)  m->resetState();
			m->trainOneBatch(net_inputs[i][0], net_exact[i][0]);
			updateWeightsSumConstraint(m, input, d1, d1, d1);
		}
		m->resetState();

		// Must reset net_inputs to original value
		net_inputs[0][0][0,0] /= 2.;
	}
	//------------------------------------------------------------------
	// Starting from initial condition, recreate the solution. 
	{
		VF2D_F x; 
		m->setSeqLen(1); 
		//printf("seq len: %d\n", m->getSeqLen()); exit(0);
		U::createMat(x, nb_batch, 1, m->getSeqLen());
		x[0][0,0] = net_inputs[0][0][0,0];

		// Done since weights are 1/2 (due to recursion and consistency) and there is no data
		// on the recurrent weight initially. Somewhat artificial. 
		x[0][0,0] *= 2.;

		for (int e=0; e < 500; e++) {
			m->x_in_history.push_back(ytarget[e]);
			m->x_out_history.push_back(x[0][0,0]);
			x = m->predictViaConnectionsBias(x);
		}
	}
	//------------------------------------------------------------------

    // Hard to abstract away since I am only printing specific weights 
	// Must find a way to only print specified weights or weight statistics
	m->addWeightHistory(input, d1);
	m->addWeightHistory(d1, d1);
	m->addParamsHistory(d1);

    m->printHistories();
	m->printWeightHistories();

	printf("XXX END XXX\n");

	exit(0);
}
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

	Model* m = processArguments(argc, argv);
	testDiffEq3(m);
}

