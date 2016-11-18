#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>


//----------------------------------------------------------------------
void testDiffEq3(Model* m)
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

	CONNECTIONS& conns = m->getConnections();
	for (int c=0; c < conns.size(); c++) {
		conns[c]->printSummary("connections, ");
	}
	CONNECTIONS& tconns = m->getTemporalConnections();
	for (int c=0; c < tconns.size(); c++) {
		tconns[c]->printSummary("temporal connections, ");
	}

	#if 1
	// FREEEZE weights and biases

	#if 1
	// FREEEZE weights  (if unfrozen, code does not run. Nans arise.)
    CONNECTIONS& cons = m->getConnections();
	for (int i=0; i < cons.size(); i++) {
		Connection* con = cons[i];
		con->printSummary();
		con->freeze();
	}

    CONNECTIONS& tcons = m->getTemporalConnections();
	for (int i=0; i < tcons.size(); i++) {
		Connection* con = tcons[i];
		con->printSummary();
		con->freeze();
	}
	#endif

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
	// This should create a stable numerical scheme
	WEIGHT& w1  = m->getConnection(input, d1)->getWeight();
	w1[0,0]    *= 0.5;
	WEIGHT& wr1 = m->getConnection(d1, d1)->getWeight();
	wr1[0,0]   *= 0.5;

	WEIGHT& w2  = m->getConnection(d1, d2)->getWeight();
	w2[0,0]    *= 0.5;
	WEIGHT& wr2 = m->getConnection(d2, d2)->getWeight();
	wr2[0,0]   *= 0.5;

	m->setLearningRate(20.);
	m->setLearningRate(5.);
	//m->setLearningRate(.01);
	//m->setLearningRate(2.);

	//------------------------------------------------------------------
	// SOLVE ODE  dy/dt = -alpha y
	// Determine alpha, given a curve YT (y target) = exp(-2 t)
	// Initial condition on alpha: alpha(0) = 1.0
	// I expect the neural net to return alpha=2 when the loss function is minimized. 

	int npts = 600;
	printf("npts= %d\n", npts); 
	printf("seq_len= %d\n", seq_len); 

	// npts should be a multiple of seq_len
	npts = (npts / seq_len) * seq_len; 

	VF1D ytarget(npts);
	VF1D x(npts);   // abscissa
	REAL delx = .005;  // will this work for uneven time steps? dt = .1, but there is a multiplier: alpha in front of it. 
	                 // Some form of normalization will probably be necessary to scale out effect of dt (discrete time step)
	m->dt = delx;
	REAL alpha_target = 2.;
	REAL alpha_initial = 1.;  // should evolve towards alpha_target

	// this works (duplicates Mark Lambert's case)
	//REAL alpha_target = 1.;
	//REAL alpha_initial = 2.;  // should evolve towards alpha_target

	for (int i=0; i < npts; i++) {
		x[i] = i*delx;
		ytarget[i] = exp(-alpha_target * x[i]);
		//printf("x: %21.14f, target: %21.14f\n", x[i], ytarget[i]);
	}

	// set all alphas to alpha_initial
	LAYERS layers = m->getLayers();

	for (int l=0; l < layers.size(); l++) {
		Layer* layer = layers[l];
		//printf("l= %d\n", l);
		// layers without parameters will ignore this call
		layer->getActivation().setParam(0, alpha_initial); // 1st parameter
		layer->getActivation().setDt(m->dt);
	}

	// Assume nb_batch=1 for now. Extract a sequence of seq_len elements to input
	// input into prediction followed by backprop followed by parameter updates.
	// What I want is a data structure: 
	//  VF2D_F[nb_batch][nb_inputs, seq_len] = VF2D_F[1][1, seq_len]
	// 

	int nb_samples = npts / seq_len; 
	std::vector<VF2D_F> net_inputs, net_exact;
	VF2D_F vf2d;
	U::createMat(vf2d, nb_batch, 1, seq_len);

	VF2D_F vf2d_exact;
	U::createMat(vf2d_exact, nb_batch, 1, seq_len);

	// Assumes nb_batch = 1 and input dimensionality = 1
	for (int i=0; i < nb_samples-1; i++) {
		for (int j=0; j < seq_len; j++) {
			vf2d[0](0, j)       = ytarget(j + seq_len*i);
			vf2d_exact[0](0, j) = ytarget(j + seq_len*i + 1);
		}
		net_inputs.push_back(vf2d);
		net_exact.push_back(vf2d_exact);
	}

	m->setStateful(false);
	m->setStateful(true);
	m->resetState();

	int nb_epochs;
	nb_epochs = 2;
	nb_epochs = 400;

	for (int e=0; e < nb_epochs; e++) {

		// First iteration, make effective weight from input to d1 equal to one
		net_inputs[0][0][0,0] *= 2.;

		printf("**** Epoch %d ****\n", e);
		for (int i=0; i < nb_samples-1; i++) {
			if (m->getStateful() == false) {
				m->resetState();
			}
			m->trainOneBatch(net_inputs[i][0], net_exact[i][0]);
			// deal with weight constraints (weights are frozon, but update by hand)
			// in this case, sum of two weights is unity
			// w1 + w2 = 1 ==> delta(w1) + delta(w2) = 0. 
			// call w1 = w and w2 = 1-w. Compute delta(w)
			// Compute  delta(w) = delta(w1) - delta(w2)
			// w1 -= lr * delta(w)
			// w2 += lr * delta(w)
			#if 0
			{
			WEIGHT& deltaw1 = m->getConnection(input, d1)->getDelta();
			WEIGHT& deltaw2 = m->getConnection(d1, d1)->getDelta();
			WEIGHT delta = deltaw1 - deltaw2;
			WEIGHT& w1 = m->getConnection(input, d1)->getWeight();
			WEIGHT& w2 = m->getConnection(d1, d1)->getWeight();
			REAL lr = m->getLearningRate();
			delta.print("delta");
			w1 -= 0.001 * lr * delta;
			w2 += 0.001 * lr * delta;
			printf("w1, w2= %f, %f\n", w1[0,0], w2[0,0]);
			}
			{
			WEIGHT& deltaw1 = m->getConnection(d1, d2)->getDelta();
			WEIGHT& deltaw2 = m->getConnection(d2, d2)->getDelta();
			WEIGHT delta = deltaw1 - deltaw2;
			WEIGHT& w1 = m->getConnection(d1, d2)->getWeight();
			WEIGHT& w2 = m->getConnection(d2, d2)->getWeight();
			REAL lr = m->getLearningRate();
			delta.print("delta");
			w1 -= 0.001 * lr * delta;
			w2 += 0.001 * lr * delta;
			printf("w1, w2= %f, %f\n", w1[0,0], w2[0,0]);
			}
			#endif
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
	testDiffEq3(m);
}

