#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>


//----------------------------------------------------------------------
//void testRecurrentModelBias1(Model* m, int layer_size, int is_recurrent, Activation* activation, REAL inc) 
void testDiffEq1(Model* m)
{
	//testRecurrentModelBias1(m, layer_size, is_recurrent, activation, inc);
	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;
	REAL inc = m->inc;
	int nb_serial_layers = m->nb_serial_layers;
	int nb_parallel_layers = m->nb_parallel_layers;


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
	Layer *d1;

	typedef std::vector<Layer*> SLAYERS;
	typedef std::vector<SLAYERS>  PLAYERS;
	//std::vector<std::vector<Layer*> > parallel_layers(nb_parallel_layers);
	//std::vector<Layer*> internal_layers;
	PLAYERS internal_layers;
	internal_layers.resize(nb_parallel_layers);
	for (int i=0; i < nb_parallel_layers; i++) {
		internal_layers[i].resize(nb_serial_layers);
	}

	is_recurrent = 0;

	for (int i=0; i < m->activations.size(); i++) {
		m->activations[i]->setParam(0, .1*i+.1);
	}

	for (int j=0; j < nb_parallel_layers; j++) {
		for (int i=0; i < nb_serial_layers; i++) {
			if (is_recurrent) {
				d1    = new RecurrentLayer(layer_size, "rdense");
			} else {
				d1    = new DenseLayer(layer_size, "rdense");
			}
			internal_layers[j][i] = d1;
			printf("--- j= %d, size: %d\n", j, internal_layers[j].size());
		}
		m->add(0,     input);
		input->setActivation(new Identity()); 

		printf("j= %d\n", j);
		printf("layer: %ld\n", internal_layers[j][0]);
    	internal_layers[j][0]->setActivation(m->activations[0+j*nb_serial_layers]); 
		m->add(input, internal_layers[j][0]);

		for (int i=1; i < nb_serial_layers; i++) {
			printf("i= %d, j= %d, activation names: %s\n", i, j, m->activations[i+j*nb_serial_layers]->getName().c_str());
    		internal_layers[j][i]->setActivation(m->activations[i+j*nb_serial_layers]); 
			m->add(internal_layers[j][i-1], internal_layers[j][i]);
		}
	}

	// connect the two parallel layers by another layer 

	if (is_recurrent) {
		d1 = new RecurrentLayer(layer_size, "rdenseOut");
	} else {
		d1 = new DenseLayer(layer_size, "rdenseOut");
	}

	for (int j=0; j < nb_parallel_layers; j++) {
		m->add(internal_layers[j][nb_serial_layers-1], d1);
	}
	d1->setActivation(new Tanh());


	// input should always be identity activation

	m->addInputLayer(input);
	m->addOutputLayer(d1);

	printf("total nb layers: %d\n", m->getLayers().size());
	m->printSummary();
	m->connectionOrderClean(); // no print statements

	//********************** END MODEL *****************************

	m->initializeWeights();

	// RUN TEST TO CHECK ACCURACY OF WEIGHT, BIAS, ACTIVATION PARAMS deltas (dLoss/d...)

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
	exit(0);
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

	Model* m = processArguments(argc, argv);
	testDiffEq1(m);
}

