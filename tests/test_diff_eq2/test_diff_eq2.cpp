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
	m->setLearningRate(0.01);

	//------------------------------------------------------------------
	// SOLVE ODE  dy/dt = -alpha y
	// Determine alpha, given a curve YT (y target) = exp(-2 t)
	// Initial condition on alpha: alpha(0) = 1.0
	// I expect the neural net to return alpha=2 when the loss function is minimized. 

	int npts = 300;
	printf("npts= %d\n", npts); 
	printf("seq_len= %d\n", seq_len); 

	// npts should be a multiple of seq_len
	npts = (npts / seq_len) * seq_len; 


	VF1D ytarget(npts);
	VF1D x(npts);   // abscissa
	REAL delx = .025;  // will this work for uneven time steps? dt = .1, but there is a multiplier: alpha in front of it. 
	                 // Some form of normalization will probably be necessary to scale out effect of dt (discrete time step)
	m->dt = delx;
	REAL alpha_target = 1.;
	REAL alpha_initial = 1.5;  // should evolve towards alpha_target

	for (int i=0; i < npts; i++) {
		x[i] = i*delx;
		ytarget[i] = exp(-alpha_target * x[i]);
		printf("x: %f, target: %f\n", x[i], ytarget[i]);
	}

	// set all alphas to alpha_initial
	LAYERS layers = m->getLayers();

	for (int l=0; l < layers.size(); l++) {
		Layer* layer = layers[l];
		printf("l= %d\n", l);
		// layers without parameters will ignore this call
		layer->getActivation().setParam(0, alpha_initial);
		layer->getActivation().setDt(m->dt);
	}

	// Assume nb_batch=1 for now. Extract a sequence of seq_len elements to input
	// input into prediction followed by backprop followed by parameter updates.
	// What I want is a data structure: 
	//  VF2D_F[nb_batch][nb_inputs, seq_len] = VF2D_F[1][1, seq_len]
	// 

	int nb_samples = npts / seq_len; 
	std::vector<VF2D_F> net_inputs;
	VF2D_F vf2d;
	U::createMat(vf2d, nb_batch, 1, seq_len);

	VF2D_F vf2d_exact;
	U::createMat(vf2d_exact, nb_batch, 1, seq_len);

	// Assumes nb_batch = 1 and input dimensionality = 1
	for (int i=0; i < nb_samples; i++) {
		printf("i= %d\n", i);
		for (int j=0; j < seq_len; j++) {
			printf("i,j= %d, %d\n", i,j);
			vf2d[0](0, j) = ytarget(j + seq_len*i);
			net_inputs.push_back(vf2d);
		}
	}


//void Model::train(VF2D_F x, VF2D_F exact, int batch_size /*=0*/, int nb_epochs /*=1*/) 
	net_inputs[0].print("net_inputs[0]");
	net_inputs[1].print("net_inputs[1]");

	for (int i=0; i < nb_samples-1; i++) {
		m->train(net_inputs[i], net_inputs[i+1], nb_batch, 1);
	}
	//------------------------------------------------------------------

	exit(0);

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
}
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

	Model* m = processArguments(argc, argv);
	testDiffEq1(m);
}

