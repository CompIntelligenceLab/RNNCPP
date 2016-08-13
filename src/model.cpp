#include <assert.h>
#include "model.h"
#include "objective.h"
#include "connection.h"
#include "typedefs.h"
#include <stdio.h>

Model::Model(int input_dim, std::string name /* "model" */) 
{
	this->name = name;
	learning_rate = 1.e-5;
	return_sequences = false;
	print_verbose = true;
	this->input_dim = input_dim;
	printf("Model constructor (%s)\n", this->name.c_str());
	optimizer = new RMSProp();
	objective = new MeanSquareError();
	//objective = NULL; // I believe this should just be a string or something specifying
               // the name of the objective function we would like to use
			   // WHY a string? (Gordon)
	batch_size = 1; // batch size of 1 is equivalent to online learning
	nb_epochs = 10; // Just using the value that Keras uses for now
    stateful = false; 
	seq_len = 1; // should be equivalent to feedforward (no time to unroll)
	initialization_type = "uniform";  // can also choose Gaussian

}
//----------------------------------------------------------------------
Model::~Model()
{
	printf("Model destructor (%s)\n", name.c_str());

	for (int i=0; i < layers.size(); i++) {
		if (layers[i]) {delete layers[i]; layers[i] = 0;}
	}

	if (optimizer) {
		delete optimizer;
		optimizer = 0;
	}

	if (objective) {
		delete objective;
		objective = 0;
	}
}
//----------------------------------------------------------------------
Model::Model(const Model& m) : stateful(m.stateful), learning_rate(m.learning_rate), 
    return_sequences(m.return_sequences), input_dim(m.input_dim), batch_size(m.batch_size),
	seq_len(m.seq_len), print_verbose(m.print_verbose), initialization_type(m.initialization_type),
	nb_epochs(m.nb_epochs)

	// What to do with name (perhaps add a "c" at the end for copy-construcor?)
{
	name = m.name + "c";
	optimizer = new Optimizer();
    *optimizer = *m.optimizer;  // Careful here, we need to implement a copy
                                // assignment operator for the Optimizer class
	objective = new MeanSquareError(); 
    *objective = *m.objective;// Careful here, we need to implement a copy
                  // assignment operator for the Optimizer class
	layers = m.layers; // Careful here, we need to implement a copy
                     // assignment operator for the Optimizer class
	printf("Model copy constructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
const Model& Model::operator=(const Model& m) 
{
	if (this != &m) {
		name = m.name + "=";
		stateful = m.stateful;
		learning_rate = m.learning_rate;
		return_sequences = m.return_sequences;
		input_dim = m.input_dim;
		batch_size = m.batch_size;
        nb_epochs = m.nb_epochs;
		seq_len = m.seq_len;
		print_verbose= m.print_verbose;
		initialization_type = m.initialization_type;

		Optimizer* opt1 = NULL;
		Objective* objective1 = NULL;

		try {
			opt1 = new Optimizer(); //*m.optimizer);
			objective1 = new MeanSquareError(); //*m.objective);
		} catch (...) {
			delete opt1;
			delete objective1;
			printf("Model throw\n");
			throw;
		}

		// Superclass::operator=(that)
		*optimizer = *opt1;
		*objective = *objective1;
		printf("Model::operator= %s\n", name.c_str());
	}
	return *this;
}
//----------------------------------------------------------------------
/**
void Model::add(Layer* layer)
{
	// check for layer size compatibility
	printf("add layer ***** layers size: %d\n", layers.size());

	if (layers.size() == 0) {
		// 0th layer is an InputLayer
		;
	} else {
		int nb_layers = layers.size();
		//printf("nb_layers= %d\n", nb_layers);
		if (layers[nb_layers-1]->getLayerSize() != layer->getInputDim()) {
			layer->setInputDim(layers[nb_layers-1]->getLayerSize());
			printf("layer[%d], layer_size= %d\n", nb_layers-1, layers[nb_layers-1]->getLayerSize());
			printf("new layer input size: %d\n", layer->getInputDim());
			//printf("Incompatible layer_size between layers %d and %d\n", nb_layers-1, nb_layers);
			//exit(0);
		}
	}

	int in_dim  = layer->getInputDim();
	int out_dim = layer->getOutputDim();
	printf("Model::add, layer dim: in_dim: %d, out_dim: %d\n", in_dim, out_dim);
	layer->createWeights(in_dim, out_dim);
  	layers.push_back(layer);
	printf("last layer in layers input size: %d\n", layers[layers.size()-1]->getInputDim());
}
**/

//------------------------------------------------------
void Model::add(Layer* layer_from, Layer* layer)
{
	printf("add(layer_from, layer)\n");
	// Layers should only require layer_size 
	layer->setInputDim(layer_from->getLayerSize());

  	layers.push_back(layer);

	int in_dim  = layer->getInputDim();
	int out_dim = layer->getOutputDim();
	printf("Model::add, layer dim: in_dim: %d, out_dim: %d\n", in_dim, out_dim);

	// Create weights
	//weights = WEIGHTS(out, in);
	//Weights* weights = new Weights(out_dim, in_dim);
	Connection* weights = new Connection(out_dim, in_dim);
	weights->initialize();
	weights_l.push_back(weights);

	// update prev and next lists in Layers class
	layer->prev.push_back(std::pair<Layer*,Connection*>(layer_from, weights));
	layer_from->next.push_back(std::pair<Layer*,Connection*>(layer, weights));
}
//----------------------------------------------------------------------
void Model::print(std::string msg /* "" */)
{
	printf("*** Model printout: ***\n");
    if (msg != "") printf("%s\n", msg.c_str());
	printf("name: %s\n", name.c_str());
	printf("stateful: %d\n", stateful);
	printf("learning_rate: %f\n", learning_rate);
	printf("return_sequences: %d\n", return_sequences);
	printf("print_verbose: %d\n", print_verbose);

  if (optimizer != NULL) 
	  optimizer->print();
  if (objective != NULL)
	  objective->print();

	if (print_verbose == false) return;

	for (int i=0; i < layers.size(); i++) {
		layers[i]->print();
	}
}
//----------------------------------------------------------------------
#if 0
VF2D_F Model::predict(VF2D_F x)
{
  	VF2D_F prod(x); //copy constructor, .n_rows);

	for (int l=1; l < layers.size(); l++) {
  		const WEIGHTS& wght= layers[l]->getWeights(); // between layer (l) and layer (l-1)

		// loop over batches
  		for (int b=0; b < x.n_rows; b++) { 
  			prod(b) = wght * prod(b); // prod(b) has different dimensions before and after the multiplication
		}
		layers[l]->setInputs(prod);

		// apply activation function
		prod = layers[l]->getActivation()(prod);
		layers[l]->setOutputs(prod);
	}
	return prod;
}
#endif
//----------------------------------------------------------------------
VF2D_F Model::predictNew(VF2D_F x)
{
  	VF2D_F prod(x); //copy constructor, .n_rows);

	Layer* cur_layer = layers[0];

	// start from the input. Follow the graph along connections with temporal=false
	for (int l=0; l < cur_layer->next.size(); l++) {
		Layer& layer = *cur_layer->next[l].first;
		//prod = 
		printf("layer name: %s\n", layer.getName().c_str());
		Connection* w = layer.next[l].second;
	}
	exit(0);
	
	for (int l=1; l < layers.size(); l++) {
  		//const WEIGHTS& wght= layers[l]->getWeights(); // between layer (l) and layer (l-1)
  		Connection* wghtc= layers[l]->prev[0].second; //getWeights(); // between layer (l) and layer (l-1)
		const WEIGHTS& wght = wghtc->getWeights();

		// loop over batches
  		for (int b=0; b < x.n_rows; b++) { 
  			prod(b) = wght * prod(b); // prod(b) has different dimensions before and after the multiplication
		}
		layers[l]->setInputs(prod);

		// apply activation function
		prod = layers[l]->getActivation()(prod);
		layers[l]->setOutputs(prod);
	}
	return prod;
}
//----------------------------------------------------------------------
// This was hastily decided on primarily as a means to construct feed forward
// results to begin implementing the backprop. Should be reevaluated
void Model::train(VF2D_F x, VF2D_F y, int batch_size /*=0*/, int nb_epochs /*=1*/) 
{
	if (batch_size == 0) { // Means no value for batch_size was passed into this function
    	batch_size = this->batch_size; // Use the current value stored in model
    	printf("model batch size: %d\n", batch_size);
		// resize x and y to handle different batch size
		assert(x.n_rows == batch_size && y.n_rows == batch_size);
	}

	VF2D_F pred = predictNew(x);
	VF1D_F loss = objective->computeError(y, pred);
	printf("loss.n_rows= ", loss.n_rows);
	loss.print("loss");
}
//----------------------------------------------------------------------
#if 0
void Model::initializeWeights(std::string initialization_type /* "uniform" */)
{
	int in_dim, out_dim;
	printf("inside initialize\n");
	printf("layers size= %d\n", layers.size());

	// NOTE: the loop starts from 1. Layer 0 (input layer) has no weights. 
	// This issue will disappear once weights act as connectors between layers. 
	for (int i=1; i < layers.size(); i++) {
		Layer* layer = layers[i];
		in_dim = layer->getInputDim(); //(i == 0) ? input_dim : layers[i-1]->getOutputDim();
		out_dim = layer->getLayerSize();
		printf("-- Model::initializeWeights, layer %d, in_dim, out_dim= %d, %d\n", i, in_dim, out_dim);
		layer->createWeights(in_dim, out_dim);
		layer->initializeWeights(initialization_type);
	}
}
#endif
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
