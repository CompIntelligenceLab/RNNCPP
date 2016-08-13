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
void Model::add(Layer* layer_from, Layer* layer)
{
	printf("add(layer_from, layer)\n");
	// Layers should only require layer_size 

	if (layer_from) {
		layer->setInputDim(layer_from->getLayerSize());
	} else {
		layer->setInputDim(this->input_dim);
	}

  	layers.push_back(layer);

	int in_dim  = layer->getInputDim();
	int out_dim = layer->getOutputDim();
	printf("Model::add, layer dim: in_dim: %d, out_dim: %d\n", in_dim, out_dim);

	// Create weights
	Connection* connection = new Connection(out_dim, in_dim);
	connection->initialize();
	connections.push_back(connection);
	connection->from = layer_from;
	connection->to = layer;

	// update prev and next lists in Layers class
	if (layer_from) {
		layer_from->next.push_back(std::pair<Layer*, Connection*>(layer, connection));
		layer->prev.push_back(std::pair<Layer*, Connection*>(layer_from, connection));
	}
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
void Model::checkIntegrity()
{
	LAYERS layer_list;  // should be a linked list
	LAYERS layers = getLayers();
	assert(layers.size() > 1);  // need at least an input layer connected to an output layer
	printf("layers size: %d\n", layers.size());

	// input layer. Eventually, we might have multiple layers in the network. How to handle that?
	// A model can only have a single input layer (at this time). How to generalize? Not clear how to input data then. 
	// Probably need a list of input layers in a vector. In that case, it is not clear that layers[0] would be the input layer. 
	Layer* input_layer = layers[0];
	layer_list.push_back(input_layer); 
	checkIntegrity(layer_list);

	// A recursive solution would be better. 
}

//------------------------------------------------------------
void Model::checkIntegrity(LAYERS& layer_list)
{
/*
   starting with first layer, connect to layer->next layers. Set their clocks to 1. 
   For each of these next layers l, connect to l->next layers. Set their clocks to 2. 
   - if the clock of l->next layers is not zero, change connection to temporal. Continue
   until no more connections to process. 
   - one should also set the connection's clock if used. 
   - need routines: model.resetLayers(), model.resetConnections() // set clock=0 for connections and layers
*/
	// input layer. Eventually, we might have multiple layers in the network. How to handle that?

	while (true) {
		Layer* cur_layer = layer_list[0]; 
		//cur_layer->incr_Clock(); // Do not increment input layers. 
	                           	   // This will allow the input layer to also act as an output layer

		int sz = cur_layer->next.size();
		for (int l=0; l < sz; l++) {
			Layer* nlayer = cur_layer->next[l].first;
			Connection* nconnection = cur_layer->next[l].second;

			if (nlayer->getClock() > 0) {
				nconnection->setTemporal(true);
			}

			nlayer->incrClock();
			nconnection->incrClock();

			if (nlayer->getClock() == 1) { // only add a layer once to the list of layers to process
				layer_list.push_back(nlayer);  // layers left to process
			}
		}
		layer_list.erase(layer_list.begin());
		if (layer_list.size() == 0) {
			return;
		}
	}
}
//----------------------------------------------------------------------
void Model::printSummary()
{
	printf("\n\n ---- MODEL SUMMARY ----\n\n");
	std::string conn_type;

	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i];
		printf("layer: %s\n", layer->getName().c_str());
		PAIRS_L prev = layer->prev;
		PAIRS_L next = layer->next;
		printf("  prev: \n");
		for (int p=0; p < prev.size(); p++) {
			Layer* pl = prev[p].first;
			Connection* pc = prev[p].second;
			conn_type = (pc->getTemporal()) ? "temporal" : "spatial";
			printf("    %s, %s\n", pl->getName().c_str(), conn_type.c_str());
		}
		printf("  next: \n");
		for (int n=0; n < next.size(); n++) {
			Layer* nl = next[n].first;
			Connection* nc = next[n].second;
			conn_type = (nc->getTemporal()) ? "temporal" : "spatial";
			printf("    %s, %s\n", nl->getName().c_str(), conn_type.c_str());
		}
	}
}
//----------------------------------------------------------------------
VF2D_F Model::predictNew(VF2D_F x)
{
	// The network is assumed to have a single input. 
	// Only propagate through the spatial networks
	//printf("   nlayer layer_size: %d\n", getLayers()[0]->getLayerSize());
	//exit(0);

  	VF2D_F prod(x); //copy constructor, .n_rows);

 	LAYERS layer_list;  
	LAYERS layers = getLayers();   // zeroth element is the input layer
    assert(layers.size() > 1);  // need at least an input layer connected to an output layer

    Layer* input_layer = layers[0];
    layer_list.push_back(input_layer);
	printf("   input layer_size: %d\n", layer_list[0]->getLayerSize());

	// going throught the above initializatino might not be necessary. However, if the topology or types of connections
	// change during training, then layer_list might change between training elements. So keep for generality. The cost 
	// is minimal compared to the cost of matrix-vector multiplication. 

	Layer* cur_layer = layers[0];

	while (true) {
		Layer* cur_layer = layer_list[0]; 
		int sz = cur_layer->next.size();
		for (int l=0; l < sz; l++) {
			printf("l= %d, sz= %d\n", l, sz);
			Layer* nlayer = cur_layer->next[l].first;
			Connection* nconnection = cur_layer->next[l].second;
			WEIGHT& wght = nconnection->getWeight();
			layer_list.push_back(nlayer);

			if (nconnection->getTemporal()) {  // only consider spatial links (for now)
				printf("skip connection: %s\n", nconnection->getName().c_str());
				continue; 
			}

			printf("-- calculate w*x: \n");
			printf("   layer: %s\n", nlayer->getName().c_str());
			printf("   cur_layer layer_size: %d\n", cur_layer->getLayerSize());
			printf("   nlayer layer_size: %d\n", nlayer->getLayerSize());
			wght.print("    wght");
			prod[0].print("    prod[0]");
			for (int b=0; b < x.n_rows; b++) {
				prod(b) = wght * prod(b);  // not possible with cube since prod(b) on 
				                           //left and right of "=" have different dimensions
			}

			printf("result of multplication:\n");
			prod(0).print("prod(result)");

			nlayer->setInputs(prod);

			// apply activation function
			prod = layers[l]->getActivation()(prod);

			nlayer->setOutputs(prod);

		}
		printf("erase layer %s\n", layer_list[0]->getName().c_str());
		layer_list.erase(layer_list.begin());
		printf("gordon\n");
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
