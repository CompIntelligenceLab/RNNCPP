#include <assert.h>
#include <stdio.h>
#include <iostream>
#include "typedefs.h"
#include "model.h"
#include "objective.h"
#include "connection.h"
#include "print_utils.h"

using namespace std;

Model::Model(std::string name /* "model" */) 
{
	this->name = name;
	learning_rate = 1.e-5;
	return_sequences = false;
	print_verbose = true;
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
    return_sequences(m.return_sequences), batch_size(m.batch_size),
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
void Model::addInputLayer(Layer* layer)
{
	input_layers.push_back(layer);
}
//----------------------------------------------------------------------
void Model::addOutputLayer(Layer* layer)
{
	output_layers.push_back(layer);
}
//----------------------------------------------------------------------
void Model::add(Layer* layer_from, Layer* layer)
{
	printf("add(layer_from, layer)\n");
	// Layers should only require layer_size 

	if (layer_from) {
		layer->setInputDim(layer_from->getLayerSize());
	} else {
		layer->setInputDim(layer->getInputDim());
	}

  	layers.push_back(layer);

	int in_dim  = layer->getInputDim();
	int out_dim = layer->getOutputDim();
	printf("Model::add, layer dim: in_dim: %d, out_dim: %d\n", in_dim, out_dim);

	// Create weights
	Connection* connection = new Connection(in_dim, out_dim);
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
	printf("layers size: %ld\n", layers.size());

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
	printf("=============================================================\n");
	printf("------ MODEL SUMMARY ------\n");
	std::string conn_type;
	char buf[80];

	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i];
		layer->printSummary("\n---- ");
		PAIRS_L prev = layer->prev;
		PAIRS_L next = layer->next;

		for (int p=0; p < prev.size(); p++) {
			Layer* pl = prev[p].first;
			Connection* pc = prev[p].second;
			conn_type = (pc->getTemporal()) ? "temporal" : "spatial";
			sprintf(buf, "   - prev[%d]: ", p);
			pl->printSummary(std::string(buf));
			pc->printSummary(std::string(buf));
		}
		for (int n=0; n < next.size(); n++) {
			Layer* nl = next[n].first;
			Connection* nc = next[n].second;
			conn_type = (nc->getTemporal()) ? "temporal" : "spatial";
			sprintf(buf, "   - next[%d]: ", n);
			nl->printSummary(buf);
			nc->printSummary(buf);
		}
	}
	printf("=============================================================\n");
}
//----------------------------------------------------------------------
VF2D_F Model::predict(VF2D_F x)
{
	// The network is assumed to have a single input. 
	// Only propagate through the spatial networks

  	VF2D_F prod(x); //copy constructor, .n_rows);

 	LAYERS layer_list;  
	LAYERS layers = getLayers();   // zeroth element is the input layer
    assert(layers.size() > 1);  // need at least an input layer connected to an output layer

    Layer* input_layer = layers[0];
    layer_list.push_back(input_layer);

	// going throught the above initializatino might not be necessary. However, if the topology or types of connections
	// change during training, then layer_list might change between training elements. So keep for generality. The cost 
	// is minimal compared to the cost of matrix-vector multiplication. 

	Layer* cur_layer = layers[0];

	while (true) {
		Layer* cur_layer = layer_list[0]; 
		int sz = cur_layer->next.size();
		if (sz == 0) {
			break;
		}
		printf("----------------\n");
		cur_layer->printSummary("current");
		for (int l=0; l < sz; l++) {
			Layer* nlayer = cur_layer->next[l].first;
			Connection* nconnection = cur_layer->next[l].second;
			WEIGHT& wght = nconnection->getWeight();
		    nconnection->printSummary("current");
		    nlayer->printSummary("current");
			layer_list.push_back(nlayer);

			if (nconnection->getTemporal()) {  // only consider spatial links (for now)
				printf("skip connection: %s\n", nconnection->getName().c_str());
				continue; 
			}

			for (int b=0; b < x.n_rows; b++) {
				prod(b) = wght * prod(b);  // not possible with cube since prod(b) on 
				                           //left and right of "=" have different dimensions
			}

			nlayer->setInputs(prod);

			// apply activation function
			prod = layers[l]->getActivation()(prod);

			nlayer->setOutputs(prod);

		}
		layer_list.erase(layer_list.begin());
	}
	
	return prod;
}
//----------------------------------------------------------------------
void Model::print(VF2D_F x, std::string msg /*""*/)
{
	cout << msg << ",  field size: " << x.n_rows << ", shape: " << x[0].n_rows << ", " << x[0].n_cols << endl;
} 
//----------------------------------------------------------------------
void Model::print(VF2D_F x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);
	cout << buf << ", field size: " << x.n_rows << ", shape: (" << x[0].n_rows << ", " << x[0].n_cols << ")" << endl;
}
//----------------------------------------------------------------------
void Model::print(VF2D x, std::string msg /*""*/)
{
	cout << msg << ", shape: " << x.n_rows << ", " << x.n_cols << endl;
} 
//----------------------------------------------------------------------
void Model::print(VF2D x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);
	cout << buf << ", shape: (" << x.n_rows << ", " << x.n_cols << ")" << endl;
}
//----------------------------------------------------------------------
void Model::print(VF1D x, std::string msg /*""*/)
{
	cout << msg << ", shape: " << x.n_rows << endl;
} 
//----------------------------------------------------------------------
void Model::print(VF1D x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);
	cout << buf << ", shape: (" << x.n_rows << endl;
}
//----------------------------------------------------------------------
VF2D_F Model::predictComplexMaybeWorks(VF2D_F x)  // for testing while Nathan works with predict
{
	// The network is assumed to have a single input. 
	// Only propagate through the spatial networks

	char buf[80];
  	VF2D_F prod(x); //copy constructor, .n_rows);

 	LAYERS layer_list;  
	// The first layer is input, the others can appear multiple times, depending on the network
	LAYERS layers = getLayers();   // zeroth element is the input layer
    assert(layers.size() > 1);  // need at least an input layer connected to an output layer

	for (int i=0; i < layers.size(); i++) {
		layers[i]->reset(); // reset inputs, outputs, and clock to zero
	}

    Layer* input_layer = layers[0];
	input_layer->setInputs(x);
    layer_list.push_back(input_layer);
	layer_list[0]->printName("layer_list[0]");

	// going throught the above initializatino might not be necessary. However, if the topology or types of connections
	// change during training, then layer_list might change between training elements. So keep for generality. The cost 
	// is minimal compared to the cost of matrix-vector multiplication. 

	Layer* cur_layer = layers[0];
	cur_layer->setInputs(prod); 
	cur_layer->setOutputs(prod); // inputs are same as outputs for an input layer

	while (true) {
		Layer* cur_layer = layer_list[0]; 
		int sz = cur_layer->next.size();
		if (sz == 0) {
			break;
		}

		// scan the layers downstream and connect to cur_layer

		for (int l=0; l < sz; l++) {
			// for each downstream layer, scan the uptream connections. 
			Connection* nconnection = cur_layer->next[l].second;

			if (nconnection->getTemporal()) {
				printf("skip forward propagation along temporal connections (%s)\n", 
				   nconnection->getName().c_str());
				continue;
			}

			Layer* nlayer = cur_layer->next[l].first;
			layer_list.push_back(nlayer);

			int csz = nlayer->prev.size();
			VF2D_F new_prod;

			for (int c=0; c < csz; c++) {
				Connection* pconnection = nlayer->prev[c].second;
				Layer*      player      = nlayer->prev[c].first;

				if (pconnection->getTemporal()) {  // only consider spatial links (for now)
					printf("skip temporal back connection: %s\n", pconnection->getName().c_str());
					continue; 
				}

				WEIGHT& wght = pconnection->getWeight();
				prod = player->getOutputs();  

				new_prod.set_size(prod.n_rows);
	
				for (int b=0; b < x.n_rows; b++) {
					new_prod(b) = wght * prod(b);  // not possible with cube since prod(b) on 
				                           	//left and right of "=" have different dimensions
				}

				nlayer->incrInputs(new_prod);
			}

			prod = nlayer->getActivation()(new_prod);
			nlayer->setOutputs(prod);
		}
		layer_list.erase(layer_list.begin());
	}

	// There should be a pointer to the output layer, or it is (for now), simply the last layer
	// in layers. 

	Layer* output_layer = layers[layers.size()-1]; // will this always work? Not for more general networks. 
	output_layer->getOutputs(); // not used
	
	return prod; 
}
//----------------------------------------------------------------------
VF2D_F Model::predictComplex(VF2D_F x)  // for testing while Nathan works with predict
{
	// The network is assumed to have a single input. 
	// Only propagate through the spatial networks

	printf("===================================\n");
	char buf[80];

  	VF2D_F prod(x); //copy constructor, .n_rows);

 	LAYERS layer_list;  
	// The first layer is input, the others can appear multiple times, depending on the network
	LAYERS layers = getLayers();   // zeroth element is the input layer
    assert(layers.size() > 1);  // need at least an input layer connected to an output layer

	for (int i=0; i < layers.size(); i++) {
		layers[i]->reset(); // reset inputs, outputs, and clock to zero
	}

    Layer* input_layer = layers[0];
	print(input_layer->getInputs(), "input_layer->getInputs()"); 
	print(x, "x"); 
	x.print("x");
	input_layer->getInputs().print("getInputs");

	input_layer->setInputs(x);
	input_layer->getInputs().print("getInputs");
    layer_list.push_back(input_layer);
	layer_list[0]->printName("layer_list[0]");

	print(layer_list[0]->getInputs(), "layer_list[0]->getInputs()");
	print(layers[0]->getInputs(), "layers[0]->getInputs()");
	layer_list[0]->getInputs().print("layer_list[0]");
	layers[0]->getInputs().print("layers[0]");

	// going throught the above initializatino might not be necessary. However, if the topology or types of connections
	// change during training, then layer_list might change between training elements. So keep for generality. The cost 
	// is minimal compared to the cost of matrix-vector multiplication. 

	Layer* cur_layer = layers[0];
	cur_layer->printName("layers[0]");
	prod.print("prod");
	cur_layer->setInputs(prod); 
	cur_layer->setOutputs(prod); // inputs are same as outputs for an input layer

	printf("-------------\n");
	//for (int i=0; i < layers.size(); i++) {
		//VF2D_F l = layers[i]->getInputs();
	//}

	while (true) {
		Layer* cur_layer = layer_list[0]; 
		int sz = cur_layer->next.size();
		if (sz == 0) {
			break;
		}

		// scan the layers downstream and connect to cur_layer
		printf("-----------------------------------------\n");
		//cur_layer->printSummary("cur_layer: layer_list[0]");

		for (int l=0; l < sz; l++) {
			printf("\n- - - - -\n ");
			// for each downstream layer, scan the uptream connections. 
			Connection* nconnection = cur_layer->next[l].second;
			if (nconnection->getTemporal()) {
				printf("skip forward propagation along temporal connections (%s)\n", 
				   nconnection->getName().c_str());
				continue;
			}
			Layer* nlayer = cur_layer->next[l].first;
			//nlayer->printSummary("*** nlayer downstream of cur_layer");
			layer_list.push_back(nlayer);

			int csz = nlayer->prev.size();
			VF2D_F new_prod;
			for (int c=0; c < csz; c++) {
			    printf("... connection %d\n", c);
				Connection* pconnection = nlayer->prev[c].second;
				Layer*      player      = nlayer->prev[c].first;
				pconnection->printSummary("connection upstream to nlayer");
				player->printSummary("layer upstream to nlayer");

				if (pconnection->getTemporal()) {  // only consider spatial links (for now)
					printf("skip temporal back connection: %s\n", pconnection->getName().c_str());
					continue; 
				}

				WEIGHT& wght = pconnection->getWeight();

				//if (cur_layer->clock == 0) {
					//prod =
				//} else {
					prod = player->getOutputs();  
					print(wght, "wght from prev connection");
					U::print(prod, "prod from prev layer");
					//prod.print("prod from input");
				//}

				//VF2D_F new_prod(prod.n_rows); // nb_batches = prod.n.rows
				new_prod.set_size(prod.n_rows);
			printf("===== before product\n");
	
				for (int b=0; b < x.n_rows; b++) {
					new_prod(b) = wght * prod(b);  // not possible with cube since prod(b) on 
				                           	//left and right of "=" have different dimensions
				}

				U::print(wght, "wght");
				U::print(prod, "prod");
				U::print(new_prod, "new_prod = wght*prod, incrInputs");
				nlayer->incrInputs(new_prod);
			}

			prod = nlayer->getActivation()(new_prod);
			printf(" --> nlayer->setOutputs(prod)");
			print(prod, " --> prod");
			nlayer->printSummary(" --> ");
			nlayer->setOutputs(prod);
			printf(" --> Finished processing nlayer: \n");
			nlayer->printSummary(" --> nlayer");
			print(prod, "final prod --> nlayer");
			printf("---------------------------\n");
		}
		layer_list[0]->print("\n*** Erase layer ***");
		layer_list.erase(layer_list.begin());
	}

	// There should be a pointer to the output layer, or it is (for now), simply the last layer
	// in layers. 

	Layer* output_layer = layers[layers.size()-1]; // will this always work? Not for more general networks. 
	output_layer->getOutputs();
	
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

	VF2D_F pred = predict(x);
	VF1D_F loss = objective->computeError(y, pred);
	loss.print("loss");
}
//----------------------------------------------------------------------
#if 0
void Model::backpropNathan(VF2D_F y, VF2D_F pred)
{
    std::vector<WEIGHT*> D; // All partial derivatives to be passed to optimizer
    // This only works for simple feed forward with one physical layer per
    // conceptual layer
    Layer* layer = layers[layers.size()-1]; // Start at the output layer
    layer->delta = objective->computeError(y, pred)(0);
    while(layer->prev != NULL) {
        Layer* prevLayer = layer->prev.first;
        Connection* prevCon = layer->prev.second;
        prevLayer->outputs.print("\n\nPREVLAYER OUTPUTS");
        prevCon->delta += (layer->delta)*(prevLayer->outputs(0).t().row(0)); // Update capital Delta
        D.push_back(&(prevCon->delta));
        prevLayer->delta = (prevCon->weight.t() * layer->delta) % prevLayer->activation->derivative(prevLayer->inputs(0).col(0));
        layer = prevLayer;
    }
}
#endif
//----------------------------------------------------------------------
void Model::backPropagation(VF2D_F y, VF2D_F pred)
{
    std::vector<WEIGHT*> D; // All partial derivatives to be passed to optimizer
    // This only works for simple feed forward with one physical layer per
    // conceptual layer

    Layer* layer = getOutputLayers()[0];   // Assume one output layer
    VF1D_F obj = objective->computeError(y, pred);
    layer->setDelta(obj);

	// Assume a linear sequence of layers (everything is per batch). 
	// weights are 2D matrices
	// Transposes of larger matrices are expensive
	// loss: VF1D_F  (seq_len)
	// dloss/dzL: VF2D_F (dim, seq_len)
	/*
	dloss/dwL   = dloss/dzL * fL'*  dzL/dwL = dloss/dzL * fL' * xL
	dxL  /dwLm1 = fLm1' * dzLm1/dwLm2 
	dxLm1/dwLm2 = fLm2' * dzLm2/dwLm3 

    dloss/dwL   = loss' * fL' * xL
    dloss/dwLm1 = loss' * fL' * fLm1' * xLm1
    dloss/dwLm2 = loss' * fL' * fLm1' * fLm2' * xLm2
    dloss/dwLm3 = loss' * fL' * fLm1' * fLm2' * fLm3' * xLm3
	---------------------
	del0 = loss' * fL'
	del1 = del0  * fLm1'
	del2 = del3  * fLm2'

    dL/dwL   = del0 * xL
    dL/dwLm1 = del1 * xLm1
    dL/dwLm2 = del2 * xLm2
	*/

    while (layer->prev[0].first) {
        Layer* prevLayer = layer->prev[0].first; // assume only one previous layer/connection pair
        Connection* prevCon = layer->prev[0].second;
        prevLayer->getOutputs().print("\n\nPREVLAYER OUTPUTS");
		VF2D out_t = prevLayer->getOutputs()[0].t();
		VF1D orow = out_t.row(0);
        VF1D delta_incr = (layer->getDelta()[0]) * orow; // Update capital Delta
        prevCon->incrDelta(delta_incr); 
        D.push_back(&prevCon->getDelta());  // SHOULD work. Does not. 
		VF2D wght_t = prevCon->getWeight().t();
		VF2D_F prev_inputs = prevLayer->getInputs(); // 
		VF2D prev_act = prevLayer->getActivation().derivative(prev_inputs)(0);
        VF1D pp = wght_t * layer->getDelta()[0];
		VF1D qq = prev_act;
		VF1D_F delta_f(1);
		delta_f(0) = pp % qq;
        prevLayer->setDelta(delta_f); 
        layer = prevLayer;
    }
}
//----------------------------------------------------------------------
