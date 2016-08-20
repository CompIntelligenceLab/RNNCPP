#include <typeinfo>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include "typedefs.h"
#include <list>
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
void Model::addLossLayer(Layer* layer)
{
	loss_layers.push_back(layer);
}
//----------------------------------------------------------------------
void Model::addProbeLayer(Layer* layer)
{
	probe_layers.push_back(layer);
}
//----------------------------------------------------------------------
void Model::add(Layer* layer_from, Layer* layer_to)
{
	printf("add(layer_from, layer)\n");
	// Layers should only require layer_size 

	if (layer_from) {
		layer_to->setInputDim(layer_from->getLayerSize());
	} else {
		layer_to->setInputDim(layer_to->getInputDim());
	}

  	layers.push_back(layer_to);

	int in_dim  = layer_to->getInputDim();
	int out_dim = layer_to->getOutputDim();
	printf("Model::add, layer_to dim: in_dim: %d, out_dim: %d\n", in_dim, out_dim);

	// Create weights
	Connection* connection = new Connection(in_dim, out_dim);
	connection->initialize();
	connections.push_back(connection);
	connection->from = layer_from;
	connection->to = layer_to;

	// update prev and next lists in Layers class
	if (layer_from) {
		layer_from->next.push_back(std::pair<Layer*, Connection*>(layer_to, connection));
		layer_to->prev.push_back(std::pair<Layer*, Connection*>(layer_from, connection));
		connection->which_lc = layer_to->prev.size()-1;
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

	// set all batch_sizes in layers to the model batch_layer
	for (int i=0; i < layers.size(); i++) {
		layers[i]->initVars(batch_size);
	}

	// input layer. Eventually, we might have multiple layers in the network. How to handle that?
	// A model can only have a single input layer (at this time). How to generalize? Not clear how to input data then. 
	// Probably need a list of input layers in a vector. In that case, it is not clear that layers[0] would be the input layer. 
	Layer* input_layer = getInputLayers()[0];
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

	while (layer_list.size()) {
		Layer* cur_layer = layer_list[0]; 
		//cur_layer->incr_Clock(); // Do not increment input layers. 
	                           	   // This will allow the input layer to also act as an output layer

		int sz = cur_layer->next.size();
		for (int l=0; l < sz; l++) {
			Layer* nlayer = cur_layer->next[l].first;
			Connection* nconnection = cur_layer->next[l].second;
			nlayer->printSummary("nlayer");

			//if (nlayer->getClock() > 0) {
			if (nconnection->getClock() > 0) {
				nconnection->setTemporal(true);
				nconnection->printSummary("checkIntegrity, setTemporal\n");
			}

		    //nlayer->incrClock();
			nconnection->incrClock();

			if (nlayer->getClock() == 1) { // only add a layer once to the list of layers to process
				layer_list.push_back(nlayer);  // layers left to process
			}
		}
		layer_list.erase(layer_list.begin());
	}
	return;
}
//----------------------------------------------------------------------
bool Model::areIncomingLayerConnectionsComplete(Layer* layer) 
{
	//printf("enter areIncomingLayerConnectionsComplete\n");
	//layer->printSummary("  ");

	int nb_arrivals = layer->prev.size();
	int nb_hits = layer->nb_hit;

	#if 0
	printf("  - nb_hits/prevsize= %d/%d\n", nb_hits, nb_arrivals);
	if (nb_hits == nb_arrivals) {
		layer->printSummary("  - INCOMING CONNECTIONS COMPLETE, ");
	} else {
		layer->printSummary("  - INCOMING CONNECTIONS NOT COMPLETE, ");
	}
	printf("exit areIncomingLayerConnectionsComplete, ");
	#endif

	return (nb_hits == nb_arrivals);
}
//----------------------------------------------------------------------
bool Model::isLayerComplete(Layer* layer) 
// Is the number of active connections = to the number of incoming connections
// Assume all connections are spatial
{
	layer->printSummary("enter isLayerComplete, ");

	// Check that all outgoing connections are "hit"

	int nb_departures = layer->next.size();
	printf(" - nb_departures= %d\n", nb_departures);

	int all_hits = 0;
	for (int i=0; i < nb_departures; i++) {
		all_hits += layer->next[i].second->hit;
	}

	#if 0
	if (all_hits < layer->next.size()) {
		printf("  - OUTGOING CONNECTIONS NOT COMPLETE, %d/%d\n", all_hits, layer->next.size());
	} else {
		printf("  - OUTGOING CONNECTIONS COMPLETE, %d/%d\n", all_hits, layer->next.size());
	}
	#endif

	//if (areIncomingLayerConnectionsComplete(layer)) printf("  LAYER COMPLETE");

	return areIncomingLayerConnectionsComplete(layer);
}
//----------------------------------------------------------------------
void Model::removeFromList(std::list<Layer*>& llist, Layer* cur_layer)
{
	llist.remove(cur_layer);
}
//----------------------------------------------------------------------
Layer* Model::checkPrevconnections(std::list<Layer*> llist)
// Find first layer in the list which has all its previous connections hit. 
{
	typedef std::list<Layer*>::iterator IT;
	IT it;

	//for (int i=0; i < llist.size(); i++) {
	for (it=llist.begin(); it != llist.end(); ++it){ 
		Layer* cur_layer = *it;
		int count = 0;
		for (int c=0; c < cur_layer->prev.size(); c++) {
			Connection* con = cur_layer->prev[c].second;
			count += con->hit;
		}
		if (count == cur_layer->prev.size()) {
		}
		//Connection* pc = cur_layer->prev[c].second;
		//Connection* nc = next[n].second;
	}
}
//----------------------------------------------------------------------
CONNECTIONS Model::connectionOrder()
{
	typedef std::list<Layer*>::iterator IT;
	IT it;

	// STL list to allow erase of elements via address
	std::list<Layer*> llist;
	std::vector<Layer*> completed_layers;
	//CONNECTIONS clist;
	Layer* cur_layer = getInputLayers()[0]; // assumes a single input layer
	cur_layer->nb_hit = 0;
	llist.push_back(cur_layer);

	int xcount = 0;

//bool Model::areIncomingLayerConnectionsComplete(Layer* layer) 
	while(llist.size()) {
		xcount++; 
		printf("******** entered while ******************************************\n");

		PAIRS_L next = cur_layer->next;

		// the following loop can only be entered if all the layer inputs are activated
		if (areIncomingLayerConnectionsComplete(cur_layer)) {
			for (int n=0; n < next.size(); n++) {
				Layer* nl      = next[n].first;
				nl->printSummary("xx nl");
				Connection* nc = next[n].second;
				clist.push_back(nc);
				nc->hit = 1;
				nl->nb_hit++; 
				llist.push_back(nl);
				if (isLayerComplete(cur_layer)) { // access error
					// these layers will be deleted from llist. They should never reappear
					// since that would imply a cycle in the network, which is prohibited.
					completed_layers.push_back(cur_layer);
				}
			}
			if (next.size() == 0) {
				if (isLayerComplete(cur_layer)) { // access error
					completed_layers.push_back(cur_layer);
				}
			}
		}
		printf("---------------------\n");

		llist.sort();
		llist.unique();

		// remove all layers that are "complete"
	    printf("before remove all complete layers, llist size: %d\n", llist.size());
		for (int i=0; i < completed_layers.size(); i++) {
			completed_layers[i]->printSummary("completed_layers");
			llist.remove(completed_layers[i]);
		}
	    printf("after remove all complete layers, llist size: %d\n", llist.size());
		//printf("before clear: completed layer size: %d\n", completed_layers.size());
		completed_layers.clear();
		for (it=llist.begin(); it != llist.end(); ++it) {
			(*it)->printSummary("llist");
		}
		//if (xcount == 2) {
			//exit(0);
		//}
		cur_layer = *llist.begin();
		//if (xcount == 7) exit(0);
	}
	//exit(0);

	//printf("list.size= %d\n", llist.size());
	//if (cur_layer->prev.size() == cur_layer->nb_hit) {
		//llist.remove(cur_layer);
	//}

	//cur_layer = *llist.begin();
	//printf("list.size= %d\n", llist.size());

	printf("Connection order\n");
	for (int c=0; c < clist.size(); c++) {
		Connection* con = clist[c];
		Layer* from = con->from;
		Layer* to = con->to;
		printf("con: %s, Layers: %s, %s\n", con->getName().c_str(), from->getName().c_str(), to->getName().c_str());
	}
    return clist;
}
//----------------------------------------------------------------------
CONNECTIONS Model::connectionOrderClean()
{
	typedef std::list<Layer*>::iterator IT;
	IT it;

	// STL list to allow erase of elements via address
	std::list<Layer*> llist;
	std::vector<Layer*> completed_layers;
	//CONNECTIONS clist;
	Layer* cur_layer = getInputLayers()[0]; // assumes a single input layer
	cur_layer->nb_hit = 0;
	llist.push_back(cur_layer);

	int xcount = 0;

//bool Model::areIncomingLayerConnectionsComplete(Layer* layer) 
	while(llist.size()) {
		xcount++; 
		//printf("******** entered while ******************************************\n");

		PAIRS_L next = cur_layer->next;

		// the following loop can only be entered if all the layer inputs are activated
		if (areIncomingLayerConnectionsComplete(cur_layer)) {
			for (int n=0; n < next.size(); n++) {
				Layer* nl      = next[n].first;
				//nl->printSummary("xx nl");
				Connection* nc = next[n].second;
				clist.push_back(nc);
				nc->hit = 1;
				nl->nb_hit++; 
				llist.push_back(nl);
				if (isLayerComplete(cur_layer)) { // access error
					// these layers will be deleted from llist. They should never reappear
					// since that would imply a cycle in the network, which is prohibited.
					completed_layers.push_back(cur_layer);
				}
			}
			if (next.size() == 0) {
				if (isLayerComplete(cur_layer)) { // access error
					completed_layers.push_back(cur_layer);
				}
			}
		}
		//printf("---------------------\n");

		llist.sort();
		llist.unique();

		// remove all layers that are "complete"
	    //printf("before remove all complete layers, llist size: %d\n", llist.size());
		for (int i=0; i < completed_layers.size(); i++) {
			completed_layers[i]->printSummary("completed_layers");
			llist.remove(completed_layers[i]);
		}
	    //printf("after remove all complete layers, llist size: %d\n", llist.size());
		completed_layers.clear();
		//for (it=llist.begin(); it != llist.end(); ++it) {
			//(*it)->printSummary("llist");
		//}
		cur_layer = *llist.begin();
	}

	printf("Connection order\n");
	for (int c=0; c < clist.size(); c++) {
		Connection* con = clist[c];
		Layer* from = con->from;
		Layer* to = con->to;
		printf("con: %s, Layers: %s, %s\n", con->getName().c_str(), from->getName().c_str(), to->getName().c_str());
	}
}
//----------------------------------------------------------------------
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
VF2D_F Model::predictViaConnections(VF2D_F x)
{
	printf("enter Model::predictViaConnections\n");
	Layer* input_layer = getInputLayers()[0];
	x.print("x");
	input_layer->setOutputs(x);
	printf("clist.size= %d\n", clist.size());

	for (int c=0; c < clist.size(); c++) {
		printf("---------- c= %d -------------\n", c);
		Connection* conn = clist[c];
		Layer* from = conn->from;
		Layer* to   = conn->to;
		VF2D_F& from_outputs = from->getOutputs();
		from_outputs.print("from_outputs");
		WEIGHT& wght = conn->getWeight();

		// SOMETHING NOT WORKING. LOOK AT OUTPUT

		VF2D_F prod(from_outputs.size());

		// matrix multiplication
		for (int b=0; b < from_outputs.size(); b++) {
			prod(b) = wght[b] * from_outputs[b];
		}

		VF2D_F& to_inputs    = to->layer_inputs[clist[c]->which_lc];

		printf("which_lc= %d\n", clist[c]->which_lc);
		to_inputs = prod;

		to_inputs.print("to_inputs");

		// w*outputs --> inputs(to), but which connection? 
		// One should store the connection number in the connection itself 
		// (although it is not satisfying)
	}
	exit(0);
	printf("exit Model::predictViaConnections\n");
	return x; // TEMPORARY
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
VF2D_F Model::predictComplexMaybeWorks(VF2D_F xf)  // for testing while Nathan works with predict
{
	// The network is assumed to have a single input. 
	// Only propagate through the spatial networks

	char buf[80];
  	VF2D_F prod(xf); //copy constructor, .n_rows);
  	xf.print("xf");

 	LAYERS layer_list;  
	// The first layer is input, the others can appear multiple times, depending on the network
	LAYERS layers = getLayers();   // zeroth element is the input layer
    assert(layers.size() > 1);  // need at least an input layer connected to an output layer

	// not necessarily a good idea if one wishes to maintain state for recursive nets. 
	// Works for feedfoward networks
	// Unless on maintains one input per connection, and resets the spatial connections. 
	for (int i=0; i < layers.size(); i++) {
		layers[i]->reset(); // reset inputs, outputs, and clock to zero
	}

    Layer* input_layer = layers[0];
	//printf("layer_inputs size: %d\n", input_layer->layer_inputs.size()); exit(0); // returned zero
    layer_list.push_back(input_layer);
	layer_list[0]->printName("layer_list[0]"); // input layer

	// going throught the above initializatin might not be necessary. However, if the topology or types of connections
	// change during training, then layer_list might change between training elements. So keep for generality. The cost 
	// is minimal compared to the cost of matrix-vector multiplication. 

	Layer* cur_layer = layer_list[0]; // input layer
	cur_layer->setOutputs(prod); // inputs are same as outputs for an input layer

	while (true) {
		Layer* cur_layer = layer_list[0]; 
		printf("New current layer: %s\n",  cur_layer->getName().c_str());
		int sz = cur_layer->next.size();
		if (sz == 0) {
			break;
		}

		// scan the layers downstream and connect to cur_layer

		printf("Scan downstream layers");
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

			printf("  Downstream layer/connection %s/%s", nlayer->getName().c_str(), nconnection->getName().c_str());
			int csz = nlayer->prev.size();
			VF2D_F new_prod;

			printf("     Scan upstream layers: ");

			for (int c=0; c < csz; c++) {
				Connection* pconnection = nlayer->prev[c].second;
				Layer*      player      = nlayer->prev[c].first;
				printf("  Upstream layer/connection %s/%s\n",  player->getName().c_str(), pconnection->getName().c_str());

				if (pconnection->getTemporal()) {  // only consider spatial links (for now)
					printf("skip temporal back connection: %s\n", pconnection->getName().c_str());
					continue; 
				}

				WEIGHT& wght = pconnection->getWeight();
				prod = player->getOutputs();  
				wght.print("wght, prev connection");
				prod.print("prod, prev layer");

				new_prod.set_size(prod.n_rows);
	
				for (int b=0; b < xf.n_rows; b++) {
				 	// several matrix/vector multiplications: innefficient since wght could stay in memory
					new_prod(b) = wght * prod(b);  // not possible with cube since prod(b) on 
				                           	//left and right of "=" have different dimensions
				}
				new_prod.print("wght*prod");
				nlayer->layer_inputs[c] = new_prod;
			}

			ZEROS(new_prod);

			// Sum all the layer inputs (if they have all arrived)
			for (int c=0; c < csz; c++) {
				for (int b=0; b < xf.n_rows; b++) {
					new_prod[b] += nlayer->layer_inputs[c][b];
				}
			}

			new_prod.print("** layer, (" + nlayer->getName() + ") input");
			prod = nlayer->getActivation()(new_prod);

			printf("activation: %s\n", nlayer->getActivation().getName().c_str());
			prod.print("** layer, (" + nlayer->getName() + ") output");
			nlayer->setOutputs(prod);
		}
		layer_list.erase(layer_list.begin());
	}

	// There should be a pointer to the output layer, or it is (for now), simply the last layer
	// in layers. 

	Layer* output_layer = layers[layers.size()-1]; // will this always work? Not for more general networks. 
	output_layer->getOutputs(); // not used

	prod.print("return value");
	printf("=== END PREDICT =====================\n");
	//exit(0);
	
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

			prod.print("** layer inputs");
			prod = nlayer->getActivation()(new_prod);
			printf(" --> nlayer->setOutputs(prod)");
			print(prod, " --> prod");
			nlayer->printSummary(" --> ");
			nlayer->setOutputs(prod);
			prod.print("** layer outputs");
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
	exit(0);
	
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
	objective->computeLoss(y, pred);
	VF1D_F loss = objective->getLoss();
	loss.print("loss");
}
//----------------------------------------------------------------------
void Model::backPropagation(VF2D_F exact, VF2D_F pred)
{
	// Not sure why required
    std::vector<WEIGHT*> D; // All partial derivatives to be passed to optimizer
    // This only works for simple feed forward with one physical layer per
    // conceptual layer

	//pred.print("xpred"); exit(0);

	// reset connection delta variables
	for (int i=0; i < connections.size(); i++) {
		printf("i=%d\n", i);
		connections[i]->resetDelta();
	}

    Layer* layer = getOutputLayers()[0];   // Assume one output layer
	exact.print("exact");
    objective->computeGradient(exact, pred);
    VF2D_F& grad = objective->getGradient();
    layer->setDelta(grad);  //DELTA: VF1D_F, but grad: VF2D_F (DELTA probably VF2D_F)
	pred.print("+++>pred"); // OK
	grad.print("+++> grad(objective), output layer"); //OK
	//VF2D grad10 = 10.*grad(0);
	//grad10.print("grad*10");
	printf("output layer\n");
	layer->printName("output layer");
	layer->getDelta().print("delta output layer");

	int nb_batch = grad.n_rows;
	VF2D delta_incr;
	VF2D_F delta_f(nb_batch);

    while (layer->prev.size()) {
		printf("*** inside while ***\n"); 
        Layer* prev_layer = layer->prev[0].first; // assume only one previous layer/connection pair (input layer)
		//prev_layer->printName(""); exit(0);
        Connection* prev_connection = layer->prev[0].second;
		VF2D_F& delta = layer->getDelta(); // grad from output layer
		//delta.print("delta"); exit(0); // grad
        prev_layer->printName("+++> prev layer"); // input
        //delta.print("+++> delta");  exit(0); // grad from outpu tlayer
		VF2D_F& out_t = prev_layer->getOutputs(); // (layer_size, seq_len)
		//out_t.print("+++> out_t, prev_layer"); exit(0); // xf from input layer (f = identity)

		for (int b=0; b < nb_batch; b++) {
        	delta_incr = delta[b] * out_t[b].t();    // EXPENSIVE
			delta_incr.print("+++> delta_incr, prev connection");
		}
        prev_connection->incrDelta(delta_incr); 
        delta_incr.print("delta_incr");
		prev_connection->printSummary("prev connection"); // input - dense01
		//exit(0);

		// Not sure about the purpose of D
        // D.push_back(&prev_connection->getDelta()); 

		const VF2D wght_t = prev_connection->getWeight().t();
		wght_t.print("+++> wght_t, prev connection");
		//const VF2D_F& prev_inputs = prev_layer->getInputs(); //  (layer_size, seq_len)
		const VF2D_F& prev_inputs = prev_layer->getOutputs(); //  (layer_size, seq_len)
		// input and output are the same. True. WHY IS IT NOT w*x? where is w? 
		prev_layer->printName("prev_layer");
		// I FEEL it should be prev-layer->getOutputs()
		wght_t.print("+++> prev_inputs, prev layer");
		const VF2D_F& prev_grad_act = prev_layer->getActivation().derivative(prev_inputs); // (layer_size, seq_len)
		prev_grad_act.print("+++> derivative, prev layer");

		for (int b=0; b < nb_batch; b++) {
        	const VF1D& pp = wght_t * delta[b];  // EXPENSIVE
			delta_f(b) = pp % prev_grad_act[b];
		}

        prev_layer->setDelta(delta_f); 
		delta_f.print("+++> delta_f, previous layer");
        layer = prev_layer;
    }
}
//----------------------------------------------------------------------
void Model::backPropagationComplex(VF2D_F exact, VF2D_F pred)
{
	// Not sure why required
    std::vector<WEIGHT*> D; // All partial derivatives to be passed to optimizer
    // This only works for simple feed forward with one physical layer per
    // conceptual layer

	// reset connection delta variables
	for (int i=0; i < connections.size(); i++) {
		printf("i=%d\n", i);
		connections[i]->resetDelta();
	}

    Layer* layer = getOutputLayers()[0];   // Assume one output layer
    objective->computeGradient(exact, pred);
    VF2D_F& grad = objective->getGradient();
    layer->setDelta(grad);  //DELTA: VF1D_F, but grad: VF2D_F (DELTA probably VF2D_F)

	int nb_batch = grad.n_rows;
	VF2D delta_incr;
	VF2D_F delta_f(nb_batch);

	// reset connections (must have a set of connections, so that there are no duplicates)

    while (layer->prev.size()) {
	  	for (int p=0; p < layer->prev.size(); p++) {
        	Layer* prev_layer = layer->prev[p].first; // assume only one previous layer/connection pair
        	Connection* prev_connection = layer->prev[p].second;

        	if (prev_connection->getTemporal()) {
				printf("skip temporal links\n");
				continue;
			}

			VF2D_F& delta = layer->getDelta(); // ==> dubious
			VF2D_F& out_t = prev_layer->getOutputs(); // (layer_size, seq_len)

			for (int b=0; b < nb_batch; b++) {
        		delta_incr = delta[b] * out_t[b].t();    // EXPENSIVE
        		prev_connection->incrDelta(delta_incr); 
			}

			// Not sure about the purpose of D
        	// D.push_back(&prev_connection->getDelta()); 

			const VF2D wght_t = prev_connection->getWeight().t();
			const VF2D_F& prev_inputs = prev_layer->getInputs(); //  (layer_size, seq_len)
			const VF2D_F& prev_grad_act = prev_layer->getActivation().derivative(prev_inputs); // (layer_size, seq_len)

			for (int b=0; b < nb_batch; b++) {
        		const VF1D& pp = wght_t * delta[b];  // EXPENSIVE
				delta_f(b) = pp % prev_grad_act[b];  // stored with layer
			}

        	printf("setDelta, layer: (%s)\n", prev_layer->getName().c_str());
        	prev_layer->setDelta(delta_f);    // setDelta or incrDelta (when a layer is hit twice?)
        	layer = prev_layer;
	  	}
    }
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
