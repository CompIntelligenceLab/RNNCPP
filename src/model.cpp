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
	learning_rate = 0.001;
	return_sequences = false;

	connections.resize(0);
	order_eval.resize(0);
	//clist.resize(0);
	//clist_temporal.resize(0);
	x_in_history.resize(0);
	x_out_history.resize(0);
	loss_history.resize(0);
	weights_to_print.resize(0);  // need better approach. For large networks, we

	print_verbose = true;
	printf("Model constructor (%s)\n", this->name.c_str());
	optimizer = new RMSProp();
	objective = new MeanSquareError();
	//objective = NULL; // I believe this should just be a string or something specifying
               // the name of the objective function we would like to use
			   // WHY a string? (Gordon)
	batch_size = 1; // batch size of 1 is equivalent to online learning
	nb_epochs = 10; // Just using the value that Keras uses for now
    stateful = false;  // all layers are stateful or not stateful
	seq_len = 1; // should be equivalent to feedforward (no time to unroll)
	initialization_type = "xavier";  // can also choose Gaussian
	init_weight_rms = .1;  // default
}
//----------------------------------------------------------------------
Model::~Model()
{
	printf("Model destructor (%s)\n", name.c_str());

	for (int i=0; i < layers.size(); i++) {
		if (layers[i]) {delete layers[i];} 
	}
	layers.resize(0);

	for (int i=0; i < activations.size(); i++) {
		if (activations[i]) {delete activations[i];}
	}
	activations.resize(0);

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
	//layer->setActivation(new Identity());
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
void Model::add(Layer* layer_from, Layer* layer_to, bool is_temporal, std::string conn_type /*"all-all"*/)
{
	// I AM ADDING A Layer if the layer doesn't already exit. Otherwise, I am adding a connection

	printf("add(layer_from, layer, temporal)\n");
	// Layers should only require layer_size 

	if (layer_from) {
		layer_to->setInputDim(layer_from->getLayerSize());
	} else {
		layer_to->setInputDim(layer_to->getInputDim());
	}

	// Only add the layer if it is not already in the layer list
	bool in_list = false;
	for (int l=0; l < layers.size(); l++) {
		if (layer_to == layers[l]) {
			in_list = true;
		}
	}
	if (in_list == false) {
  		layers.push_back(layer_to);
		layer_to->setNbBatch(nb_batch); // check
		layer_to->setSeqLen(seq_len); // check
		printf("layers: nb_batch= %d\n", nb_batch);
	}

	int in_dim  = layer_to->getInputDim();
	int out_dim = layer_to->getOutputDim();
	printf("Model::add, layer_to dim: in_dim: %d, out_dim: %d\n", in_dim, out_dim);

	// Create weights
	// Later on, we might create different kinds of connection. This would require a rework of 
	// the interface. 
	Connection* connection = Connection::ConnectionFactory(in_dim, out_dim, conn_type);
	printf("add temporal connection= %ld\n", connection);
	connection->setTemporal(is_temporal);  // not in other add() routine

	// connections must contain all connections (to allow retrieval of the connection given the layers)
	//if (connection->getTemporal() == false) {
		connections.push_back(connection);
	//}
	connection->from = layer_from;
	connection->to = layer_to;
	connection->setWeightRMS(init_weight_rms); 
	connection->initialize(initialization_type); // must be called after layer_to definition

	// update prev and next lists in Layers class
	if (connection->getTemporal() == false) {
		if (layer_from) layer_from->next.push_back(std::pair<Layer*, Connection*>(layer_to, connection));
		if (layer_to) layer_to->prev.push_back(std::pair<Layer*, Connection*>(layer_from, connection));
		connection->which_lc = layer_to->prev.size()-1;
		printf("    connection->which_lc= %d\n", connection->which_lc);
	}

	if (connection->getTemporal() == true) {
		clist_temporal.push_back(connection);
		if (layer_from) layer_from->next_temporal.push_back(std::pair<Layer*, Connection*>(layer_to, connection));
		if (layer_to) layer_to->prev_temporal.push_back(std::pair<Layer*, Connection*>(layer_from, connection));
	}
}
//----------------------------------------------------------------------
void Model::add(Layer* layer_from, Layer* layer_to, std::string conn_type /*"all-all"*/)
{
	// I AM ADDING A Layer if the layer doesn't already exit. Otherwise, I am adding a connection

	printf("add(layer_from, layer)\n");
	// Layers should only require layer_size 

	if (layer_from) {
		layer_to->setInputDim(layer_from->getLayerSize());
	} else {
		layer_to->setInputDim(layer_to->getInputDim());
	}

	// Only add the layer if it is not already in the layer list
	bool in_list = false;
	for (int l=0; l < layers.size(); l++) {
		if (layer_to == layers[l]) {
			in_list = true;
		}
	}
	if (in_list == false) {
  		layers.push_back(layer_to);
		layer_to->setNbBatch(nb_batch); // check
		layer_to->setSeqLen(seq_len); // check
		printf("layers: nb_batch= %d\n", nb_batch);
	}

	int in_dim  = layer_to->getInputDim();
	int out_dim = layer_to->getOutputDim();
	printf("Model::add, layer_to dim: in_dim: %d, out_dim: %d\n", in_dim, out_dim);

	// Create weights
	// Later on, we might create different kinds of connection. This would require a rework of 
	// the interface. 
	Connection* connection = Connection::ConnectionFactory(in_dim, out_dim, conn_type);
	printf("add connection= %ld\n", connection);
	//if (connection->getTemporal() == false) {
		connections.push_back(connection);
	//}
	connection->from = layer_from;
	connection->to = layer_to;
	connection->setWeightRMS(init_weight_rms); 
	connection->initialize(initialization_type); // must be called after layer_to definition

	// update prev and next lists in Layers class
	if (layer_from && connection->getTemporal() == false) {
		layer_from->next.push_back(std::pair<Layer*, Connection*>(layer_to, connection));
		layer_to->prev.push_back(std::pair<Layer*, Connection*>(layer_from, connection));
		connection->which_lc = layer_to->prev.size()-1;
		printf("    connection->which_lc= %d\n", connection->which_lc);
	}
}
//----------------------------------------------------------------------
void Model::print(std::string msg /* "" */)
{
	printf("----BEGIN MODEL PRINTOUT -----------------\n");
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
	printf("----END MODEL PRINTOUT -----------------\n");
}
//----------------------------------------------------------------------
void Model::checkIntegrity()
{
	LAYERS layer_list;  // should be a linked list
	LAYERS layers = getLayers();

	//assert(layers.size() > 1);  // need at least an input layer connected to an output layer
	if (layers.size() < 2) return;

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
   Starting with first layer, connect to layer->next layers. Set their clocks to 1. 
   For each of these next layers l, connect to l->next layers. Set their clocks to 2. 
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
			//nlayer->printSummary("nlayer");

			if (nconnection->getClock() > 0) {
				nconnection->setTemporal(true);
				nconnection->printSummary("checkIntegrity, setTemporal\n");
			}

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

	int nb_arrivals = layer->prev.size();
	int nb_hits = layer->nb_hit;

	return (nb_hits == nb_arrivals);
}
//----------------------------------------------------------------------
bool Model::isLayerComplete(Layer* layer) 
// Is the number of active connections = to the number of incoming connections
// Assume all connections are spatial
{
	// Check that all outgoing connections are "hit"

	int nb_departures = layer->next.size();

	int all_hits = 0;
	for (int i=0; i < nb_departures; i++) {
		all_hits += layer->next[i].second->hit;
	}

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

	for (it=llist.begin(); it != llist.end(); ++it){ 
		Layer* cur_layer = *it;
		int count = 0;
		for (int c=0; c < cur_layer->prev.size(); c++) {
			Connection* con = cur_layer->prev[c].second;
			count += con->hit;
		}
		if (count == cur_layer->prev.size()) {
		}
	}
}
//----------------------------------------------------------------------
void Model::connectionOrderClean()
{
	typedef std::list<Layer*>::iterator IT;
	IT it;

	// STL list to allow erase of elements via address
	std::list<Layer*> llist;
	std::list<Layer*>::iterator llist_iter;
	std::vector<Layer*> completed_layers;
	Layer* cur_layer = getInputLayers()[0]; // assumes a single input layer
	cur_layer->nb_hit = 0;
	llist.push_back(cur_layer);

	// set correct batch size in layers
	for (int l=0; l < layers.size(); l++) {
		layers[l]->setNbBatch(nb_batch);
		layers[l]->setSeqLen(seq_len);
	}

	int xcount = 0;

	while(llist.size()) {
		xcount++; 

		PAIRS_L next = cur_layer->next;

		// the following loop can only be entered if all the layer inputs are activated
		if (areIncomingLayerConnectionsComplete(cur_layer)) {
			cur_layer->printSummary(); // --------
			printf("*complete\n");     // --------
			for (int n=0; n < next.size(); n++) {
				Layer* nl      = next[n].first;
				Connection* nc = next[n].second;
				clist.push_back(nc);
				nc->hit = 1;
				nl->nb_hit++; 
				llist.push_back(nl);
				if (isLayerComplete(cur_layer)) { 
					// these layers will be deleted from llist. They should never reappear
					// since that would imply a cycle in the network, which is prohibited.
					completed_layers.push_back(cur_layer);
				}
			}
			if (next.size() == 0) {
				if (isLayerComplete(cur_layer)) {
					cur_layer->printSummary(); // --------
					printf("completed layers push_back\n");  // ------- not reached
					completed_layers.push_back(cur_layer);
				}
			}
		}

		llist.sort();
		llist.unique();

		// remove all layers that are "complete"
		for (int i=0; i < completed_layers.size(); i++) {
			llist.remove(completed_layers[i]);
		}
		completed_layers.clear();
		cur_layer = *llist.begin();
	}

	for (int c=0; c < clist.size(); c++) {
		Connection* con = clist[c];
		Layer* from = con->from;
		Layer* to = con->to;
	}

	for (int l=0; l < input_layers.size(); l++) {
		if (input_layers[l]->prev.size() == 0) {
			input_layers[l]->prev.resize(1);
		}
	}

	// Assign memory
	for (int l=0; l < layers.size(); l++) {
		layers[l]->layer_inputs.resize(layers[l]->prev.size());
		layers[l]->layer_deltas.resize(layers[l]->prev.size());

		for (int i=0; i < layers[l]->layer_inputs.size(); i++) {
			layers[l]->layer_inputs[i] = VF2D_F(nb_batch);
			int input_dim = layers[l]->getLayerSize();
			int seq_len   = layers[l]->getSeqLen();

			for (int b=0; b < nb_batch; b++) {
				layers[l]->layer_inputs[i](b) = VF2D(input_dim, seq_len);
			}
		}

		// layer_deltas are unsused at this time
	}
	//printf("============== EXIT connectionOrderClean =================\n");
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void Model::printSummary()
{
	printf("\n-------------------------------------------------------------\n");
	printf("------ MODEL SUMMARY ----------------------------------------\n");
	std::string conn_type;
	char buf[80];

	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i];
		layer->printSummary("\n---- ");
		PAIRS_L prev = layer->prev;
		PAIRS_L next = layer->next;
		//Activation& activation = layer->getActivation();


		for (int p=0; p < prev.size(); p++) {
			Layer* pl = prev[p].first;
			Connection* pc = prev[p].second;
	printf("pc1= %ld\n", pc);
			conn_type = (pc->getTemporal()) ? "temporal" : "spatial";
			sprintf(buf, "   - prev[%d]: ", p);
			pl->printSummary(std::string(buf));
			pc->printSummary(std::string(buf));
			printf("\n");
		}
		for (int n=0; n < next.size(); n++) {
			Layer* nl = next[n].first;
			Connection* nc = next[n].second;
	printf("nc1= %ld\n", nc);
			conn_type = (nc->getTemporal()) ? "temporal" : "spatial";
			sprintf(buf, "   - next[%d]: ", n);
			nl->printSummary(buf);
			nc->printSummary(buf);
			printf("\n");
		}

		prev = layer->prev_temporal;
		next = layer->next_temporal;

		for (int p=0; p < prev.size(); p++) {
			Layer* pl = prev[p].first;
			Connection* pc = prev[p].second;
	printf("pc2= %ld\n", pc);
			conn_type = (pc->getTemporal()) ? "temporal" : "spatial";
			sprintf(buf, "   - prev[%d]: ", p);
			pl->printSummary(std::string(buf));
			pc->printSummary(std::string(buf));
			printf("\n");
		}
		for (int n=0; n < next.size(); n++) {
			Layer* nl = next[n].first;
			Connection* nc = next[n].second;
	printf("nc2= %ld\n", nc);
			conn_type = (nc->getTemporal()) ? "temporal" : "spatial";
			sprintf(buf, "   - next[%d]: ", n);
			nl->printSummary(buf);
			nc->printSummary(buf);
			printf("\n");
		}
	}
	printf("\n------ END MODEL SUMMARY ------------------------------------\n\n");
}
//----------------------------------------------------------------------
VF2D_F Model::predictViaConnectionsBias(VF2D_F x)
{
	if (stateful == false) {
		resetState(); 
		printf("reset state GE\n");
	}

	//printf("****************** ENTER predictViaConnections ***************\n");


	VF2D_F prod(x.n_rows);

	Layer* input_layer = getInputLayers()[0];
	input_layer->layer_inputs[0] = x; 
	input_layer->setOutputs(x);  // although input layer "could" have a nonlinear activation function (maybe)

 	for (int t=0; t < seq_len; t++) {  // CHECK LOOP INDEX LIMIT
		for (int l=0; l < layers.size(); l++) {
			layers[l]->nb_hit = 0;
		}

		// go through all the layers and update the temporal connections
		// On the first pass, connections are empty
		// TEMPORARY: should be #if 0
		#if 0    
		for (int l=0; l < layers.size(); l++) {
			layers[l]->forwardLoops(t-1);    // does not change with biases (empty functions it seems)
		}
		#endif

		// update all other temporal connections coming into the layers (arbitrary order, I think)
		// ...........
		// go through all the layers and update the temporal connections
		// On the first pass, connections are empty
		// Added Nov. 14. 
		// Temporary: should be #if 1
		#if 1
		//printf("clist_temporal size: %d\n", clist_temporal.size());
		// I FORGOT TO PUT RECURRENT LINKS WITH clist_temporal
		for (int c=0; c < clist_temporal.size(); c++) {
			Connection* conn = clist_temporal[c];
			//conn->printSummary(""); conn->getWeight().print("temporal weights");
			Layer* to_layer = conn->to;
			to_layer->forwardLoops(conn, t-1);
		}
		#endif
		
		for (int c=0; c < clist.size(); c++) {
			Connection* conn  = clist[c];
			Layer* to_layer   = conn->to;
			to_layer->processOutputDataFromPreviousLayer(conn, prod, t);
		}
 	}


	// update all other temporal connections coming into the layers (arbitrary order, I think)
	// ...........

	//printf("before last for in predict\n");
	//for (int c=0; c < clist.size(); c++) { // }
	for (int c=0; c < clist_temporal.size(); c++) {  // WILL THIS CHANGE eq3 test? 
		Connection* conn  = clist_temporal[c];
		Layer* to_layer   = conn->to;
		to_layer->forwardLoops(conn, seq_len-1, 0);
	}

	return prod;
}
//----------------------------------------------------------------------
void Model::predictViaConnectionsBias(VF2D_F x, VF2D_F& prod)
{
	if (stateful == false) {
		resetState(); 
	}

	printf("****************** ENTER predictViaConnections ***************\n");

	Layer* input_layer = getInputLayers()[0];
	input_layer->layer_inputs[0] = x; 
	input_layer->setOutputs(x);  // although input layer "could" have a nonlinear activation function (maybe)

	// for t = 0, somehow take previous state into account

 	for (int t=0; t < (seq_len); t++) {  // CHECK LOOP INDEX LIMIT
		for (int l=0; l < layers.size(); l++) {
			layers[l]->nb_hit = 0;
		}

		// go through all the layers and update the temporal connections
		// On the first pass, connections are empty
		// TEMPORARY: should be #if 0
		#if 0    
		for (int l=0; l < layers.size(); l++) {
			layers[l]->forwardLoops(t-1);    // does not change with biases (empty functions it seems)
		}
		#endif

		// update all other temporal connections coming into the layers (arbitrary order, I think)
		// ...........
		// go through all the layers and update the temporal connections
		// On the first pass, connections are empty
		// Added Nov. 14. 
		// Temporary: should be #if 1
		#if 1
		//printf("clist_temporal size: %d\n", clist_temporal.size());
		// I FORGOT TO PUT RECURRENT LINKS WITH clist_temporal
		for (int c=0; c < clist_temporal.size(); c++) {
			Connection* conn = clist_temporal[c];
			//conn->printSummary(""); conn->getWeight().raw_print(arma::cout, "temporal weights");
			Layer* to_layer = conn->to;
			to_layer->forwardLoops(conn, t-1);
		}
		#endif
		
		for (int c=0; c < clist.size(); c++) {
			Connection* conn  = clist[c];
			//conn->printSummary(""); conn->getWeight().raw_print(arma::cout, "spatial weights");
			Layer* to_layer   = conn->to;
			// Responsible for memory leak
			to_layer->processOutputDataFromPreviousLayer(conn, prod, t);
		}
 	}


	// update all other temporal connections coming into the layers (arbitrary order, I think)
	// ...........

	for (int c=0; c < clist_temporal.size(); c++) {  // WILL THIS CHANGE eq3 test? 
		Connection* conn  = clist_temporal[c];
		Layer* to_layer   = conn->to;
		to_layer->forwardLoops(conn, seq_len-1, 0); // IS THIS OK? 
	}

	for (int l=0; l < layers.size(); l++) {
		printf("---------------\n");
		layers[l]->printSummary();
		layers[l]->getInputs()[0].raw_print(cout, "in prediction, inputs");
		layers[l]->getOutputs()[0].raw_print(cout, "in prediction, outputs");
	}

	//printf("QUIT AT END of methods::predict...()\n"); exit(0);

	//prod[0].raw_print(cout, "prod"); exit(0);
	// I should do prod.reset(), but cannot or else I cannot return the data. 
	// Therefore, pass prod via argument. 
	return;
}
//----------------------------------------------------------------------
// Treat a single batch. x has dimesions [batch][input_dim, seq_len]
// exact has dimesions [batch][last_layer_size, seq_len]
// batch_size is encoded in the dimensions of "x"
// I might create another function where x and exact does not include the batchsize, and 
// the arrays would be computed inside the function
// results to begin implementing the backprop. Should be reevaluated

void Model::trainOneBatch(VF2D_F& x, VF2D_F& exact)
{
	// MUST REWRITE THIS PROPERLY
	// DEAL WITH BATCH and SEQUENCES CORRECTLY
	// FOR NOW, ASSUME BATCH=1
	//printf("ENTER trainOneBatch ******************************\n");
	cout.precision(11);

	if (stateful == false) {
		resetState();
	}

	//VF2D_F x(1); x[0] = x_;
	// WRONG IN GENERAL. Only good for batch == 1
	//VF2D_F exact(1); exact[0] = exact_;
	//U::print(x);

	VF2D_F pred; //new
	printf("ENTER predict GE\n");
	predictViaConnectionsBias(x, pred); // new
	printf("EXIT predict GE\n");

	// PRINT INPUTS AND OUTPUTS TO NETWORK (TEMP)
	printf("\nINPUTS AND OUTPUTS TO NETWORK\n");
	for (int l=0; l < layers.size(); l++) {
		if (l != 0) {
			layers[l]->printSummary("");
			layers[l]->getInputs()(0).raw_print(arma::cout, "layer inputs");
		}
	}

	//TEMPORARY
	clist[0]->getWeight().raw_print(cout, "Weights: input-d1");
	clist[1]->getWeight().raw_print(cout, "Weights: d1-d2");
	clist_temporal[0]->getWeight().raw_print(cout, "Weights: d1-d1");

	objective->computeLoss(exact, pred);

	const LOSS& loss = objective->getLoss();
	REAL rloss = arma::sum(loss[0]);
	printf("rloss= %21.14f\n", rloss);

	// If save loss, ...
	//loss_history.push_back(loss); // slight leak (should print out every n iterations
	for (int l=0; l < layers.size(); l++) {
		layers[l]->getPreviousState()(0).raw_print(arma::cout, "before backprop, previous state");
	}

//	backPropagationViaConnectionsRecursion(exact, pred);

	for (int l=0; l < layers.size(); l++) {
		// does not appear to be necessary for prediction to work properly. 
		// WHY? Because of forward Loops? 
		layers[l]->setPreviousState();
	}

	parameterUpdate();

	pred.reset(); // handle memory leaks
}
//----------------------------------------------------------------------
void Model::trainOneBatch(VF2D x_, VF2D exact_)
{
	// MUST REWRITE THIS PROPERLY
	// DEAL WITH BATCH and SEQUENCES CORRECTLY
	// FOR NOW, ASSUME BATCH=1
	//printf("ENTER trainOneBatch ******************************\n");
	cout.precision(11);

	//printf("stateful: %d\n", stateful); //exit(0);
	if (stateful == false) {
		resetState();
	}

	VF2D_F x(1); x[0] = x_;
	// WRONG IN GENERAL. Only good for batch == 1
	VF2D_F exact(1); exact[0] = exact_;

	VF2D_F pred = predictViaConnectionsBias(x);
	//pred[0].raw_print(cout, "pred");
	//exact[0].raw_print(cout, "exact");
	objective->computeLoss(exact, pred);

	const LOSS& loss = objective->getLoss();
	//loss_history.push_back(loss);

	return;
	backPropagationViaConnectionsRecursion(exact, pred);

	// Save Previous State
	for (int l=0; l < layers.size(); l++) {
		layers[l]->setPreviousState();
	}

	parameterUpdate();
}
//----------------------------------------------------------------------
// This was hastily decided on primarily as a means to construct feed forward
// results to begin implementing the backprop. Should be reevaluated
void Model::train(VF2D_F x, VF2D_F exact, int batch_size /*=0*/, int nb_epochs /*=1*/) 
{
	if (batch_size == 0) { // Means no value for batch_size was passed into this function
    	batch_size = this->batch_size; // Use the current value stored in model
    	//printf("model batch size: %d\n", batch_size);
		// resize x and exact to handle different batch size
		assert(x.n_rows == batch_size && exact.n_rows == batch_size);
	}

	// MUST REWRITE THIS PROPERLY
	// DEAL WITH BATCH and SEQUENCES CORRECTLY
	// FOR NOW, ASSUME BATCH=1

	//printf("nb_epochs= %d\n", nb_epochs);
	for (int i=0; i < nb_epochs; i++) {
		printf("**** epoch %d ****\n", i);
		VF2D_F pred = predictViaConnectionsBias(x);
		objective->computeLoss(exact, pred);
		//exact.print("exact");
		//pred.print("pred");
		const LOSS& loss = objective->getLoss();
		LOSS ll = loss;
		//ll(0) = ll(0) / 3.7;
		//ll(0).raw_print(std::cout, "loss");
		backPropagationViaConnectionsRecursion(exact, pred);
		parameterUpdate();
		pred.reset();
	}
}
//----------------------------------------------------------------------
void Model::storeGradientsInLayersRec(int t)
{
	//printf("---- enter storeGradientsInLayersRec ----\n");
	//printf("store: t= %d\n", t);
	for (int l=0; l < layers.size(); l++) {
		// activation gradient
		layers[l]->computeGradient(t);
	}
	//printf("---- exit storeGradientsInLayersRec ----\n");
}
//----------------------------------------------------------------------
void Model::storeDActivationDOutputInLayersRecCon(int t)
{
	typedef CONNECTIONS::reverse_iterator IT;
	IT it;

	printf("ENTER storeDActivationDOutputInLayersRecCon\n");

	// if two layers (l+1) feed back into layer (l), one must accumulate into layer (l)
	// Run layers backwards
	// Run connections backwards

	// ASSUMES that clist is ordered from front to back 
	// (for the spatial connections)
	for (it=clist.rbegin(); it != clist.rend(); ++it) {
		//(*it)->printSummary();
		(*it)->dLossDOutput(t, t);
	}

	// Question: Must I somehow treat the loop connections of recurrent layers? 
	// Answer: yes, and I must increment the delta
	// Temporal connections

	// REPLACED by more general temporal connection
	// All other temporal connections into this layer
	// ........

	for (int c=0; c < clist_temporal.size(); c++) {
		//if (t == 0) continue;
		clist_temporal[c]->dLossDOutput(t, t-1);
	}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void Model::storeDLossDWeightInConnectionsRecCon(int t)
{
	typedef CONNECTIONS::reverse_iterator IT;
	IT it;

	//printf("********** ENTER storeDLossDWeightInConnections ***********\n");

	for (it=clist.rbegin(); it != clist.rend(); ++it) {
		Connection* con = (*it);

		// How to do this for a particular sequence element? 
		// Currently, only works for sequence length of 1
		// Could work if sequence were the field index

		con->dLossDWeight(t);
	}

	// Needed when there are recurrent layers

	// Perhaps one should store the delay itself with the connection? 
	// So a delay of zero is spatial, a delay of 1 or greater is temporal. 

	// Set Deltas of all the connections of temporal layers

	// deal with remainder temporal layers
	// ...
	for (int c=0; c < clist_temporal.size(); c++) {
		clist_temporal[c]->dLossDWeight(t);
	}

	//printf("********** EXIT storeDLossDWeightInConnections ***********\n");
}
//----------------------------------------------------------------------
void Model::storeDLossDBiasInLayersRec(int t)
{
	VF1D delta;
	printf("\nENTER storeDLossDBiasInLayerRec\n"); // ******** TEMP

	for (int l=0; l < layers.size(); l++) {
		printf("---------------------------------\n");
		Layer* layer = layers[l];

		if (layer->getActivation().getDerivType() == "decoupled") {
			const VF2D_F& grad      = layer->getGradient(); // df/dz
			const VF2D_F& old_deriv = layer->getDelta(); // dL/d(out)
			layer->printSummary();
			//layer->getInputs().print("layer inputs");
			//layer->getOutputs().print("layer outpus");
			// ERROR in old_derivative (layer->getDelta)

			for (int b=0; b < nb_batch; b++) {
				delta = (old_deriv[b].col(t) % grad[b].col(t));
				//delta.raw_print(cout, "delta");
				layer->incrBiasDelta(delta);
			}
			if (l != 0) { // tanh layer
				//grad[0].raw_print(cout, "\nbias activation grad"); // ok
				//old_deriv[0].raw_print(cout, "\nbias old_deriv"); // WRONG for bh?
				delta.raw_print(cout, "BiasDelta");
			}

		} else {  // coupled
			for (int b=0; b < nb_batch; b++) {
				const VF1D& x =  layer->getInputs()(b).col(t);   // ERROR
				const VF1D& y = layer->getOutputs()(b).col(t);
				const VF2D grad = layer->getActivation().jacobian(x, y); // not stored (3,3)
				const VF2D_F& old_deriv = layer->getDelta();

				const VF2D& gg = old_deriv[b].col(t).t() * grad; // (1,3)
				delta = gg.t();
				layer->incrBiasDelta(delta);
			}
		}
	}
}
//----------------------------------------------------------------------
// MUST REWRITE. Use as template. 
void Model::storeDLossDActivationParamsInLayer(int t)
{
	//printf("enter storeDLossDActivationParamsInLayer *****\n");
	VF1D delta;
	VF2D_F g;

	for (int l=0; l < layers.size(); l++) {
		Layer* layer = layers[l];

		Activation& activation = layer->getActivation();
		if (activation.getNbParams() == 0) {
			continue;
		} else {
			delta.resize(activation.getNbParams());
			delta.zeros();
		}
		//printf("xx activation name: %s\n", activation.getName().c_str());

		// REWRITE FOLLOWING LOOP

		if (activation.getDerivType() == "decoupled") {
			const VF2D_F& grad      = layer->getGradient();
			const VF2D_F& old_deriv = layer->getDelta();

			for (int i=0; i < activation.getNbParams(); i++) 
			{
				if (activation.isFrozen(i)) {
					//printf("parameter %d frozen\n", i);
					continue;
				}
				else {
					const VF2D_F& x = layer->getInputs();
					g = activation.computeGradientWRTParam(x, i);
					//g[0].raw_print(cout, "==> activation gradient"); // OK
				}
			//----------------

				for (int b=0; b < nb_batch; b++) {
					//old_deriv[b].col(t).print("    ==> deriv");
					//g[b].col(t).print("    ==> g");
					delta[i] += arma::dot(old_deriv[b].col(t), g[b].col(t)); //% grad[b].col(t);
					//layer->getActivationDelta().print("    ==> intermediate activation Delta\n"); 
				}
			}
			//delta.raw_print(cout, "==> final delta");
			//printf("increment delta\n");
		    layer->incrActivationDelta(delta);
			//layer->getActivationDelta().print("==> activation Delta\n"); exit(0);
		} else { // coupled
			;
		}
	}
	g.reset();
}
//----------------------------------------------------------------------
void Model::resetDeltas()
{
	typedef CONNECTIONS::reverse_iterator IT;
	IT it;

	for (it=clist.rbegin(); it != clist.rend(); ++it) {
		(*it)->resetDelta();
	}

	for (it=clist_temporal.rbegin(); it != clist_temporal.rend(); ++it) {
		(*it)->resetDelta();
	}

	for (int l=0; l < layers.size(); l++) {
		layers[l]->resetDelta();
	}
}
//----------------------------------------------------------------------
void Model::resetState()
{
	for (int l=0; l < layers.size(); l++) {
		layers[l]->resetState();
	}
}
//----------------------------------------------------------------------
void Model::backPropagationViaConnectionsRecursion(const VF2D_F& exact, const VF2D_F& pred)
{
	printf("\n**************** ENTER BACKPROPAGATION ***************\n");
	nb_batch = pred.n_rows;

	if (nb_batch == 0) {
		printf("backPropagationViaConnections, nb_batch must be > 0\n");
		exit(0);
	}

	resetDeltas();


	// derivative of loss function wrt output layer output
    objective->computeGradient(exact, pred);
    VF2D_F& grad = objective->getGradient(); // Check on gradient
	getOutputLayers()[0]->setDelta(grad); 

	// gradients of activation functions
 	for (int t=seq_len-1; t > -1; --t) {  // CHECK LOOP INDEX LIMIT
		storeGradientsInLayersRec(t);
	}

	// derivative of loss wrt layer output
 	for (int t=seq_len-1; t > -1; --t) {  // CHECK LOOP INDEX LIMIT
		storeDActivationDOutputInLayersRecCon(t);
	}

	// derivative of loss wrt weights
 	for (int t=seq_len-1; t > -1; --t) {  // CHECK LOOP INDEX LIMIT
		storeDLossDWeightInConnectionsRecCon(t);
	}

	// derivative of loss wrt bias
 	for (int t=seq_len-1; t > -1; --t) {  // CHECK LOOP INDEX LIMIT
		storeDLossDBiasInLayersRec(t);
	}

	// derivative of loss wrt activation parameters
 	for (int t=seq_len-1; t > -1; --t) {  // CHECK LOOP INDEX LIMIT
		storeDLossDActivationParamsInLayer(t);
	}

	printf("EXIT AFTER FIRST BACK PROP\n"); //exit(0);
}
//----------------------------------------------------------------------
Connection* Model::getConnection(Layer* layer1, Layer* layer2)
{
	// Very inefficient, but only done rarely at beginning of simulations, and number of connection is never very large. 
	// If this were a problem, I would use a dictionary. 

	for (int c=0; c < connections.size(); c++) {
		Connection* conn = connections[c];
		if (conn->from == layer1 && conn->to == layer2) {
			return conn;
		}
	}
	//printf("Model:: No connection between layers %s, %s\n", 
		//layer1->getName().c_str(), layer2->getName().c_str());
	return 0;
}
//----------------------------------------------------------------------
void Model::weightUpdate()
{
	printf("ENTER WEIGHT UPDATE\n");

	// Assume that all connections in clist are spatial
	// spatial connections
	//printf("clist size: %d\n", clist.size());
	//printf("clist_temporal size: %d\n", clist_temporal.size());

	for (int c=0; c < clist.size(); c++) {
		Connection* con = clist[c];
		//con->printSummary("clist[c]"); // GE
		//printf("con->frozen= %d\n", con->frozen);
		WEIGHT& wght = con->getWeight();
		if (con->frozen) continue;
		wght = wght - learning_rate * con->getDelta();
		con->getDelta().raw_print(cout, "spatial WEIGHT DELTA"); // GE
	    con->computeWeightTranspose();
		//con->weight_history.push_back(wght);
		//con->printSummary();
	}

	// temporal connections 
	for (int c=0; c < clist_temporal.size(); c++) {
		Connection* con = clist_temporal[c];
		//con->printSummary("clist_temporal[c]"); // GE
		WEIGHT& wght = con->getWeight();
		//printf("con->frozen= %d\n", con->frozen);
		//wght.print("update: w3");
		if (con->frozen) continue;
		//con->getDelta().print("temporal WEIGHT DELTA"); // GE
		con->getDelta().raw_print(cout, "temporal WEIGHT DELTA"); // GE
		wght = wght - learning_rate * con->getDelta();
	    con->computeWeightTranspose();
		//con->weight_history.push_back(wght);
		//con->printSummary();
	}
}
//----------------------------------------------------------------------
void Model::biasUpdate()
{
	for (int l=0; l < layers.size(); l++) {
		//printf("************ biasUpdate, layer %d\n", l);
		//printf("frozen bias: %d\n", layers[l]->getIsBiasFrozen());
		//if (layers[l]->getIsBiasFrozen() == false) {
			BIAS& bias = layers[l]->getBias();
			bias = bias - learning_rate * layers[l]->getBiasDelta();
			layers[l]->printSummary();
			//layers[l]->getBiasDelta().print("dBias");
		//}
	}
}
//----------------------------------------------------------------------
void Model::activationUpdate()
{
	for (int l=0; l < layers.size(); l++) {
		Activation& activation = layers[l]->getActivation();
		int nb_params = activation.getNbParams();
		if (nb_params == 0) continue;
		const VF1D& delta = layers[l]->getActivationDelta();

		std::vector<REAL> params;
		//layers[l]->printSummary("activationUpdate");

		for (int p=0; p < nb_params; p++) {
			if (activation.isFrozen(p)) continue;
			//printf("bef param= %21.14f, delta= %21.14f\n", activation.getParam(p), delta[p]);
			REAL param = activation.getParam(p) - learning_rate * delta[p];
			activation.setParam(p, param);
			//exit(0);
			//printf("aft param= %21.14f\n", param);
			//params.push_back(param);
		}
		//layers[l]->params_history.push_back(params);
	}
}
//----------------------------------------------------------------------
void Model::parameterUpdate()
{
	weightUpdate();  // TEMPORARY
	biasUpdate();  // TEMPORARY
    activationUpdate();
}
//----------------------------------------------------------------------
void Model::initializeWeights()
{
	CONNECTIONS& conn = getConnections();

	for (int c=0; c < conn.size(); c++) {
		conn[c]->initialize(getInitializationType());
	}

	CONNECTIONS& tconn = getTemporalConnections();
	for (int c=0; c < tconn.size(); c++) {
		tconn[c]->initialize(getInitializationType());
	}
}
//----------------------------------------------------------------------
void Model::printAllConnections()
{
	CONNECTIONS& conns = getConnections();
	for (int c=0; c < conns.size(); c++) {
		conns[c]->printSummary("connections, ");
	}
	CONNECTIONS& tconns = getTemporalConnections();
	for (int c=0; c < tconns.size(); c++) {
		tconns[c]->printSummary("temporal connections, ");
	}
}
//----------------------------------------------------------------------
void Model::freezeBiases()
{
	const LAYERS& layers = getLayers();
	for (int i=0; i < layers.size(); i++) {
		layers[i]->setIsBiasFrozen(true);
	}
}
//----------------------------------------------------------------------
void Model::freezeWeights()
{
	// FREEEZE weights  (if unfrozen, code does not run. Nans arise.)
    CONNECTIONS& cons = getConnections();
	for (int i=0; i < cons.size(); i++) {
		Connection* con = cons[i];
		//con->printSummary();
		con->freeze();
	}

    CONNECTIONS& tcons = getTemporalConnections();
	for (int i=0; i < tcons.size(); i++) {
		Connection* con = tcons[i];
		//con->printSummary();
		con->freeze();
	}
}
//----------------------------------------------------------------------
void Model::addWeightHistory(Layer* l1, Layer* l2)
{
    std::vector<WEIGHT>& hist  = getConnection(l1, l2)->weight_history;
	//hist[0].print("hist");
	weights_to_print.push_back(hist);
	//printf("weights_to_print size: %d\n", weights_to_print.size());
}
//----------------------------------------------------------------------
void Model::addParamsHistory(Layer* l1)
{
	printf("addParamsHistory, layer name: %s\n", l1->getName().c_str());
	std::vector<std::vector<REAL> >& hist = l1->params_history; // MUST FIX
	params_to_print.push_back(hist);
}
//----------------------------------------------------------------------
void Model::printWeightHistories()
{
    FILE* fd = fopen("weights.out", "w");

	if (weights_to_print.size() == 0) return;
	int nb_weights = weights_to_print.size();
	int hist_size  = weights_to_print[0].size();

	#if 1
	printf("*** print weight histories ***\n");
    for (int i=0; i < hist_size; i++) {
        fprintf(fd, "\n%d ", i);
		// For now, assume that all links have only a single weight
		printf("nb_weights= %d\n", nb_weights);
		for (int j=0; j < nb_weights; j++) {
			const WEIGHT& w = weights_to_print[j][i];
			printf("w[0,0] = %f\n", w(0,0));
			fprintf(fd, "%f ", w(0,0));
		}
    }
    fclose(fd);
	#endif
}
//----------------------------------------------------------------------
void Model::printHistories()
{
	FILE* fd;

	// Parameter history
	fd = fopen("params.out", "w");
	int sz = params_to_print.size();
	//printf("params_to_print size: %d\n"< param_to_print.size()); exit(0);
	const LAYERS& layers = getLayers();
	if (sz == 0) return;

	printf("params_to_print size: %d\n", sz); // 2

	std::vector<std::vector<REAL> > hi = params_to_print[0];
	if (sz > 0) {
		printf("nb_to_print: %d\n", hi.size()); // 5900
	}
	int nb_to_print = hi.size();

	for (int i=0; i < nb_to_print; i++) {    // ERROR
		fprintf(fd, "%d ", i);
		for (int s=0; s < sz; s++) {
			std::vector<std::vector<REAL> >& pms = params_to_print[s];
			fprintf(fd, "%f ", pms[i][0]); // but an activation could have more than 1 parameter
		}
		fprintf(fd, "\n");
	}
	fclose(fd);

	fd = fopen("loss.out", "w");
	// Assumes batch of size 1
	for (int i=0; i < loss_history.size(); i++) {
		for (int j=0; j < seq_len; ++j) {
			fprintf(fd, "%d %f\n", i*seq_len+j, loss_history[i][0][j]);
		}
	}
	fclose(fd);

	fd = fopen("x.out", "w");
	for (int i=0; i < x_in_history.size(); i++) {
		fprintf(fd, "%d %f %f\n", i, x_in_history[i], x_out_history[i]);
	}
	fclose(fd);
}
//----------------------------------------------------------------------
void Model::setSeqLen(int seq_len) 
{ 
// change sequence, and also change integrity of 

	this->seq_len = seq_len;

	const LAYERS& layers = getLayers();
	for (int l=0; l  < layers.size(); l++) {
		layers[l]->setSeqLen(seq_len);
		layers[l]->initVars(batch_size);
	}

	checkIntegrity();
}
//----------------------------------------------------------------------
void Model::setWeightsAndBiases(Model* mfrom)
{
	CONNECTIONS::iterator c;
	CONNECTIONS& clist_from = mfrom->getClist();
	CONNECTIONS& clist_to   =  this->getClist();
	//CONNECTIONS::iterator ct;

	// ignore weight transposes

	for (int i=0; i < clist_from.size(); i++) {
		WEIGHT& w_from = clist_from[i]->getWeight();
		clist_to[i]->setWeight(w_from);
	}

	CONNECTIONS& clist_temporal_from = mfrom->getTemporalConnections();
	CONNECTIONS& clist_temporal_to   =  this->getTemporalConnections();

	for (int i=0; i < clist_temporal_from.size(); i++) {
		WEIGHT& w_from = clist_temporal_from[i]->getWeight();
		clist_temporal_to[i]->setWeight(w_from);
	}

	const LAYERS& layers_from = mfrom->getLayers();
	const LAYERS& layers_to   =  this->getLayers();

	//LAYERS::iterator la;
	for (int i=0; i < layers_from.size(); i++) {
		BIAS& bias = layers[i]->getBias();
	}
}
//----------------------------------------------------------------------
