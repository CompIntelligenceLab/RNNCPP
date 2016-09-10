//#include <armadillo>
#include <assert.h>
#include "model.h"
#include "objective.h"
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

  	layers.push_back(layer_to);
	layer_to->setNbBatch(nb_batch); // check
	layer_to->setSeqLen(seq_len); // check
	printf("layers: nb_batch= %d\n", nb_batch);

	int in_dim  = layer_to->getInputDim();
	int out_dim = layer_to->getOutputDim();
	printf("Model::add, layer_to dim: in_dim: %d, out_dim: %d\n", in_dim, out_dim);

	// Create weights
	// Later on, we might create different kinds of connection. This would require a rework of 
	// the interface. 
	Connection* connection = Connection::ConnectionFactory(in_dim, out_dim, conn_type);
	connection->initialize();
	connections.push_back(connection);
	connection->from = layer_from;
	connection->to = layer_to;

	// update prev and next lists in Layers class
	if (layer_from) {
		layer_from->next.push_back(std::pair<Layer*, Connection*>(layer_to, connection));
		layer_to->prev.push_back(std::pair<Layer*, Connection*>(layer_from, connection));
		connection->which_lc = layer_to->prev.size()-1;
		printf("    connection->which_lc= %d\n", connection->which_lc);
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
VF2D_F Model::predict(VF2D_F x)
{
  	VF2D_F prod(x); //copy constructor, .n_rows);

	for (int l=1; l < layers.size(); l++) {
  		const WEIGHTS& wght= layers[l]->getWeights(); // between layer (l) and layer (l-1)

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
//----------------------------------------------------------------------
void Model::connectionOrderClean()
{
	typedef std::list<Layer*>::iterator IT;
	IT it;

	// STL list to allow erase of elements via address
	std::list<Layer*> llist;
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
	printf("\n------ END MODEL SUMMARY ------------------------------------\n\n");
}
//----------------------------------------------------------------------
VF2D_F Model::predictViaConnectionsBias(VF2D_F x)
{
	VF2D_F prod(x.size());
	//printf("****************** ENTER predictViaConnections ***************\n");

	Layer* input_layer = getInputLayers()[0];
	input_layer->layer_inputs[0] = x; 
	input_layer->setOutputs(x);  // although input layer "could" have a nonlinear activation function (maybe)

 	for (int t=0; t < (seq_len); t++) {  // CHECK LOOP INDEX LIMIT
		for (int l=0; l < layers.size(); l++) {
			layers[l]->nb_hit = 0;
		}

		// go through all the layers and update the temporal connections
		// On the first pass, connections are empty
		for (int l=0; l < layers.size(); l++) {
			layers[l]->forwardLoops(t-1);    // does not change with biases (empty functions it seems)
		}
		
		for (int c=0; c < clist.size(); c++) {
			Connection* conn  = clist[c];
			Layer* to_layer   = conn->to;
			to_layer->processOutputDataFromPreviousLayer(conn, prod, t);
		}
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

	VF2D_F pred = predict(x);
	VF1D_F loss = objective->computeError(y, pred);
	printf("loss.n_rows= ", loss.n_rows);
	loss.print("loss");
}
//----------------------------------------------------------------------
void Model::initializeWeights(std::string initialization_type /* "uniform" */)
{
	//printf("---- enter storeGradientsInLayersRec ----\n");
	for (int l=0; l < layers.size(); l++) {
		layers[l]->computeGradient(t);
	}
	//printf("---- exit storeGradientsInLayersRec ----\n");
}
//----------------------------------------------------------------------
void Model::storeDactivationDoutputInLayersRecCon(int t)
{
	typedef CONNECTIONS::reverse_iterator IT;
	IT it;

	// if two layers (l+1) feed back into layer (l), one must accumulate into layer (l)
	// Run layers backwards
	// Run connections backwards

	// Memory allocated in gradMulDLda)(
	//VF2D_F prod(nb_batch);

	for (it=clist.rbegin(); it != clist.rend(); ++it) {
		Connection* conn = (*it);
		conn->gradMulDLda(t, t);
	}

	// Question: Must I somehow treat the loop connections of recurrent layers? 
	// Answer: yes, and I must increment the delta

	for (int l=0; l < layers.size(); l++) {
		Connection* conn = layers[l]->getConnection();

		if (!conn) continue;

		// Question: where should this operation occur. Given that an activation function can be scalar or vector, 
		// the operation should be split between the activation function and the model (or layer or connection)
		// For example: 
		// prod = grad % old_deriv   (or)
		// prod = grad * old_deriv   (or) (or  old_deriv * grad)
		// prod = wght_t * prod

		if (t == 0) continue;
		conn->gradMulDLda(t, t-1);  
	}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void Model::storeDLossDweightInConnectionsRecCon(int t)
{
	typedef CONNECTIONS::reverse_iterator IT;
	IT it;

	//printf("********** ENTER storeDLossDweightInConnections ***********\n");

	for (it=clist.rbegin(); it != clist.rend(); ++it) {
		Connection* con = (*it);

		// How to do this for a particular sequence element? 
		// Currently, only works for sequence length of 1
		// Could work if sequence were the field index

		con->dLdaMulGrad(t);
	}

	// Needed when there are recurrent layers

	// Perhaps one should store the delay itself with the connection? 
	// So a delay of zero is spatial, a delay of 1 or greater is temporal. 

	// Set Deltas of all the connections of temporal layers
	for (int l=0; l < layers.size(); l++) {
		Connection* con = layers[l]->getConnection();
		if (!con) continue;
		con->dLdaMulGrad(t);
	}
	//printf("********** EXIT storeDLossDweightInConnections ***********\n");
}
//----------------------------------------------------------------------
void Model::storeDLossDbiasInLayersRec(int t)
{
	VF1D delta;

	for (int l=0; l < layers.size(); l++) {
		Layer* layer = layers[l];

		if (layer->getActivation().getDerivType() == "decoupled") {
			const VF2D_F& grad      = layer->getGradient();
			const VF2D_F& old_deriv = layer->getDelta();

			for (int b=0; b < nb_batch; b++) {
				delta = (old_deriv[b].col(t) % grad[b].col(t));
			}

			layer->incrBiasDelta(delta);
		} else {
			for (int b=0; b < nb_batch; b++) {
				const VF1D& x =  layer->getInputs()(b).col(t);   // ERROR
				const VF1D& y = layer->getOutputs()(b).col(t);
				const VF2D grad = layer->getActivation().jacobian(x, y); // not stored (3,3)
				const VF2D_F& old_deriv = layer->getDelta();

				const VF2D& gg = old_deriv[b].col(t).t() * grad; // (1,3)
				delta = gg.t();
			}

			layer->incrBiasDelta(delta);
		}
	}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void Model::backPropagationViaConnectionsRecursion(const VF2D_F& exact, const VF2D_F& pred)
{
	nb_batch = pred.n_rows;

	if (nb_batch == 0) {
		printf("backPropagationViaConnections, nb_batch must be > 0\n");
		exit(0);
	}

	resetDeltas();

    objective->computeGradient(exact, pred);
    VF2D_F& grad = objective->getGradient();
	getOutputLayers()[0]->setDelta(grad);  // assumes single output layer

 	for (int t=seq_len-1; t > -1; --t) {  // CHECK LOOP INDEX LIMIT
		storeGradientsInLayersRec(t);
		//storeDactivationDoutputInLayersRec(t);
		storeDactivationDoutputInLayersRecCon(t);
		//storeDLossDweightInConnectionsRec(t);
		storeDLossDweightInConnectionsRecCon(t);
		storeDLossDbiasInLayersRec(t);
	}
	//printf("***************** EXIT BACKPROPVIACONNECTIONS_RECURSIONS <<<<<<<<<<<<<<<<<<<<<<\n");
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
