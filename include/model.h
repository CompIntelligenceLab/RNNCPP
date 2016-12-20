#ifndef __Model_H__
#define __Model_H__

#include <vector>
#include <list>
#include <string>
#include "typedefs.h"
#include "gradient.h"
#include "connection.h"
#include "layers.h"
#include "optimizer.h"
#include "objective.h"
#include "activations.h"
#include "layers.h"

//class Optimizer;
//class Objective;
//class Layer;

class Model
{
public:
// Variables specific to command line arguments to simplify calling function processArguments(argc, argv)
	REAL inc;
	int layer_size;
	bool is_recurrent;
	//Activation* activation;
	std::vector<Activation*> activations;
	int nb_serial_layers;
	int nb_parallel_layers;
	REAL dt;
  	int nb_epochs;
	std::string obj_err_type;
	REAL init_weight_rms;

	// Parameter histories
	// true: save history every iteration (each iteration is sequence length of seq_len)
	bool params_hist, x_in_hist, x_out_hist;  
	// assume input has dimension 1. Else, one should store VF2D (if nb_batch = 1) or else VF2D_F
	std::vector<REAL>   x_in_history;
	std::vector<REAL>   x_out_history;
	std::vector<LOSS>   loss_history;
	std::vector<std::vector<WEIGHT> > weights_to_print;  // need better approach. For large networks, we
	                                                    // cannot save all weights. 
	std::vector<std::vector<std::vector<REAL> > > params_to_print;  

private:
	int nb_batch; // equal to batch_size?
	// order in which connections should be handled, computed by connectionOrderClean()
	CONNECTIONS clist;  // spatial connections
	CONNECTIONS clist_temporal;  // temporal connections
	std::string name;
	bool   stateful;
	REAL learning_rate;
	bool return_sequences;
	Optimizer* optimizer;
	// More general models would have several loss functions running concurrently
	Objective* objective;
	int input_dim;   // dimensional input into the model
	int batch_size;  // batch_size used for training, etc.
	int seq_len;     // sequence length (should not be a layer property)
	                // represents the number of times to unroll
	bool print_verbose;
	std::string initialization_type; // how to initialize weights
	// keep pointers to all weights into a dynamical linked list
	LAYERS layers;
	LAYERS input_layers;
	LAYERS output_layers;
	LAYERS loss_layers;
	// Probe layers have no output
	LAYERS probe_layers; // to help read output of previous layer 
	CONNECTIONS connections; // (l)ist of weights
	CONNECTIONS order_eval; // Order in which connections are evaluated for prediction, and possibly training.

public:
  Model(std::string name="model");
  ~Model();
  Model(const Model&); // probably do not need it, but it is a good exercise. 
  const Model& operator=(const Model&); 
  void print(std::string msg="");

  /** print connections, connection type, weight matrix size, layers, layer types */
  void printSummary();
  void printAllConnections();

  void setObjective(Objective* obj) { objective = obj; }
  Objective* getObjective() { return objective; }

  // Use pointer instead of reference to avoid including layers.h
  /** update layer list. check for layer compatibility with previous layer */

  //void add(Layer* layer);
  std::string getInitializationType() { return initialization_type; }
  void setInitializationType(std::string initialization_type) { this->initialization_type = initialization_type; }
  void add(Layer* layer_from, Layer* layer, std::string conn_type="all-all");
  // Identical routine to previous add(), except for the addition of is_temporal
  // When is_temporal == true, the next() and prev() elements of connected layers are not updated. 
  // next() and prev() will only be use for spatial connections
  void add(Layer* layer_from, Layer* layer_to, bool is_temporal, std::string conn_type="all-all");
  void addInputLayer(Layer* layer);
  // Specify output layer and activation function to Identity()
  void addOutputLayer(Layer* layer);
  void addLossLayer(Layer* layer);
  void addProbeLayer(Layer* layer);
  LAYERS getOutputLayers() { return output_layers; }
  LAYERS getInputLayers() { return input_layers; }
  LAYERS getLossLayers() { return loss_layers; }
  LAYERS getProbeLayers() { return probe_layers; }
  void setOptimizer(Optimizer* opt) {optimizer = opt;}
  Optimizer* getOptimizer() const {return optimizer;}
  void setLoss(Objective* obj) {objective = obj;}
  Objective* getObjective() const {return objective;}
  void setStateful(bool stateful) {this->stateful = stateful;}
  bool getStateful() const {return stateful;}
  void setReturnSequences(bool ret_seq) {return_sequences = ret_seq;}
  bool getReturnSequences() const {return return_sequences;}
  void setLearningRate(REAL lr) {learning_rate = lr;}
  REAL getLearningRate() const {return learning_rate;}
  int getInputDim() const {return input_dim;}
  int getBatchSize() const {return batch_size;}
  int getSeqLen() const {return seq_len;}
  void setInputDim(int input_dim) {this->input_dim = input_dim;}
  void setBatchSize(int batch_size) {this->batch_size = this->nb_batch = batch_size;}
  void setSeqLen(int seq_len);
  void setName(std::string name) { this->name = name; }
  const LAYERS& getLayers() const { return layers; };
  void setLayers(LAYERS layer_list) { layers = layer_list; }
  std::string getName() const { return name; }
  void checkIntegrity(); // change connections from spatial to temporal if necessary
                                // A signal propagating along spatial connections should never
								// encounter a node already reached

  // Still need to decided the data structures and use of this
  GRADIENTS getGradient() const;

  /** return vector of weights for each layer */
  CONNECTIONS& getConnections() { return connections; }
  CONNECTIONS& getTemporalConnections() { return clist_temporal; }
  CONNECTIONS& getClist() { return clist; }
  //WeightList& getWeightsL();

  /** predict: run through the model  */
  //  x: signal input: (batch_size, seq_length, dimension)
  //  For non-recursive networks, x has size (batch_size, 1, dimension)
  //VF2D_F predictComplexMaybeWorks(VF2D_F x);  // for testing while Nathan works with predict
  //VF2D_F predictComplex(VF2D_F x);  // for testing while Nathan works with predict
  VF2D_F predictViaConnections(VF2D_F x); 
  // Same as predictviaConnections, but take bias into account
  VF2D_F predictViaConnectionsBias(VF2D_F x);
  void predictViaConnectionsBias(VF2D_F x, VF2D_F& prod);
  void  storeGradientsInLayers();
  void  storeDactivationDoutputInLayers();
  void 	storeDLossDweightInConnections();
  //void 	storeDLossDweightInConnectionsCon();
  // derivative of loss function wrt activation parameters
  void storeDLossDactivationParamsInLayer(int t);

  void storeGradientsInLayersRec(int t);
  void storeDactivationDoutputInLayersRecCon(int t);
  void storeDLossDweightInConnectionsRecCon(int t);
  void storeDLossDbiasInLayersRec(int t);

  // x are predicted values, y are exact labels
  void train(VF2D_F x, VF2D_F y, int batch_size=0, int nb_epochs=1);
  // train a single batch. x[input_size, seq_len], exact[last_layer_size, seq_len]
  void trainOneBatch(VF2D_F& x, VF2D_F& y);
  void trainOneBatch(VF2D x, VF2D y);
  void backPropagation(VF2D_F y, VF2D_F prep);
  // networks that have multiple layers leaving a layer arriving at a layer
  // should be the inverse of the forward propagation (predict)
  //void backPropagationComplex(VF2D_F y, VF2D_F pred);
  void backPropagationViaConnections(const VF2D_F& exact, const VF2D_F& pred);
  // version for sequences and recursion
  void backPropagationViaConnectionsRecursion(const VF2D_F& exact, const VF2D_F& pred);
  void compile();
  // Evaluate connection order to run prediction of a spatial network
  //CONNECTIONS connectionOrder();
  void connectionOrderClean();  // in code diff_eq4.cpp
	Layer* checkPrevconnections(std::list<Layer*> llist);
	void removeFromList(LAYERS& llist, Layer* cur_layer);

  /** for now, initialize with random weights in [-1,1], from a Gaussian distribution.  */
  // Also allow fixed initialization in [-.8, .8] from a uniform distribution */
  // "gaussian", "uniform", "orthogonal"
  void removeFromList(std::list<Layer*>& llist, Layer* cur_layer);
  void resetDeltas();
  void resetState();
  Connection* getConnection(Layer* layer1, Layer* layer2);
  void initializeWeights();
  void printHistories();
  void printWeightHistories();
  void addWeightHistory(Layer* l1, Layer* l2);
  void addParamsHistory(Layer* l1);

  // might need in the future
  //void initializeBiases();

  // Ultimately, this should probably go into another polymorphic clas sequence
  void weightUpdate();
  void biasUpdate();
  void activationUpdate();
  void parameterUpdate();
  void freezeWeights();
  void freezeBiases();

  // take weights and biases from model m and copy them to current model
  // assume the model sizes and architecture are identical. 
  void setWeightsAndBiases(Model* m);

private:
  void checkIntegrity(LAYERS& layer_list);
  bool isLayerComplete(Layer* layer);
  bool areIncomingLayerConnectionsComplete(Layer* layer);
};

#endif
