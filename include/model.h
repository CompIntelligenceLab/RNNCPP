#ifndef __Model_H__
#define __Model_H__

#include <vector>
#include <string>
#include "typedefs.h"
#include "gradient.h"
#include "weights.h"
#include "layers.h"
#include "optimizer.h"
#include "objective.h"
#include "layers.h"

//class Optimizer;
//class Objective;
//class Layer;

class Model
{
private:
	std::string name;
	bool   stateful;
	float learning_rate;
	bool return_sequences;
	Optimizer* optimizer;
	// More general models would have several loss functions running concurrently
	Objective* objective;
	std::string initialization_type;
	int input_dim;   // dimensional input into the model
	int batch_size;  // batch_size used for training, etc.
  	int nb_epochs;
	int seq_len;     // sequence length (should not be a layer property)
	                // represents the number of times to unroll
	bool print_verbose;
	// keep pointers to all weights into a dynamical linked list
	LAYERS layers;
	WEIGHT_L weights_l; // (l)ist of weights

public:
  Model(int input_dim, std::string name="model");
  ~Model();
  Model(const Model&); // probably do not need it, but it is a good exercise. 
  const Model& operator=(const Model&); 
  void print(std::string msg=std::string());

  // Use pointer instead of reference to avoid including layers.h
  /** update layer list. check for layer compatibility with previous layer */

  void add(Layer* layer);
  void add(Layer* layer_from, Layer* layer);
  void setOptimizer(Optimizer* opt) {optimizer = opt;}
  Optimizer* getOptimizer() const {return optimizer;}
  void setLoss(Objective* obj) {objective = obj;}
  Objective* getObjective() const {return objective;}
  void setStateful(bool stateful) {this->stateful = stateful;}
  bool getStateful() const {return stateful;}
  void setReturnSequences(bool ret_seq) {return_sequences = ret_seq;}
  bool getReturnSequences() const {return return_sequences;}
  void setLearningRate(float lr) {learning_rate = lr;}
  float getLearningRate() const {return learning_rate;}
  int getInputDim() const {return input_dim;}
  int getBatchSize() const {return batch_size;}
  int getSeqLen() const {return seq_len;}
  void setInputDim(int input_dim) {this->input_dim = input_dim;}
  void setBatchSize(int batch_size) {this->batch_size = batch_size;}
  void setSeqLen(int seq_len) { this->seq_len = seq_len;}
  void setName(std::string name) { this->name = name; }
  LAYERS getLayers() const { return layers; };
  std::string getName() const { return name; }

  // Still need to decided the data structures and use of this
  GRADIENTS getGradient() const;

  /** return vector of weights for each layer */
  WEIGHT_L& getWeights() { return weights_l; }
  //WeightList& getWeightsL();

  /** predict: run through the model  */
  //  x: signal input: (batch_size, seq_length, dimension)
  //  For non-recursive networks, x has size (batch_size, 1, dimension)
  VF2D_F predict(VF2D_F x); 
  void train(VF2D_F x, VF2D_F y, int batch_size=0, int nb_epochs=1);
  void compile();

  /** for now, initialize with random weights in [-1,1], from a Gaussian distribution.  */
  // Also allow fixed initialization in [-.8, .8] from a uniform distribution */
  // "gaussian", "uniform", "orthogonal"
  void initializeWeights(std::string initialization_type="uniform");
};

#endif
