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
	Objective* loss;
	LAYERS layers; // operate by value for safety)
	std::string initialization_type;
	int input_dim;   // dimensional input into the model
	int batch_size;  // batch_size used for training, etc.
  int nb_epoch;
	int seq_len;     // sequence length (should not be a layer property)
	                // represents the number of times to unroll
	bool print_verbose;

public:
  Model(int input_dim, std::string name="model");
  ~Model();
  Model(const Model&); // probably do not need it, but it is a good exercise. 
  Model& operator=(const Model&); 
  void print(std::string msg=std::string());

  // Use pointer instead of reference to avoid including layers.h
  void add(Layer* layer);

  void setOptimizer(Optimizer* opt);
  Optimizer* getOptimizer();
  void setStateful(bool stateful);
  bool getStateful();
  void setReturnSequences(bool state);
  bool getReturnSequences();
  void setLearningRate(float lr);
  float getLearningRate();
  GRADIENTS getGradient();
  int getInputDim() {return input_dim;}
  int getBatchSize() {return batch_size;}
  int getSeqLen() {return seq_len;}
  void setInputDim(int input_dim) {this->input_dim = input_dim;}
  void setBatchSize(int batch_size) {this->batch_size = batch_size;}
  void setSeqLen(int seq_len) { this->seq_len = seq_len;}
  void setName(std::string name) { this->name = name; }
  LAYERS getLayers() { return layers; };
  std::string getName() { return name; }

  /** return vector of weights for each layer */
  WeightList getWeights();

  /** predict: run through the model  */
  //  x: signal input: (batch_size, seq_length, dimension)
  //  For non-recursive networks, x has size (batch_size, 1, dimension)
  void predict(VF3D x); // If this is to be accessed by the user, they must have armadillo. Not good
  void train(MATRIX x, MATRIX y, int batch_size=0, int nb_epoch=0); // 0 defaults are flags not actual values. See model.cpp
  void compile();

  /** for now, initialize with random weights in [-1,1], from a Gaussian distribution.  */
  // Also allow fixed initialization in [-.8, .8] from a uniform distribution */
  // "gaussian", "uniform", "orthogonal"
  void initializeWeights(std::string initialization_type="uniform");
};

#endif
