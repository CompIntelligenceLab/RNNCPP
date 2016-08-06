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
	Objective* loss;
	LAYERS layers; // operate by value for safety) 

public: 
   Model(std::string name="model");
   ~Model();
   void print(std::string msg="");

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

   /** return vector of weights for each layer */
   WeightList getWeights();
};

#endif
