#ifndef __Model_H__
#define __Model_H__

#include <vector>
#include "typedefs.h"
#include "gradient.h"
#include "weights.h"
#include "layers.h"

class Optimizer;
class Objective;
class Layer;

class Model
{
private:
	bool   stateful;
	float learning_rate;
	bool return_sequences;
	Optimizer* optimizer;
	Objective* loss;
	LAYERS layers; // operate by value for safety) 

public: 
   Model();
   ~Model();

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
   WEIGHTS getWeights();
};

#endif
