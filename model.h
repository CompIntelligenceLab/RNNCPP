#ifndef __Model_H__
#define __Model_H__

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
};

#endif
