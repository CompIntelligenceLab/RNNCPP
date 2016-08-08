#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <vector>
#include <math.h>
#include <string>
#include "typedefs.h"
#include "weights.h"
#include "gradient.h"
#include "activations.h"

class Activation;

class Layer
{
protected:
	static int counter;
	std::string name;
	int seq_len;
	int batch_size;
	int layer_size; // number of nodes in layer
	int input_dim; // size of previous layer
	VF inputs;  // inputs to activation function
	VF outputs; // outputs to activation function
	WeightList weights; // class Weights:
	Activation* activation;
	bool print_verbose;

public:
   Layer(int layer_size=1, std::string name="layer"); // allows for default constructor
   ~Layer();
   Layer(const Layer&);
   Layer& operator=(const Layer&);
   virtual void print(std::string msg="");

   virtual void setBatchSize(int batch_size) { this->batch_size = batch_size; }
   virtual int  getBatchSize() { return batch_size; }
   virtual void setSeqLen(int seq_len) { this->seq_len = seq_len; }
   virtual int  getSeqLen() { return seq_len; }
   virtual void setInputDim(int input_dim) { this->input_dim = input_dim; }
   virtual int  getInputDim() { return input_dim; }
   virtual void setActivation(Activation* activation) { this->activation = activation; }
   virtual Activation* getActivation() { return activation; }

   /** Compute the outputs given the inputs */
   virtual void execute();

   /** get layer weights */
   WeightList& getWeights() { return weights; }
   ;  // not sure of data structure
   virtual void createWeights(int in, int out);  // not sure of data structure
   virtual void initializeWeights(std::string initialization_type="uniform");  // not sure of data structure

   // gradient of loss function with respect to weights
   void computeGradient();
   GRADIENTS getGradient();
};
//----------------------------------------------------------------------
/* use of this typedef requires inclusion of this file */
typedef std::vector<Layer*> LAYERS;
//----------------------------------------------------------------------

#endif
