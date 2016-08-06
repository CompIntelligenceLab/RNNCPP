#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <vector>
#include <math.h>
#include <string>
#include "typedefs.h"
#include "weights.h"
#include "gradient.h"

class Layer
{
private:
	int seq_len;
	int batch_size;
	int dim;  // dimension of data (scalar, ...)
	int layer_size; // number of nodes in layer
	VI3 input_dim;
	Weights* weights;
	VF inputs;  // inputs to activation function
	VF outputs; // outputs to activation function

public:
   Layer(int layer_size=1); // allows for default constructor
   ~Layer();
   Layer(Layer&);
   virtual void print() {;}

   virtual void setBatchSize(int batch_size) { this->batch_size = batch_size; }
   virtual int  getBatchSize() { return batch_size; }
   virtual void setSeqLen(int seq_len) { this->seq_len = seq_len; }
   virtual int getSeqLen() { return seq_len; }
   virtual void setInputDim(VI3& input_dim) { this->input_dim = input_dim; }
   virtual VI3& getInputDim() { return input_dim; }

   /** get layer weights */
   WeightList getWeights();  // not sure of data structure

   // gradient of loss function with respect to weights
   void computeGradient();
   GRADIENTS getGradient();
};
//----------------------------------------------------------------------
/* use of this typedef requires inclusion of this file */
typedef std::vector<Layer> LAYERS;
//----------------------------------------------------------------------

#endif
