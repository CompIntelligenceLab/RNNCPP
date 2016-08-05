#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <vector>
#include <math.h>
#include "typedefs.h"


class Layer
{
private:
	int seq_len;
	int batch_size;
	VI3 input_dim;

public:
   Layer();
   ~Layer();
   Layer(Layer&);

   void setBatchSize(int batch_size) { this->batch_size = batch_size; }
   int  getBatchSize() { return batch_size; }
   void setSeqLen(int seq_len) { this->seq_len = seq_len; }
   int getSeqLen() { return seq_len; }
   void setInputDim(VI3& input_dim) { this->input_dim = input_dim; }
   VI3& getInputDim() { return input_dim; }

   // gradient of loss function with respect to weights
   void computeGradient();
};
//----------------------------------------------------------------------

#endif
