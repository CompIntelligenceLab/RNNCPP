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
	int layer_size; // number of nodes in layer
	int input_dim; // size of previous layer
	VF inputs;  // inputs to activation function
	VF outputs; // outputs from activation function
	WEIGHTS weights;
	//Weights* weights;  // original code. Nathan wants to simplify
    GRADIENTS gradients;
	Activation* activation;
	bool print_verbose;
	int nb_batch; // number of batches (batch_size = nb_batch)
	int seq_len; // sequence length for recurrent networks. Otherwise equal to 1. 

public:
   Layer(int layer_size=1, std::string name="layer"); // allows for default constructor
   ~Layer();
   Layer(const Layer&);
   const Layer& operator=(const Layer&);
   virtual void print(std::string msg="");

   virtual int  getInputDim() const { return input_dim; }
   virtual void setInputDim(int input_dim) { this->input_dim = input_dim; }
   virtual Activation* getActivation() const { return activation; }
   virtual void setActivation(Activation* activation) { this->activation = activation; }

   /** Compute the outputs given the inputs */
   virtual void execute();

   /** get layer weights */
   virtual void createWeights(int in, int out);  // not sure of data structure (Just simple matrix for now)
   virtual void initializeWeights(std::string initialization_type="uniform");  // not sure of data structure
   WEIGHTS getWeights() const {return weights;}  // not sure of data structure (Just simple matrix for now)

   // gradient of loss function with respect to weights
   void computeGradient();
   GRADIENTS getGradient() const {return gradients;}

	int getNbBatch() { return nb_batch; }
   	void setNbBatch(int nb_batch) { this->nb_batch = nb_batch; }
	//int getInputDim() const {return input_dim;}  // in reality, the layer size
  	//void setInputDim(int input_dim) {this->input_dim = input_dim;}
  	void setSeqLen(int seq_len) { this->seq_len = seq_len;}
  	int getSeqLen() const {return seq_len;}

	int getLayerSize() { return layer_size; }
	void setLayerSize(int layer_size) { this->layer_size = layer_size; }
};

//----------------------------------------------------------------------
/* use of this typedef requires inclusion of this file */
typedef std::vector<Layer*> LAYERS;
//----------------------------------------------------------------------

#endif
