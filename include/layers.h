#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <vector>
#include <math.h>
#include <string>
#include "typedefs.h"
#include "connection.h"
#include "gradient.h"
#include "activations.h"

class Activation;


class Layer
{
public:
	// list of layers this layer is sending information to
	PAIRS_L prev;
	PAIRS_L next;
	// main inputs to activation in a list to better handle backpropagation when 
	// more than one layer hits a downstream layer
	LAYER_INPUTS layer_inputs;  // change to inputs later


protected:
	static int counter;
	int clock; // initialized to zero. updates by one when signal arrives. If signal arrives when clock != 0,
	           // change the connection to temporal from spatial. Also used in predict(). 
	std::string name;
	int layer_size; // number of nodes in layer
	int input_dim; // size of previous layer
	VF2D_F inputs;  // inputs to activation function (batch_sz, seq_len)  // change to input later
	VF2D_F outputs; // outputs from activation function
	DELTA delta; // outputs from activation function
	//WEIGHTS weights; // between this layer and the previous one. Breaks down 
	                // if layers form graphs (recurrent or not)
					// in the first layer, weights is not initialized. 
	//Weights* weights;  // original code. Nathan wants to simplify

	//std::vector<std::pair<Layer*, Weight*> > prev;

	// list of layers this layer is receiving information from
	//std::vector<std::pair<Layer*, Weights*> > next;



	// Eventually, we will have two lists of nodes: 
	// std::vector<Weight*> w_prev;
	// std::vector<Layer*> l_prev; 
	// std::vector<Weight*> w_next;
	// std::vector<Layer*> l_next; 
	//
	// Alternatively
	// std::vector<std::pair<Layer*, Weight*> > prev;
	// std::vector<std::pair<Layer*, Weight*> > next;
	//
	// Usage:   prev.push_back(std::pair(new Layer()..., new Weight()...))
	// Be careful with destructors: must delete layers and weights, unless
	// they all point to a separate list that contains all weights/layers in the system. 
	// Implementation is not clear. 
	// 
	// it might be more efficient to work with objects: 
	// std::vector<std::pair<Layer, Weight> > prev;
	// std::vector<std::pair<Layer, Weight> > next;

    //
	//   Usage:    prev.push_back(Layer(...), Weight(...))
	//          Destructor of this list, automatically deletes layers and weights 
	//          it is associated with. I doubt we want that. 
	// 
	// Note that in standard linked lists, one would have 
	//    Object* next;
	//    Object* prev;

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
   virtual void printSummary(std::string msg="");
   virtual void printName(std::string msg="");

   virtual int  getInputDim() const { return input_dim; }
   // input from previous layer
   virtual void setInputDim(int input_dim) { this->input_dim = input_dim; }
   virtual int  getOutputDim() { return this->getLayerSize(); }
   virtual void setOutputDim(int output_dim) { this->setLayerSize(output_dim); }
   virtual Activation& getActivation() const { return *activation; }
   virtual void setActivation(Activation* activation) { this->activation = activation; }

   /** Compute the outputs given the inputs */
   virtual void execute();

   /** get layer weights */
   //virtual void createWeights(int in, int out);  // not sure of data structure (Just simple matrix for now)
   //virtual void initializeWeights(std::string initialization_type="uniform");  // not sure of data structure
   //WEIGHTS getWeights() const {return weights;}  // not sure of data structure (Just simple matrix for now)

   // gradient of loss function with respect to weights
   void computeGradient();
   GRADIENTS getGradient() const {return gradients;}

	int getNbBatch() { return nb_batch; }
   	void setNbBatch(int nb_batch) { this->nb_batch = nb_batch; initVars(nb_batch);  }
	//int getInputDim() const {return input_dim;}  // in reality, the layer size
  	//void setInputDim(int input_dim) {this->input_dim = input_dim;}
  	void setSeqLen(int seq_len) { this->seq_len = seq_len;}
  	int getSeqLen() const {return seq_len;}

	int getLayerSize() { return layer_size; }
	void setLayerSize(int layer_size) { this->layer_size = layer_size; }
	void setInputs(VF2D_F& inputs) { this->inputs = inputs; }
	VF2D_F& getInputs() { return inputs; }
	void setOutputs(VF2D_F& outputs) { this->outputs = outputs; }
	VF2D_F& getOutputs() { return outputs; }
	void incrOutputs(VF2D_F& x);
	void incrInputs(VF2D_F& x);
	void incrDelta(VF2D_F& x);
	// reset inputs and ouputs to zero
	void reset();
	// reset deltas
	void resetBackprop();
	void setName(std::string name) { this->name = name; } // normally not used
	std::string getName() { return name; }
	int getClock() { return clock; }
	void incrClock() { clock += 1; }
	DELTA& getDelta() { return delta; }
	void setDelta(DELTA delta) { this->delta = delta; }

public:
	virtual void initVars(int nb_batch);
};

//----------------------------------------------------------------------
/* use of this typedef requires inclusion of this file */
typedef std::vector<Layer*> LAYERS;
//----------------------------------------------------------------------

#endif
