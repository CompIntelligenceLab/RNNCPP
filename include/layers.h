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

	// For more general network. Used to handle all temporal links
	// The first link will be the self loop of a recurrent node
	PAIRS_L prev_temporal;
	PAIRS_L next_temporal;

	std::string type;
	// main inputs to activation in a list to better handle backpropagation when 
	// more than one layer hits a downstream layer
	std::vector<VF2D_F> layer_inputs;
	std::vector<VF2D_F> layer_deltas; 

#ifdef DEBUG
    //  DELTA: VF2D_F: (batch)(layer_size, seq_len)
	std::vector<DELTA> deltas;  // d(loss) / d(layer), one per sequence element
#endif


	int nb_hit; // used to determine order of evaluation of a general spatial network
	VF2D_F dLdOut;  // use getters and setters later.  d(loss)/d(layer_output)

	// should really be in recurrent.h, but I do not know how to elegantly add up the inputs
	// otherwise. One approach would be to add up the inputs as they arrive: keep them in both 
	// inputs and in layer_inputs. 
	Connection* recurrent_conn;
	VF2D_F loop_input;
	VF2D_F loop_delta;

protected:
	static int counter;
	int clock; // initialized to zero. updates by one when signal arrives. If signal arrives when clock != 0,
	           // change the connection to temporal from spatial. Also used in predict(). 
	std::string name;
	int layer_size; // number of nodes in layer
	int input_dim; // size of previous layer
	VF2D_F inputs;  // inputs to activation function (batch_sz, seq_len)  // change to input later
	VF2D_F outputs; // outputs from activation function
	DELTA delta; // d(loss) / d(layer output)  (transform to row vector)
	BIAS bias;
	bool is_bias_frozen;
	VF1D bias_delta; // arma::Col, d(loss) / d(bias)   (one bias per layer)
	VF1D activation_delta;   // one delta per activation parameter (set to zero if frozen parameter)

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

    GRADIENTS gradient;  // derivative of activation function with respect to argument
	Activation* activation;
	bool print_verbose;
	int nb_batch; // number of batches (batch_size = nb_batch)
	int seq_len; // sequence length for recurrent networks. Otherwise equal to 1. 

public:
   Layer(int layer_size=1, std::string name="layer"); // allows for default constructor
   ~Layer();
   Layer(const Layer&);
   const Layer& operator=(const Layer&);
   virtual void noop() = 0;  // To make sure that the Layer class is not instantiated
   virtual void print(std::string msg="");
   virtual void printSummary(std::string msg="");
   virtual void printName(std::string msg="");

   virtual int  getInputDim() const { return input_dim; }
   // input from previous layer
   virtual void setInputDim(int input_dim) { this->input_dim = input_dim; }
   virtual int  getOutputDim() { return this->getLayerSize(); }
   virtual void setOutputDim(int output_dim) { this->setLayerSize(output_dim); }
   virtual Activation& getActivation() const { return *activation; }
   virtual void setActivation(Activation* activation) 
   { 
        this->activation = activation; 
		//activation.print("act");
		int p = activation->getNbParams();
		activation_delta.set_size(activation->getNbParams());
   } 

   /** Compute the outputs given the inputs */
   virtual void execute();

   /** get layer weights */
   //virtual void createWeights(int in, int out);  // not sure of data structure (Just simple matrix for now)
   //virtual void initializeWeights(std::string initialization_type="uniform");  // not sure of data structure
   //WEIGHTS getWeights() const {return weights;}  // not sure of data structure (Just simple matrix for now)

   // gradient of loss function with respect to weights
   GRADIENTS getGradient() const {return gradient;}
   BIAS& getBias() {return bias;}  // return  reference or const ref? 
   const BIAS& getBias() const {return bias;}  // return  reference or const ref? 
   BIAS& getBiasDelta() { return bias_delta; }
   const BIAS& getBiasDelta() const { return bias_delta; }
   // make private?
   void computeGradient();
   void computeGradient(int t); // for sequences

	int getNbBatch() { return nb_batch; }
   	void setNbBatch(int nb_batch) { this->nb_batch = nb_batch; initVars(nb_batch);  }
	//int getInputDim() const {return input_dim;}  // in reality, the layer size
  	//void setInputDim(int input_dim) {this->input_dim = input_dim;}
  	void setSeqLen(int seq_len) { this->seq_len = seq_len;}
  	int getSeqLen() const {return seq_len;}

	int getLayerSize() { return layer_size; }
	void setLayerSize(int layer_size) { this->layer_size = layer_size; }
	void setInputs(VF2D_F& inputs) { this->inputs = inputs; }
	const std::vector<VF2D_F>& getLayerInputs() { return layer_inputs; }
	const VF2D_F& getInputs() { return inputs; }
	const VF2D_F& getLoopInput() const { return loop_input; }
	VF2D_F& getLoopInput() { return loop_input; }
	void setOutputs(VF2D_F& outputs) { this->outputs = outputs; }
	VF2D_F& getOutputs() { return outputs; }
	void incrOutputs(VF2D_F& x);
	void incrOutputs(VF2D_F& x, int t); // for sequences
	void incrInputs(VF2D_F& x);
	void incrInputs(VF2D_F& x, int t);
	void resetInputs();
	void resetInputs(int t);
	void incrDelta(VF2D_F& x);
	void incrDelta(VF2D_F& x, int t);
	void incrBiasDelta(VF1D& x);
	void incrActivationDelta(VF1D& x);  // NOT sure whether argument if VF1D or VF2D_F or VF2D)
	const VF1D& getActivationDelta() {return activation_delta; }  // NOT sure whether argument if VF1D or VF2D_F or VF2D)
	// reset inputs and ouputs to zero
	void reset();
	// reset deltas
	void resetBackprop();
	void setName(std::string name) { this->name = name; } // normally not used
	std::string getName() { return name; }
	int getClock() { return clock; }
	void incrClock() { clock += 1; }
	const DELTA& getDelta() const { return delta; } // both this and the above function are required
	DELTA& getDelta() { return delta; }   // I might with to retrieve a pointer to delta and modify it directly (SECURITY RISK)
	const DELTA& getDelta(int which_lc) { return layer_deltas[which_lc]; }
	void setDelta(DELTA delta) { this->delta = delta; }
	void setDelta(DELTA delta, int which_lc) { layer_deltas[which_lc] = delta; } // CHANGE
	void resetDelta();
	void resetState();
	void setIsBiasFrozen(bool is_frozen) { this->is_bias_frozen = is_frozen; }
	bool getIsBiasFrozen() { return is_bias_frozen; }
	virtual void forwardData(Connection* conn, VF2D_F& prod, int seq);
	virtual void forwardLoops();
	virtual void forwardLoops(int t1, int t2);
	virtual void forwardLoops(int seq_index);
	virtual void forwardLoops(Connection* con, int t);
	virtual void forwardLoops(Connection* con, int t1, int t2);
	virtual void processData(Connection* conn, VF2D_F& prod);
	virtual bool areIncomingLayerConnectionsComplete();
	//virtual void processOutputDataFromPreviousLayer(Connection* conn, VF2D_F& prod);
	virtual void processOutputDataFromPreviousLayer(Connection* conn, VF2D_F& prod, int seq);
	virtual void addBiasToInput(int t);

	// Doing this in layers since the approach to matrix multiplication will depend on whether the activation function
	// gradient is done componentwise or via a Jacobian matrix
	#if 0
	virtual void gradMulDLda(VF2D_F& prod, const Connection& conn, int t_from, int t_to); // for now, do not store anything (at least until we know what is needed)
	virtual void dLdaMulGrad(Connection* con, int t);
	#endif
	//Connection* getConnection() {return recurrent_conn;}
	Connection* getConnection() {printf("Layer::getConnection: not implemented;\n"); exit(0); } //return recurrent_conn;}

public:
	virtual void initVars(int nb_batch);
};

//----------------------------------------------------------------------
/* use of this typedef requires inclusion of this file */
typedef std::vector<Layer*> LAYERS;
//----------------------------------------------------------------------

#endif
