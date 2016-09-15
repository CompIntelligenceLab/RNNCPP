#ifndef __WEIGHTS_H__
#define __WEIGHTS_H__

//#ifdef __APPLE__
  //#include <eigen3/Eigen>
//#elif __linux__
  //#include <eigen3/Eigen/Eigen>
//#endif

#include <vector>
#include <string>
#include "typedefs.h"

class Layer;

/** General structure to store weights. How to organize the data remains to be seen */
/** Possibly overload this class polymorphically */
/* It is possible that this class is not necessary. Not sure yet */

//----------------------------------------------------------------------
class Connection
{
public:
	Layer* from;  // pointers to the two layers that form the connection
	Layer* to;    // useful for some algorithms
	int hit;      // track whether a connection was hit (make private later perhaps)
	int which_lc; // which index into layer_connections
	bool freeze;  // do not update connections (NOT IMPLEMENTED). False by default
	// used to handle connections with delays
	// t_from: will be zero, but perhaps will acquire non-zero value in the future
	// t_to: the connection delay. A "signal" takes (t_to-t_from) units to reach Layer "from"
	// t_clock: set to t_from when a signal arrives at the "from" layer, incremented by one unit 
	// every clock cycle. When t_clock == t_to, the signal is transferred to "from" inputs. 
	// At the moment, time passes at the same rate in all connections. That might change in the future. 
	int t_from, t_to;
	int t_clock;

protected:
	static int counter;
	std::string name;
	WEIGHT weight;
	WEIGHT weight_t; // transpose of weight matrix: neede for backprop. 
	                 // Cost is reduced once batch > 1 and using sequences. 
	                 // Disadvantage: memory use is doubled. 
	WEIGHT delta;
	int in_dim, out_dim;
	bool print_verbose;
	bool temporal; // false: spatial link, true: temporal link
	int clock; // 0: weight has not been used. 1 otherwise. Potential problem if a weight is used twice, 
	           // which is possible if they are shared. Step 1: change Connection class to Connections class. . 
				 
    // passthrough (from/to layers have same layer_size, pass through information), weights not used. Not used. 
    // all-all, one-one,  [ DEFAULT ],  mxn weight
    // one-one: between two layers of equal size, with no weights. Requires the two layers to have the same size
	//          Weights can be frozen or not.
	std::string type; 

public:
	// input and output dimensions 
	static Connection* ConnectionFactory(int in_dim, int out_dim, std::string conn_type);
	Connection(int in, int out, std::string name="weight");
	~Connection();
	Connection(const Connection&);
	const Connection& operator=(const Connection& w);
	std::string getName() { return name; } // more generally, return a vector of weights
	void setWeight(WEIGHT w) { weight = w; } // 
	const WEIGHT& getWeight() const { return weight; } // more generally, return a vector of weights
	WEIGHT& getWeight() { return weight; } // more generally, return a vector of weights
	const WEIGHT& getWeightTranspose() const { return weight_t; } // more generally, return a vector of weights
	void computeWeightTranspose();
	void print(std::string name= "");
	void printSummary(std::string name= "");
	int getNRows() { return weight.n_rows; }
	int getNCols() { return weight.n_cols; }
	void setTemporal(bool temporal) { this->temporal = temporal;}
	bool getTemporal() {return temporal;}
    virtual void initialize(std::string initialization_type="xavier");  // not sure of data structure
	int getClock() { return clock; }
	void incrClock() { clock += 1; }
	void backProp() {;} // for future use
	void weightUpdate(REAL learning_rate);
	WEIGHT& getDelta() { return this->delta; }
	void setDelta(WEIGHT delta) { this->delta = delta; }
	void resetDelta() { delta.zeros(); }
	virtual void incrTClock() {t_clock++;}
	virtual void setTTo(int to) {t_to = to;}
	virtual int getTTo() { return t_to; }
	virtual void gradMulDLda(int ti_from, int ti_to);
	virtual void dLdaMulGrad(int t);
	//virtual void gradMulDLda(VF2D_F& prod, int ti_from, int ti_to);


	Connection(Connection&& w); // C++11

	// If I use operator+(const Connection&) const, I get the error: 
	// Error: no matching constructor for initialization of 'Connection'
	Connection operator+(const Connection&); // not needed, check dimensionality
	Connection operator*(const Connection&); // not needed, check dimensionality
	VF2D_F operator*(const VF2D_F&);
	REAL& operator()(const int i, const int j) { return weight(i,j); }
	const REAL& operator()(const int i, const int j) const { return weight(i,j); }
	void incrDelta(WEIGHT& x);
};


//----------------------------------------------------------------------
//typedef std::vector<Weights> WeightList;
//typedef std::vector<Connection> ConnectionList;

#endif

/*
*/
