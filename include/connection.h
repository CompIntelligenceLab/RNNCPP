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

protected:
	static int counter;
	std::string name;
	WEIGHT weight;
	WEIGHT delta;
	int in_dim, out_dim;
	bool print_verbose;
	bool temporal; // false: spatial link, true: temporal link
	int clock; // 0: weight has not been used. 1 otherwise. Potential problem if a weight is used twice, 
	           // which is possible if they are shared. Step 1: change Connection class to Connections class. . 
			   // step 2: 

public:
	// input and output dimensions 
	Connection(int in, int out, std::string name="weight");
	~Connection();
	Connection(const Connection&);
	const Connection& operator=(const Connection& w);
	std::string getName() { return name; } // more generally, return a vector of weights
	void setWeight(WEIGHT w) { weight = w; } // 
	WEIGHT& getWeight() { return weight; } // more generally, return a vector of weights
	void print(std::string name= "");
	void printSummary(std::string name= "");
	int getNRows() { return weight.n_rows; }
	int getNCols() { return weight.n_cols; }
	void setTemporal(bool temporal) { this->temporal = temporal;}
	bool getTemporal() {return temporal;}
    virtual void initialize(std::string initialization_type="uniform");  // not sure of data structure
	int getClock() { return clock; }
	void incrClock() { clock += 1; }
	void backProp() {;} // for future use
	void weightUpdate(float learning_rate);
	WEIGHT& getDelta() { return this->delta; }
	void setDelta(WEIGHT delta) { this->delta = delta; }
	void resetDelta() { delta.zeros(); }


	Connection(Connection&& w); // C++11

	// If I use operator+(const Connection&) const, I get the error: 
	// Error: no matching constructor for initialization of 'Connection'
	Connection operator+(const Connection&); // not needed, check dimensionality
	Connection operator*(const Connection&); // not needed, check dimensionality
	VF2D_F operator*(const VF2D_F&);
	float& operator()(const int i, const int j) { return weight(i,j); }
	void incrDelta(WEIGHT& x);
};


//----------------------------------------------------------------------
//typedef std::vector<Weights> WeightList;
//typedef std::vector<Connection> ConnectionList;

#endif

/*
*/
