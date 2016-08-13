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

/** General structure to store weights. How to organize the data remains to be seen */
/** Possibly overload this class polymorphically */
/* It is possible that this class is not necessary. Not sure yet */

#if 0
class Weights
{
protected:
	static int counter;
	std::string name;
	WEIGHTS weights;
	WEIGHTS_F weights_f; // using fields
	int in_dim, out_dim;
	bool print_verbose;
	bool temporal; // false: spatial link, true: temporal link
	int clock; // 0: weight has not been used. 1 otherwise. Potential problem if a weight is used twice, 
	           // which is possible if they are shared. Step 1: change Weights class to Connections class. . 
			   // step 2: 

public:
	// input and output dimensions 
	Weights(int in, int out, std::string name="weights");
	~Weights();
	Weights(const Weights&);
	const Weights& operator=(const Weights& w);
	const WEIGHTS& getWeights() const { return weights; }
	void print(std::string name= "");
	WEIGHTS_F& getWeightsF() { return  weights_f; }
	int getNRows() { return weights.n_rows; }
	int getNCols() { return weights.n_cols; }
	void setTemporal(bool temporal) { this->temporal = temporal;}
	bool getTemporal() {return temporal;}
    virtual void initialize(std::string initialization_type="uniform");  // not sure of data structure

	Weights(Weights&& w); // C++11

	// If I use operator+(const Weights&) const, I get the error: 
	// Error: no matching constructor for initialization of 'Weights'
	Weights operator+(const Weights&); // not needed, check dimensionality
	Weights operator*(const Weights&); // not needed, check dimensionality
	VF2D_F operator*(const VF2D_F&);
	float& operator()(const int i, const int j) { return weights(i,j); }
	//float& operator()(const int b, const int i, const int j) { return weights[b](i,j); }
};
#endif
//----------------------------------------------------------------------
class Connection
{
protected:
	static int counter;
	std::string name;
	WEIGHTS weights;
	WEIGHTS_F weights_f; // using fields
	int in_dim, out_dim;
	bool print_verbose;
	bool temporal; // false: spatial link, true: temporal link
	int clock; // 0: weight has not been used. 1 otherwise. Potential problem if a weight is used twice, 
	           // which is possible if they are shared. Step 1: change Connection class to Connections class. . 
			   // step 2: 

public:
	// input and output dimensions 
	Connection(int in, int out, std::string name="weights");
	~Connection();
	Connection(const Connection&);
	const Connection& operator=(const Connection& w);
	const WEIGHTS& getWeights() const { return weights; } // more generally, return a vector of weights
	void print(std::string name= "");
	WEIGHTS_F& getConnectionF() { return  weights_f; }
	int getNRows() { return weights.n_rows; }
	int getNCols() { return weights.n_cols; }
	void setTemporal(bool temporal) { this->temporal = temporal;}
	bool getTemporal() {return temporal;}
    virtual void initialize(std::string initialization_type="uniform");  // not sure of data structure

	Connection(Connection&& w); // C++11

	// If I use operator+(const Connection&) const, I get the error: 
	// Error: no matching constructor for initialization of 'Connection'
	Connection operator+(const Connection&); // not needed, check dimensionality
	Connection operator*(const Connection&); // not needed, check dimensionality
	VF2D_F operator*(const VF2D_F&);
	float& operator()(const int i, const int j) { return weights(i,j); }
	//float& operator()(const int b, const int i, const int j) { return weights[b](i,j); }
};


//----------------------------------------------------------------------
//typedef std::vector<Weights> WeightList;
typedef std::vector<Connection> ConnectionList;

#endif
