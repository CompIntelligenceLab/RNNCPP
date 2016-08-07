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
/** This class will slow the program down: yet another level of indirection */

class Weights
{
protected:
	static int counter;
	std::string name;
	WEIGHTS* weights;
	int in_dim, out_dim;
	bool print_verbose;

public:
	// input and output dimensions 
	Weights(int in, int out, std::string name="weights");
	~Weights();
	Weights(const Weights&);
	WEIGHTS* getWeights() { return weights; } 
	void initializeWeights(std::string initialize_type="uniform");
	void print(std::string name= "");
};

//----------------------------------------------------------------------
typedef std::vector<Weights> WeightList;

#endif
