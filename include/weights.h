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
class Weights
{
protected:
	std::string name;
	WEIGHTS* weights;
	int in_dim, out_dim;

public:
	// input and output dimensions 
	Weights(int in, int out, std::string name="weights");
	~Weights();
	Weights(Weights&);
	void initialize();
	void print(std::string name= "");
};

//----------------------------------------------------------------------
typedef std::vector<Weights> WeightList;

#endif
