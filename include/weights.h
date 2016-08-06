#ifndef __WEIGHTS_H__
#define __WEIGHTS_H__

//#ifdef __APPLE__
  //#include <eigen3/Eigen>
//#elif __linux__
  //#include <eigen3/Eigen/Eigen>
//#endif

#include <vector>
#include "typedefs.h"

/** General structure to store weights. How to organize the data remains to be seen */
/** Possibly overload this class polymorphically */
class Weights
{
private:
std::string name;
WEIGHTS* weights;
int in_dim, out_dim;

public:
	// input and output dimensions 
	Weights(int in, int out);
	~Weights();
	Weights(Weights&);
	void initialize();
	void print();
};

//----------------------------------------------------------------------
typedef std::vector<Weights> WeightList;

#endif
