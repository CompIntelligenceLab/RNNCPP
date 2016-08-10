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

class Weights
{
protected:
	static int counter;
	std::string name;
	WEIGHTS weights;
	WEIGHTS_F weights_f; // using fields
	int in_dim, out_dim;
	bool print_verbose;

public:
	// input and output dimensions 
	Weights(int in, int out, std::string name="weights");
	~Weights();
	Weights(const Weights&);
	const Weights& operator=(const Weights& w);
	WEIGHTS& getWeights() { return weights; }
	void initializeWeights(std::string initialize_type="uniform");
	void print(std::string name= "");
	WEIGHTS_F& getWeightsF() { return  weights_f; }
	int getNRows() { return weights.n_rows; }
	int getNCols() { return weights.n_cols; }

	// If I use operator+(const Weights&) const, I get the error: 
	// Error: no matching constructor for initialization of 'Weights'
	Weights operator+(const Weights&); // not needed, check dimensionality
	Weights operator*(const Weights&); // not needed, check dimensionality
	VF2D_F operator*(const VF2D_F&);
	float& operator()(const int i, const int j) { return weights(i,j); }
	//float& operator()(const int b, const int i, const int j) { return weights[b](i,j); }
};

//----------------------------------------------------------------------
typedef std::vector<Weights> WeightList;

#endif
