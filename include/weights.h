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
	//WEIGHTS weights;
	WEIGHTS_F weights_f; // using fields
	int in_dim, out_dim;
	bool print_verbose;

public:
	// input and output dimensions 
	Weights(int in, int out, std::string name="weights");
	~Weights();
	Weights(const Weights&);
	const Weights& operator=(const Weights& w);
	//WEIGHTS& getWeights() { return weights; }
	void initializeWeights(std::string initialize_type="uniform");
	void print(std::string name= "");
	WEIGHTS_F& getWeightsF() { return  weights_f; }

	// If I use operator+(const Weights&) const, I get the error: 
	// Error: no matching constructor for initialization of 'Weights'
	Weights operator+(const Weights&);
	Weights operator*(const Weights&);
	float& operator()(const int i, const int j) { return weights_f[0](i,j); }
	float& operator()(const int b, const int i, const int j) { return weights_f[b](i,j); }
};

//----------------------------------------------------------------------
typedef std::vector<Weights> WeightList;

#endif
