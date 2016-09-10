#ifndef __TYPEDEFS__H_
#define __TYPEDEFS__H_

#include <vector>

// Disable bound checks
//#define ARMA_NO_DEBUG

#include <armadillo>
#include <stdio.h>

#define ARMADILLO
//#define EIGEN


#include <vector>
#if 0
#ifdef __APPLE__
#include <eigen/Eigen>
#elif __linux__
#include <eigen3/Eigen/Eigen>
#endif
#endif

// Add additional print information if verpose is 1
#define VERBOSE 0

class Layer;
//class Weights;
class Connection;

  // The user is not suppose to use eigen or armadillo. Therefore to input data
  // they will need some other mechanism. Eventually we should write functions
  // to parse data files so the user need only specify a file. For testting
  // purposes, now I will specify a simply matrix like structure for the
  // training data
  typedef std::vector< std::vector<float> > MATRIX;

#ifdef ARMADILLO

#define ZEROS(field) for (int i=0; i < field.size(); i++) { field[i].zeros(); }

	typedef std::vector<Connection*> CONNECTIONS;
	typedef std::vector<std::pair<Layer*, Connection*> > PAIRS_L; 
	// 1st argument: input to a layer, or output to a connection
	// 2nd argument: connection type, or clock or counter
	typedef arma::Col<float> AF;
	typedef arma::Col<float> VF;
	typedef arma::Col<float> VF1D;
	typedef arma::field<arma::Row<float> > LOSS;
	typedef arma::Mat<float> VF2D;
	typedef arma::field<arma::Col<float> > VF1D_F;  // [batch](seq_len) 
	typedef arma::field<arma::Mat<float> > VF2D_F;  // [batch](dimension, seq_len) 
	typedef std::vector<std::pair<VF2D_F, int> > LAYER_INPUTS;
	typedef arma::Cube<float> VF3D;
	typedef arma::Row<int> VI; // not possible to allocate a row of size 3;
	//typedef arma::Mat<int> VI;
	typedef arma::Mat<float> WEIGHT;  // (layer(j-1), layer(j))
	typedef arma::Mat<float>* WEIGHTP;  // pointers will help share weights between connections
	typedef arma::Col<float> BIAS;  // 
	typedef arma::Col<float>* BIASP;  // pointers to bias will help with bias sharing
	typedef VF2D_F DELTA;  // (layer(j-1), layer(j))
    typedef VF2D_F GRADIENTS; // Remove this if we use the Gradients class
	// Do not forget () around arguments in macro definition
	#define APPROX_EQUAL(x,y)  arma::approx_equal((x),(y),"absdiff",.001);
#else
	typedef Eigen::ArrayXf AF;
	typedef Eigen::VectorXf VF;
	typedef Eigen::Vector3i  VI3;
	typedef Eigen::MatrixXf WEIGHTS;
#endif

//#define WEIGHT  	VF2D
//#define DELTA   	VF1D
//#define GRADIENTS   VF2D

#define NEAR_ZERO 1.e-4


// assumes sequential model (for now)

#endif
