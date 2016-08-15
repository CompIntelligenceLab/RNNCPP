#ifndef __TYPEDEFS__H_
#define __TYPEDEFS__H_

#include <vector>
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
	typedef std::vector<Connection*> CONNECTIONS;
	typedef std::vector<std::pair<Layer*, Connection*> > PAIRS_L; 
	typedef arma::Col<float> AF;
	typedef arma::Col<float> VF;
	typedef arma::Col<float> VF1D;
	typedef arma::Mat<float> VF2D;
	typedef arma::field<arma::Col<float> > VF1D_F;  // [batch](seq_len) 
	typedef arma::field<arma::Mat<float> > VF2D_F;  // [batch](dimension, seq_len) 
	typedef arma::Cube<float> VF3D;
	typedef arma::Row<int> VI; // not possible to allocate a row of size 3;
	//typedef arma::Mat<int> VI;
	typedef arma::Mat<float> WEIGHT;  // (layer(j-1), layer(j))
	typedef arma::Col<float> DELTA;  // partial of error w.r.t output to each "next" layer
    typedef arma::Mat<float> GRADIENTS; // Remove this if we use the Gradients class
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


// assumes sequential model (for now)

#endif
