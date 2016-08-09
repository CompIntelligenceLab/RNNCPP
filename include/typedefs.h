#ifndef __TYPEDEFS__H_
#define __TYPEDEFS__H_

#include <vector>
#include <armadillo>
#include <stdio.h>

#define ARMADILLO
//#define EIGEN


#include <vector>
#ifdef __APPLE__
#include <eigen/Eigen>
#elif __linux__
#include <eigen3/Eigen/Eigen>
#endif

  // The user is not suppose to use eigen or armadillo. Therefore to input data
  // they will need some other mechanism. Eventually we should write functions
  // to parse data files so the user need only specify a file. For testting
  // purposes, now I will specify a simply matrix like structure for the
  // training data
  typedef std::vector< std::vector<float> > MATRIX;

#ifdef ARMADILLO
	typedef arma::Col<float> AF;
	typedef arma::Col<float> VF;
	typedef arma::Col<float> VF1D;
	typedef arma::Mat<float> VF2D;
	typedef arma::Cube<float> VF3D;
	typedef arma::Row<int> VI; // not possible to allocate a row of size 3;
	//typedef arma::Mat<int> VI;
	typedef arma::Mat<float> WEIGHTS;
  typedef arma::Mat<float> GRADIENTS; // Remove this if we use the Gradients class
#else
	typedef Eigen::ArrayXf AF;
	typedef Eigen::VectorXf VF;
	typedef Eigen::Vector3i  VI3;
	typedef Eigen::MatrixXf WEIGHTS;
#endif

// assumes sequential model (for now)

#endif
