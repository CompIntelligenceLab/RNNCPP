#ifndef __TYPEDEFS__H_
#define __TYPEDEFS__H_

#include <vector>
#include <armadillo>

#define ARMADILLO
//#define EIGEN


#include <vector>
#ifdef __APPLE__
#include <eigen/Eigen>
#elif __linux__
#include <eigen3/Eigen/Eigen>
#endif

#ifdef ARMADILLO
	typedef arma::Col<float> AF;
	typedef arma::Col<float> VF;
	typedef arma::Row<int> VI; // not possible to allocate a row of size 3;
	//typedef arma::Mat<int> VI;
	typedef arma::Mat<float> WEIGHTS;
#else
	typedef Eigen::ArrayXf AF;
	typedef Eigen::VectorXf VF;
	typedef Eigen::Vector3i  VI3;
	typedef Eigen::MatrixXf WEIGHTS;
#endif

// assumes sequential model (for now)

#endif
