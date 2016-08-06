#ifndef __TYPEDEFS__H_
#define __TYPEDEFS__H_

#include <vector>

#include <vector>
#ifdef __APPLE__
#include <eigen/Eigen>
#elif __linux__
#include <eigen3/Eigen/Eigen>
#endif

typedef Eigen::ArrayXf AF;
typedef Eigen::VectorXf VF;
typedef Eigen::Vector3i  VI3;
typedef Eigen::MatrixXf WEIGHTS;

// assumes sequential model (for now)

#endif
