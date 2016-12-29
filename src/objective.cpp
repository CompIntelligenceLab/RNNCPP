#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "print_utils.h"
#include "objective.h"
#include "typedefs.h"

int Objective::counter = 0;

Objective::Objective(std::string name /* "objective" */)
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("Objective::Objective : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("Objective constructor (%s)\n", this->name.c_str());

	learning_rate = 1.e-5; // arbitrary value

	for (int b=0; b < weight.n_rows; b++) {
		weight[b] = 1.;
	}
}

Objective::~Objective()
{
	printf("Objective destructor (%s)\n", name.c_str());
}

Objective::Objective(const Objective& o) : learning_rate(o.learning_rate)
{
	name = o.name + "c";
	printf("Objective copy constructor (%s)\n", name.c_str());
}

const Objective& Objective::operator=(const Objective& o)
{
	printf("Objective::operator= (%s)\n", o.name.c_str());

	if (this != &o) {
		learning_rate = o.learning_rate;
		name = o.getName() + "=";
	}
	return *this;
}

void Objective::print(std::string msg /* "" */)
{
	printf("*** Objective printout (%s): ***\n", name.c_str());
    if (msg != "") printf("%s\n", msg.c_str());
	printf("learning_rate: %f\n", learning_rate);
}

//----------------------------------------------------------------------
MeanSquareError::MeanSquareError(std::string name /* mse */) : Objective(name)
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("MeanSquare::MeanSquare : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	//printf("MeanSquareError constructor (%s)\n", this->name.c_str());
	counter++;
}

MeanSquareError::~MeanSquareError()
{
	printf("MeanSquareError destructor (%s)\n", name.c_str());
}

MeanSquareError::MeanSquareError(const MeanSquareError& mse) : Objective(mse)
{
}

void MeanSquareError::computeLoss(const VF2D_F& exact, const VF2D_F& predict)
{
	int nb_batch = exact.n_rows;
	loss.set_size(nb_batch); // needed
	VF2D tmp;

	// LOSS is a row vector. One value per sequence element

	//printf("error type: %s\n", error_type.c_str()); exit(0);

	if (error_type == "rel") {
		for (int b=0; b < nb_batch; b++) {
			tmp = (exact[b] - predict[b]) / exact[b];  // relative error
			tmp = arma::square(tmp);  // sum of output dimensions
			loss[b] = arma::sum(tmp, 0);  // sum over 1st index (dimension)
		}
	} else {
		for (int b=0; b < nb_batch; b++) {
			tmp = exact[b] - predict[b];  // check size compatibility
			tmp = arma::square(tmp);  // sum of output dimensions
			loss[b] = arma::sum(tmp, 0);  // sum over 1st index (dimension)
		}
	}
}

void MeanSquareError::computeGradient(const VF2D_F& exact, const VF2D_F& predict)
{
	// compute the gradient of L with respect to all outputs for every sequence element

	int nb_batch = exact.n_rows;
	gradient.set_size(nb_batch);

	if (error_type == "rel") {
		for (int b=0; b < nb_batch; b++) {
			gradient[b] = 2.* (predict[b] - exact[b]) / (exact[b] % exact[b]); // based on relative error
		}
	} else {
		for (int b=0; b < nb_batch; b++) {
			gradient[b] = 2.* (predict[b] - exact[b]); // check size compatibility
		}
	}
	//U::print(gradient, "computeGradient, MeanSquareError");
	gradient.print("loss gradient");
	//exit(0);
}
//----------------------------------------------------------------------
LogMeanSquareError::LogMeanSquareError(std::string name /* logmse */) 
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("LogMeanSquare::LogMeanSquare : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	//printf("LogMeanSquareError constructor (%s)\n", this->name.c_str());
	counter++;
}

LogMeanSquareError::~LogMeanSquareError()
{
	printf("LogMeanSquareError destructor (%s)\n", name.c_str());
}

LogMeanSquareError::LogMeanSquareError(const LogMeanSquareError& mse) : Objective(mse)
{
}

void LogMeanSquareError::computeLoss(const VF2D_F& exact, const VF2D_F& predict)
{
	int nb_batch = exact.n_rows;
	loss.set_size(nb_batch); // needed
	VF2D tmp;
	VF2D output(size(predict[0]));

	// LOSS is a row vector. One value per sequence element

	for (int b=0; b < nb_batch; b++) {
		tmp = exact[b] - predict[b];  // check size compatibility
		tmp = arma::square(tmp);  // sum of output dimensions
		output = arma::clamp(tmp, NEAR_ZERO, 1000.-NEAR_ZERO); 
		loss[b] = arma::sum(arma::log(output), 0);  // sum over 1st index (dimension)
	}
}

void LogMeanSquareError::computeGradient(const VF2D_F& exact, const VF2D_F& predict)
{
	// compute the gradient of L with respect to all outputs for every sequence element

	int nb_batch = exact.n_rows;
	gradient.set_size(nb_batch);
	VF2D output(size(predict[0]));
	VF2D tmp;


	for (int b=0; b < nb_batch; b++) {
		tmp = predict[b] - exact[b];
		tmp = arma::abs(tmp);
		output = arma::clamp(tmp, NEAR_ZERO, 1000.-NEAR_ZERO); 
		gradient[b] = 2. / output;  // element by element division
	}
}
//----------------------------------------------------------------------
BinaryCrossEntropy::BinaryCrossEntropy(std::string name /* bce */) 
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("MeanSquare::MeanSquare : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	//printf("BinaryCrossEntropy constructor (%s)\n", this->name.c_str());
	counter++;
}

BinaryCrossEntropy::~BinaryCrossEntropy()
{
	printf("BinaryCrossEntropy destructor (%s)\n", name.c_str());
}

BinaryCrossEntropy::BinaryCrossEntropy(const BinaryCrossEntropy& bce) : Objective(bce)
{
}

void BinaryCrossEntropy::computeLoss(const VF2D_F& exact, const VF2D_F& predict)
{
	int nb_batch = exact.n_rows;
	loss.set_size(nb_batch); // needed
	VF2D output(size(predict[0]));

	for (int b=0; b < nb_batch; b++) {
		// if predict is 0 or 1, loss goes to infinity. So clip. 
		output = arma::clamp(predict[b], NEAR_ZERO, 1.-NEAR_ZERO); 
		loss[b] = exact[b]*arma::log(output) + (1.-exact[b]) * arma::log(1.-output); // check size compatibility
		loss[b] = arma::sum(loss[b], 0);  // sum over 1st index (dimension)
	}
}


void BinaryCrossEntropy::computeGradient(const VF2D_F& exact, const VF2D_F& predict)
{
	int nb_batch = exact.n_rows;
	gradient.set_size(nb_batch);
	VF2D output(size(predict[0]));

	for (int b=0; b < nb_batch; b++) {
		output = arma::clamp(predict[b], NEAR_ZERO, 1.-NEAR_ZERO);  // prevent next line from going to infinity
		gradient[b] = exact[b] / output + (1-exact[b]) /(1-output);
	}
}
//----------------------------------------------------------------------
WeightedMeanSquareError::WeightedMeanSquareError(std::string name /* wmse */) : Objective(name)
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("WeightedMeanSquare::WeightedMeanSquare : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("WeightedMeanSquareError constructor (%s)\n", this->name.c_str());
	counter++;
}

WeightedMeanSquareError::~WeightedMeanSquareError()
{
	printf("WeightedMeanSquareError destructor (%s)\n", name.c_str());
}

WeightedMeanSquareError::WeightedMeanSquareError(const WeightedMeanSquareError& mse) : Objective(mse)
{
	printf("WeightedMeanSquareError copy constructor (%s)\n", this->name.c_str());
}

void WeightedMeanSquareError::computeLoss(const VF2D_F& exact, const VF2D_F& predict)
{
	//exact.print("exact");
	//U::print(exact, "exact");
	int nb_batch = exact.n_rows;
	int seq_len = exact[0].n_cols;
	loss.set_size(nb_batch); // needed
	VF2D tmp;
	int input_dim = exact[0].n_rows;

	//printf("seq_len= %d\n", seq_len);
	//U::createMat(weight, nb_batch, seq_len); // should probably only be done once unless seq_len or nb_batch changes
	weight.resize(nb_batch); // required? 
	weight.zeros();

	for (int b=0; b < nb_batch; b++) {
		for (int i=0; i < input_dim; i++) {
			for (int s=0; s < seq_len; s++) {
				weight[b] += exact[b](s,i);
			}
			weight[b] /= (seq_len*input_dim);
			weight[b] = 1. / (weight[b] + .05);
		}
	}

	// LOSS is a row vector. One value per sequence element

	for (int b=0; b < nb_batch; b++) {
		tmp = exact[b] - predict[b];  // check size compatibility
		tmp = arma::square(tmp);  // sum of output dimensions
		loss[b] = arma::sum(tmp, 0);  // sum over 1st index (dimension)
	}
	for (int b=0; b < nb_batch; b++) {
		//printf("b= %d\n", b);
		for (int s=0; s < exact[0].n_cols; s++) {
			//printf("s= %d\n", s);
			for (int in=0; in < exact[0].n_rows; in++) {
		//weight.print("weight");
		//U::print(this->weight, "weight"); // Not initialized
		//printf("in= %d\n", in);
		//exit(0);
		//U::print(loss, "loss");
				loss[b](in,s) = weight[b] * loss[b](in,s);
			}
		}
	}
	//U::print(loss, "computeLoss, WeightedMeanSquareError");
	//loss.print("MeanSquareError loss");
	//exit(0);
}

void WeightedMeanSquareError::computeGradient(const VF2D_F& exact, const VF2D_F& predict)
{
	// compute the gradient of L with respect to all outputs for every sequence element

	int nb_batch = exact.n_rows;
	gradient.set_size(nb_batch);

	for (int b=0; b < nb_batch; b++) {
		gradient[b] = VF2D(exact[0]);
		for (int s=0; s < gradient[0].n_cols; s++) {
			for (int in=0; in < gradient[0].n_rows; in++) {
				gradient[b](in,s) = 2. * weight[b] * (predict[b](in,s) - exact[b](in,s)); // check size compatibility
			}
		}
	}
	// ERROR: gradient is zero size. DO NOT KNOW WHY. 
	//U::print(gradient, "computeGradient, WeightedMeanSquareError");
	//gradient.print("WeightedMeanSquareError gradient");
	//exit(0);
}
//----------------------------------------------------------------------
CrossEntropy::CrossEntropy(std::string name /* bce */) 
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("MeanSquare::MeanSquare : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	counter++;
}

CrossEntropy::~CrossEntropy()
{
	printf("CrossEntropy destructor (%s)\n", name.c_str());
}

CrossEntropy::CrossEntropy(const CrossEntropy& bce) : Objective(bce)
{
}

//----------------------------------------------------------------------
void CrossEntropy::computeLoss(const VF2D_F& exact, const VF2D_F& predict)
{
// First compute softmax of prediction to transform the to probabilities

	int seq_len = exact[0].n_cols;
	int input_dim = exact[0].n_rows;
	int nb_batch = exact.n_rows;

	VF2D_F y(predict);

	// softmax over the dimension index of VF2D (first index)
	for (int b=0; b < nb_batch; b++) {
		REAL mx = arma::max(arma::max(predict[b]));
	    for (int s=0; s < seq_len; s++) {
		    y(b).col(s) = arma::exp(y(b).col(s)-mx);
			// trick to avoid overflows
			REAL ssum = 1. / arma::sum(y[b].col(s)); // = arma::exp(y[b]);
			y[b].col(s) = y[b].col(s) * ssum;  // % is elementwise multiplication (arma)
		}
	}
	//y.print("softmax, computeLoss");
	//exact.print("exact");
	//predict.print("predict");
	//exit(0);

	//printf("(%d, %d) CrossEntropy::computeLoss\n", seq_len, input_dim);
	//U::print(exact, "exact");
	//U::print(predict, "predict");
	//for (int b=0; b < 20; b++) {
		//printf("exact[%d]= %f, %f, pred[%d]= %f, %f, soft= %f, %f\n", b, exact[b](0,0), exact[b](1,0), b, predict[b](0,0), predict[b](1,0), y[b](0,0), y[b](1,0)); 
	//}


	loss.set_size(nb_batch); // needed
	VF2D output(size(predict[0]));

	// LOSS[batch][sequence]
	for (int b=0; b < nb_batch; b++) {
		//output = arma::clamp(predict[b], NEAR_ZERO, 1.-NEAR_ZERO); 
		output = arma::clamp(y[b], NEAR_ZERO, 1.-NEAR_ZERO); 

		loss[b].zeros(seq_len);
		for (int s=0; s < seq_len; s++) {
		   //printf("s= %d\n", s);

			// Sum over input_dim (most terms are zero)
			for (int i=0; i < input_dim; i++) {
				//printf("i= %d\n", i);
				loss[b](s) -= exact[b](i,s) * log(output(i,s));
				//printf("exact(%d,%d): %f, output(%d,%d)= %f\n", i,s,exact[b](i,s), i,s, output(i,s));
				// I should not have to do the sum, since exact is all zeros except one element, and 
				// I know which one
			}
		}
		// Really need the average over the sequence
		//loss[b] = loss[b] / seq_len; // orig
	}
	//loss.print("loss in CrossEntropy computeLoss\n");
		//exit(0);
	//loss.print("exit loss");
}

//----------------------------------------------------------------------
void CrossEntropy::computeGradient(const VF2D_F& exact, const VF2D_F& predict)
{
	//printf("**** enter crossEntropy Gradient ****\n");
	int nb_batch = exact.n_rows;
	int seq_len  = exact[0].n_cols;
	gradient.reset(); // empty the datastructure
	gradient.set_size(nb_batch);

	// WRONG: predict[b] must the the result of the softmax
	VF2D_F y(predict); // additional copy

	// softmax over the dimension index of VF2D (first index)
	for (int b=0; b < nb_batch; b++) {
		REAL mx = arma::max(arma::max(predict[b]));
	    for (int s=0; s < seq_len; s++) {
		    y(b).col(s) = arma::exp(y(b).col(s)-mx);
			// trick to avoid overflows
			REAL ssum = 1. / arma::sum(y[b].col(s)); // = arma::exp(y[b]);
			y[b].col(s) = y[b].col(s) * ssum;  // % is elementwise multiplication (arma)
		}
	}

	for (int b=0; b < nb_batch; b++) {
		//U::print(predict, "predict");
		//U::print(exact, "exact");
		//gradient[b] = (predict[b] - exact[b]) / seq_len; // average gradient
		gradient[b] = (y[b] - exact[b]); // average gradient
		// although all exact are zero except one (for a given sequence index), predict are all non-zero.
		// So I do not think there is a faster procedure to evaluate the gradient. 

		// Clip to [-5,5] to avoid exploding gradients
		gradient[b] = arma::clamp(gradient[b], -5., 5.);
	}
	//gradient(0).raw_print(arma::cout, "Cross-Entropy Gradient");
}
//----------------------------------------------------------------------
GMM1D::GMM1D(std::string name /* bce */) 
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("MeanSquare::MeanSquare : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	counter++;
}

GMM1D::~GMM1D()
{
	printf("GMM1D destructor (%s)\n", name.c_str());
}

GMM1D::GMM1D(const GMM1D& bce) : Objective(bce)
{
}

//----------------------------------------------------------------------
arma::Row<REAL> GMM1D::computeLossOneBatch(const VF2D& exact, const VF2D& predict)
{
// First compute softmax of prediction to transform the to probabilities

	// predict[batch][inputs, seq]
	// predict[batch][0:inputs/3, :] ==> amplitudes
	// predict[batch][inputs/3:2*inputs/3, :] ==> means
	// predict[batch][2*inputs/3:3*inputs/3, :] ==> standard deviations

	int seq_len = exact.n_cols;
	int input_dim = exact.n_rows;

	int ia1   = 0;
	int ia2   = input_dim / 3;
	int imu1  = input_dim / 3;
	int imu2  = 2* input_dim / 3;
	int isig1 = 2* input_dim / 3;
	int isig2 = 3* input_dim / 3;

	int Npi = ia2; // number of distributions

	VF2D pi(predict.rows(ia1,ia2));

	// softmax over the dimension index of VF2D (first index)
	REAL mx = arma::max(arma::max(pi));
	for (int s=0; s < seq_len; s++) {
	   pi.col(s) = arma::exp(pi.col(s)-mx);
	   // trick to avoid overflows
	   REAL ssum = 1. / arma::sum(pi.col(s)); // = arma::exp(y[b]);
	   pi.col(s) = pi.col(s) * ssum;  // % is elementwise multiplication (arma)
	}

	// standard deviations: exp(output) => sig

	VF2D sig(predict.rows(isig1, isig2));
	sig = arma::exp(sig);
	VF2D mu(predict.rows(imu1, imu2));

	// Compute sum of N probabilities
	VF2D prob(pi);
	prob.zeros();
	VF2D x(exact);

	// ignore factor 1/sqrt(2.*pi)
	prob = pi * arma::exp(-arma::square(x-mu) / (sig%sig)) / arma::sqrt(sig);

	arma::Row<REAL> sprob(seq_len);
	//VF2D sprob(1, seq_len);
	sprob.zeros();

	for (int r=0; r < prob.n_rows; r++) {
		sprob += prob.row(r);
	}
	sprob = arma::log(sprob);

	#if 0
	REAL cost = 0.;
	for (int s=0; s < seq_len; s++) {
		cost += sprob(s);
	}
	#endif

	return sprob;
}
//----------------------------------------------------------------------
void GMM1D::computeLoss(const VF2D_F& exact, const VF2D_F& predict)
{
	int nb_batch = predict.n_rows;
	VF2D_F loss(nb_batch);

	for (int b=0; b < nb_batch; b++) {
		loss(b) = computeLossOneBatch(exact(b), predict(b));
	}
}

//----------------------------------------------------------------------
VF2D GMM1D::computeGradientOneBatch(const VF2D& exact, const VF2D& predict)
{
// First compute softmax of prediction to transform the to probabilities

	// predict[batch][inputs, seq]
	// predict[batch][0:inputs/3, :] ==> amplitudes
	// predict[batch][inputs/3:2*inputs/3, :] ==> means
	// predict[batch][2*inputs/3:3*inputs/3, :] ==> standard deviations

	int seq_len = exact.n_cols;
	int input_dim = exact.n_rows;

	int ia1   = 0;
	int ia2   = input_dim / 3;
	int imu1  = input_dim / 3;
	int imu2  = 2* input_dim / 3;
	int isig1 = 2* input_dim / 3;
	int isig2 = 3* input_dim / 3;

	int Npi = ia2; // number of distributions

	VF2D pi(predict.rows(ia1,ia2));

	// softmax over the dimension index of VF2D (first index)
	REAL mx = arma::max(arma::max(pi));
	for (int s=0; s < seq_len; s++) {
	   pi.col(s) = arma::exp(pi.col(s)-mx);
	   // trick to avoid overflows
	   REAL ssum = 1. / arma::sum(pi.col(s)); // = arma::exp(y[b]);
	   pi.col(s) = pi.col(s) * ssum;  // % is elementwise multiplication (arma)
	}

	// standard deviations: exp(output) => sig

	VF2D sig(predict.rows(isig1, isig2));
	sig = arma::exp(sig);
	VF2D mu(predict.rows(imu1, imu2));

	// Compute sum of N probabilities
	VF2D prob(pi);
	prob.zeros();
	VF2D x(exact);

	// ignore factor 1/sqrt(2.*pi)
	prob = pi * arma::exp(-arma::square(x-mu) / (sig%sig)) / arma::sqrt(sig);

	// transform prob into softmax functions, one per sequence index

	VF2D yprob(prob);

	// softmax over the dimension index of VF2D (first index)
	mx = arma::max(arma::max(yprob));
	for (int s=0; s < seq_len; s++) {
	   yprob.col(s) = arma::exp(yprob.col(s)-mx);
	   // trick to avoid overflows
	   REAL ssum = 1. / arma::sum(yprob.col(s)); // = arma::exp(y[b]);
	   yprob.col(s) = yprob.col(s) * ssum;  // % is elementwise multiplication (arma)
	}

	VF2D dLdpi  = pi - yprob;
	VF2D dLdmu  = -yprob * (x-mu) / sig;
	VF2D dLdsig = -arma::square(yprob * (x-mu) / sig) - 1.;

	// Combine the derivatives into one vector. 
	VF2D grad(size(predict));
	grad.cols(ia1,ia2)     = dLdpi;
	grad.cols(imu1,imu2)   = dLdmu;
	grad.cols(isig1,isig2) = dLdsig;
	
	return grad;
}
//----------------------------------------------------------------------
void GMM1D::computeGradient(const VF2D_F& exact, const VF2D_F& predict)
{
	int nb_batch = predict.n_rows;
	VF2D_F loss(nb_batch);

	gradient.reset(); // empty the datastructure
	gradient.set_size(nb_batch);

	for (int b=0; b < nb_batch; b++) {
		gradient(b) = computeGradientOneBatch(exact(b), predict(b));
	}
}
//----------------------------------------------------------------------
