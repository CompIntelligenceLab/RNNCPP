#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
	U::print(loss, "computeLoss, MeanSquareError");
	loss.print("loss gradient");
	//exit(0);
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
	U::print(gradient, "computeGradient, MeanSquareError");
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
				weight[b] += exact[b][s,i];
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
				loss[b][in,s] = weight[b] * loss[b][in,s];
			}
		}
	}
	U::print(loss, "computeLoss, WeightedMeanSquareError");
	loss.print("MeanSquareError loss");
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
				gradient[b][in,s] = 2. * weight[b] * (predict[b][in,s] - exact[b][in,s]); // check size compatibility
			}
		}
	}
	// ERROR: gradient is zero size. DO NOT KNOW WHY. 
	U::print(gradient, "computeGradient, WeightedMeanSquareError");
	gradient.print("WeightedMeanSquareError gradient");
	//exit(0);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
