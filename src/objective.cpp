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
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	//printf("Objective constructor (%s)\n", this->name.c_str());

	learning_rate = 1.e-5; // arbitrary value
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

MeanSquareError::MeanSquareError(std::string name /* mse */) 
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

//MeanSquareError::MeanSquareError=(const MeanSquareError& mse)
//{
	//if (this != &mse) {
		//name = o.getName() + "=";
	//}
	//return *this;
//}

void MeanSquareError::computeLoss(const VF2D_F& exact, const VF2D_F& predict)
{
	int nb_batch = exact.n_rows;
	loss.set_size(nb_batch); // needed
	VF2D tmp;

	//U::print(exact, "1 exact");
	//U::print(exact(0), "2 exact(0)");
	//U::print(predict, "3 predict");

	// LOSS is a row vector. One value per sequence element

	for (int b=0; b < nb_batch; b++) {
		//U::print(loss(b), "4 loss(b)");  
		//exact[b].print("5 exact(b)");
		//predict[b].print("6 predict(b)");

		tmp = exact[b] - predict[b];  // check size compatibility
		//tmp.print("tmp");
		tmp = arma::square(tmp);  // sum of output dimensions
		//tmp.print("tmp");
		loss[b] = arma::sum(tmp, 0);  // sum over 1st index (dimension)
		//tmp.print("tmp");
	}
	#if 0
	for (int b=0; b < nb_batch; b++) {
		//U::print(loss(b), "4 loss(b)");  
		//exact[b].print("5 exact(b)");
		//predict[b].print("6 predict(b)");

		//tmp = exact[b] - predict[b];  // check size compatibility
		//tmp = arma::square(tmp);  // sum of output dimensions
		//loss[b] = arma::sum(tmp, 0);  // sum over 1st index (dimension)
		loss[b] = exact[b] - predict[b];  // check size compatibility
		loss[b].print("loss[b]");
		loss[b] = arma::square(loss[b]);  // sum of output dimensions
		loss[b].print("loss[b]");
		loss[b] = arma::sum(loss[b], 0);  // sum over 1st index (dimension)
		loss[b].print("loss[b]");
	}
	#endif
}

void MeanSquareError::computeGradient(const VF2D_F& exact, const VF2D_F& predict)
{
	// compute the gradient of L with respect to all outputs for every sequence element

	int nb_batch = exact.n_rows;
	gradient.set_size(nb_batch);

	for (int b=0; b < nb_batch; b++) {
		gradient[b] = 2.* (predict[b] - exact[b]); // check size compatibility
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

//BinaryCrossEntropy::BinaryCrossEntropy=(const BinaryCrossEntropy& bce)
//{
	//if (this != &bce) {
		//name = o.getName() + "=";
	//}
	//return *this;
//}

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
