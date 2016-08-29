#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "print_utils.h"
#include "objective.h"

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

	for (int i=0; i < nb_batch; i++) {
		loss[i] = exact[i] - predict[i]; // check size compatibility
		loss[i] = arma::square(loss[i]);  // sum of output dimensions
		loss[i] = arma::sum(loss[i], 0);  // sum over 1st index (dimension)
		//loss[i].print("loss[i]");
		//exit(0);
	}
	//exit(0);
}

void MeanSquareError::computeGradient(const VF2D_F& exact, const VF2D_F& predict)
{
	int nb_batch = exact.n_rows;
	gradient.set_size(nb_batch);

	printf("gordon\n");
	U::print(exact, "exact");
	U::print(predict, "predict");

	for (int i=0; i < nb_batch; i++) {
		gradient[i] = 2.* (exact[i] - predict[i]); // check size compatibility
	}
}

