#include <stdio.h>
#include "objective.h"

Objective::Objective(std::string name)
{
	this->name = name;
}

Objective::~Objective()
{
	printf("Objective destructor (%s)\n", name.c_str());
}

Objective::Objective(const Objective& o) : learning_rate(o.learning_rate)
{
	printf("Objective copy constructor (%s)\n", o.name.c_str());
	name = o.name + "c";
}

Objective& Objective::operator=(const Objective& o)
{
	printf("Objective::operator= (%s)\n", o.name.c_str());

	if (this != &o) {
		learning_rate = o.learning_rate;
		name = o.getName() + "=";
	}
	return *this;
}

void Objective::print(std::string msg)
{
	printf("*** Objective printout (%s): ***\n", name.c_str());
    if (msg != "") printf("%s\n", msg.c_str());
	printf("learning_rate: %f\n", learning_rate);
}
