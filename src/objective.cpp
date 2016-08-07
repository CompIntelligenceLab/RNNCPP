#include "objective.h"

Objective::Objective(std::string name)
{
	this->name = name;
}

Objective::~Objective()
{
}

Objective::Objective(const Objective& o) : learning_rate(o.learning_rate)
{
	name = o.name + "c";
}

Objective& Objective::operator=(const Objective& o)
{
	printf("Objective::operator= (%s)\n", name.c_str());
	if (this != &o) {
		learning_rate = o.learning_rate;
		name = o.getName() + "c";
	}
	return *this;
}

/*
const Objective& Objective::operator=(const Objective& o) const
{
	printf("obj=\n");
	learning_rate = o.learning_rate;
	name = o.getName() + "c";
	printf("exit obj=\n");
}
*/

void Objective::print(std::string msg)
{
	printf("*** Objective printout (%s): ***\n", name.c_str());
    if (msg != "") printf("%s\n", msg.c_str());
	printf("learning_rate: %f\n", learning_rate);
}
