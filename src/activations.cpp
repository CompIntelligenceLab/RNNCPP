#include "activations.h"
#include <stdio.h>

int Activation::counter = 0;

Activation::Activation(std::string name /* "activation" */) 
{
	char cname[80];

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("Activation constructor (%s)\n", this->name.c_str());
	counter++;
}
//----------------------------------------------------------------------
Activation::~Activation() 
{
	printf("Activation destructor (%s)\n", name.c_str());
}
//----------------------------------------------------------------------
void Activation::print(std::string msg /* "" */)
{
	printf("Activation: name= %s\n", this->name.c_str());
	if (msg != "") printf("%s\n", msg.c_str());
}
//----------------------------------------------------------------------
Activation::Activation(const Activation& a)
{
	char cname[80];

	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	counter++;
}

const Activation& Activation::operator=(const Activation& a)
{
	if (this != &a) {
		name = a.name;
	}
	return *this;
}


Tanh::~Tanh() 
{
	printf("Tanh constructor (%s)\n", this->name.c_str());
}

Tanh::Tanh(const Tanh& t) : Activation(t)
{ }

const Tanh& Tanh::operator=(const Tanh& t)
{
	if (this != &t) {
		name = t.name + '=';
	}
	return *this;
}


Sigmoid::~Sigmoid()
{
}
 
Sigmoid::Sigmoid(const Sigmoid& s) : Activation(s)
{
	printf("Sigmoid constructor (%s)\n", this->name.c_str());
}

const Sigmoid& Sigmoid::operator=(const Sigmoid& s)
{
	if (this != &s) {
		name = s.name + '=';
	}
	return *this;
}

//----------------------------------------------------------------------
