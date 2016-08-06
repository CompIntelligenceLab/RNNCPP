#include "activations.h"

Activation::Activation(std::string name) 
{
	this->name = name;
}
//----------------------------------------------------------------------
Activation::~Activation() 
{
}
//----------------------------------------------------------------------
void Activation::print(std::string msg)
{
	printf("activation: name= %s\n", this->name.c_str());
	if (msg != "") printf("%s\n", msg.c_str());
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
