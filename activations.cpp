#include "activations.h"

Activation::Activation() 
{
}
//----------------------------------------------------------------------
Activation::~Activation() 
{
}
//----------------------------------------------------------------------
void Activation::print()
{
	printf("activation: name= %s\n", this->name.c_str());
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
