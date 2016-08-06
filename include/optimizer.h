#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include <string>

class Optimizer
{
protected:
	float learning_rate;
	std::string name;

public:
	Optimizer(std::string name="optimizer");
	~Optimizer();
	Optimizer(Optimizer&);
	virtual void print();
};

#endif
