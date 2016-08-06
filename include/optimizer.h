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

	virtual void print(std::string msg="");
	virtual void setLearningRate(float lr) {learning_rate = lr; }
	virtual float getLearningRate(float lr) {return learning_rate; };
};

//-------------------------------------------

class RMSProp : public Optimizer
{
public:
	RMSProp(std::string name="RMSProp");
	~RMSProp();
	RMSProp(RMSProp&);
};

#endif
