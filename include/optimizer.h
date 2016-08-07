#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include <string>
#include "typedefs.h"

class Optimizer
{
protected:
	float learning_rate;
	std::string name;
	VF loss;  // allows for batches, with dimensionality of 1.  

public:
	Optimizer(std::string name="optimizer");
	~Optimizer();
	Optimizer(const Optimizer&);
	Optimizer& operator=(const Optimizer&);
	//Optimizer& operator=(const Optimizer&);

	virtual void print(std::string msg="");
	virtual void setName(std::string name="optimizer") { this->name = name; }
	virtual const std::string getName() const { return name; }
	virtual void setLearningRate(float lr) {learning_rate = lr; }
	virtual float getLearningRate() {return learning_rate; }
	virtual VF getLoss() {return loss; }
};

//-------------------------------------------

class RMSProp : public Optimizer
{
public:
	RMSProp(std::string name="RMSProp");
	~RMSProp();
	RMSProp(const RMSProp&);
};

#endif
