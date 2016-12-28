#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include <string>
#include "typedefs.h"

class Model;

class Optimizer
{
protected:
	REAL learning_rate;
	std::string name;
	VF loss;  // allows for batches, with dimensionality of 1.  
	static int counter;
	REAL beta1, beta2, alpha, eps; // for Adam

public:
	Optimizer(std::string name="optimizer");
	~Optimizer();
	Optimizer(const Optimizer&);
	const Optimizer& operator=(const Optimizer&);
	//Optimizer& operator=(const Optimizer&);

	virtual void print(const std::string msg=std::string());
	virtual void setName(std::string name) { this->name = name; }
	virtual const std::string getName() const { return name; }
	virtual void setLearningRate(REAL lr) {learning_rate = lr; }
	virtual REAL getLearningRate() {return learning_rate; }
	virtual VF getLoss() {return loss; }
	//virtual VF2D_F update(VF2D& w) {;}
	virtual void update(Model* mo, VF2D& w, VF2D& m, VF2D& v, VF2D& dLdw, int& count);
};

//-------------------------------------------
class RMSProp : public Optimizer
{
public:
	RMSProp(std::string name="RMSProp");
	~RMSProp();
	RMSProp(const RMSProp&);
};
//-------------------------------------------
class Adam : public Optimizer
{
public:
	//Adam(REAL alpha=.001, REAL beta1=.9, REAL beta2=.999, REAL eps=1.e-8);
	Adam(std::string name="adam");
	~Adam();
	Adam(const Adam&);
	//Adam& operator=(const Adam&);
	//VF2D_F update(VF2D& w);
	void update(Model* mo, VF2D& w, VF2D& m, VF2D& v, VF2D& dLdw, int& count);
};
//-------------------------------------------
class Adagrad : public Optimizer
{
public:
	//Adam(REAL alpha=.001, REAL beta1=.9, REAL beta2=.999, REAL eps=1.e-8);
	Adagrad(std::string name="adam");
	~Adagrad();
	Adagrad(const Adagrad&);
	//Adam& operator=(const Adam&);
	//VF2D_F update(VF2D& w);
	void update(Model* mo, VF2D& w, VF2D& m, VF2D& v, VF2D& dLdw, int& count);
};
//-------------------------------------------


#endif
