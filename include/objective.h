#ifndef __OBJECTIVE_H__
#define __OBJECTIVE_H__

#include <string>

class Objective //
{
protected:
	float learning_rate;
	std::string name;

public:
	Objective(std::string name);
	~Objective();
	Objective(Objective&);
	virtual void print();
};

#endif
