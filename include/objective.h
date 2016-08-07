#ifndef __OBJECTIVE_H__
#define __OBJECTIVE_H__

#include <string>

class Objective
{
protected:
	float learning_rate;
	std::string name;

public:
	Objective(std::string name="objective");
	~Objective();
	Objective(const Objective&);
	Objective& operator=(const Objective&);
	virtual void print(std::string name= "");
	virtual void setName(std::string name) { this->name = name; }
	virtual const std::string getName() const { return name; }
};

#endif
