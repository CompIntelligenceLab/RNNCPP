#ifndef __GRADIENTS_H__
#define __GRADIENTS_H__

#include <string>

class Gradients
{
protected:
	std::string name;

public:
	Gradients();
	~Gradients();
	Gradients(Gradients&);
	virtual void print(std::string name= "");
};
//----------------------------------------------------------------------
typedef std::vector<Gradients> GRADIENTS;

#endif
