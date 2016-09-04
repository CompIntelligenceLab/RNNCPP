#ifndef __INPUTLAYER_H__
#define __INPUTLAYER_H__

#include <string>
#include "layers.h"

class InputLayer : public Layer
{
protected:
	std::string name;

public:
	// input_dim: effectively the layer size. There is no layer feeding information 
	// to an input layer, although it is theoretically possible via recursion. 
	InputLayer(int input_dim=1, std::string name="input");
	~InputLayer();
	InputLayer(InputLayer&);
	void noop() {;}
};

#endif
