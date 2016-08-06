#ifndef __LSTMLAYER_H__
#define __LSTMLAYER_H__

#include "layers.h"

class LSTMLayer : public Layer
{
private:
	str::string name;

public:
	LSTMLayer(std::string name="lstm");
	~LSTMLayer();
	LSTMLayer(LSTMLayer&);
};

#endif
