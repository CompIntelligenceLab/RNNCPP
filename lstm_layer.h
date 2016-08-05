#ifndef __LSTMLAYER_H__
#define __LSTMLAYER_H__

#include "layers.h"

class LSTMLayer : public Layer
{
private:
public:
	LSTMLayer();
	~LSTMLayer();
	LSTMLayer(LSTMLayer&);
};

#endif
