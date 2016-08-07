#ifndef __GMMLAYER_H__
#define __GMMLAYER_H__

#include <string>
#include "layers.h"

class GMMLayer : public Layer
{
protected:
	std::string name;

public:
	GMMLayer(int layer_size, std::string name="gmm");
	~GMMLayer();
	GMMLayer(GMMLayer&);
};

#endif
