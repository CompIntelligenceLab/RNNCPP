#ifndef __OUTLAYER_H__
#define __OUTLAYER_H__

#include <string>
#include "layers.h"

class OutLayer : public Layer
{
protected:
	std::string name;

public:
	OutLayer(int layer_size, std::string name="out");
	~OutLayer();
	OutLayer(OutLayer&);
};

#endif
