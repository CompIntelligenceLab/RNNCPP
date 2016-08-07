#ifndef __DENSELAYER_H__
#define __DENSELAYER_H__

#include <string>
#include "layers.h"

class DenseLayer : public Layer
{
protected:
	std::string name;

public:
	DenseLayer(int layer_size, std::string name="dense");
	~DenseLayer();
	DenseLayer(DenseLayer&);
};

#endif
