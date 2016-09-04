#ifndef __DENSELAYER_H__
#define __DENSELAYER_H__

#include <string>
#include "layers.h"

class DenseLayer : public Layer
{
protected:
	std::string name;

public:
	DenseLayer(int layer_size=1, std::string name="dense");
	~DenseLayer();
	DenseLayer(DenseLayer&);
	void noop() {;} 
};

#endif
