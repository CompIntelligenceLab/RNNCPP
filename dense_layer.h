#ifndef __DENSELAYER_H__
#define __DENSELAYER_H__

#include "layers.h"

class DenseLayer : public Layer
{
private:
public:
	DenseLayer();
	~DenseLayer();
	DenseLayer(DenseLayer&);
};

#endif
