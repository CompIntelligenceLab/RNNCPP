#ifndef __RECURRENT_LAYER_H__
#define __RECURRENT_LAYER_H__

#include "layers.h"

// Recurrent Layer have a connection to itself, which, for now, 
// is treated specially. 

class RecurrentLayer : public Layer
{
public:
	VF2D_F loop_input;
	VF2D_F loop_delta;

protected:
	Connection* recurrent_conn;

public:
   	RecurrentLayer(int layer_size=1, std::string name="layer"); // allows for default constructor
   	~RecurrentLayer();
   	RecurrentLayer(const RecurrentLayer&);
   	const RecurrentLayer& operator=(const RecurrentLayer&);
   	virtual void forwardData(Connection* conn, VF2D_F& prod, int seq);
	// there should always be data (or zero) at the input node of a temporal connection
	//virtual bool areIncomingLayerConnectionsComplete();
	virtual void processData(Connection* conn, VF2D_F& prod);
	virtual void forwardLoops();
};

//----------------------------------------------------------------------

#endif
