#ifndef __RECURRENT_LAYER_H__
#define __RECURRENT_LAYER_H__

#include "layers.h"

// Recurrent Layer have a connection to itself, which, for now, 
// is treated specially. 

class RecurrentLayer : public Layer
{
public:

protected:

public:
   	RecurrentLayer(int layer_size=1, std::string name="layer"); // allows for default constructor
   	~RecurrentLayer();
   	RecurrentLayer(const RecurrentLayer&);
   	const RecurrentLayer& operator=(const RecurrentLayer&);
	// there should always be data (or zero) at the input node of a temporal connection
	//virtual bool areIncomingLayerConnectionsComplete();
	#if 0
   	virtual void forwardData(Connection* conn, VF2D_F& prod, int seq);
	virtual void processData(Connection* conn, VF2D_F& prod);
	virtual void forwardLoops();
	virtual void forwardLoops(int seq_index);
	void forwardLoops(int t1, int t2);
	#endif
	virtual void initVars(int nb_batch);
	virtual void noop() {;}
};

//----------------------------------------------------------------------

#endif
