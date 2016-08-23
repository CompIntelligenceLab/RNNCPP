#ifndef __RECURRENT_LAYER_H__
#define __RECURRENT_LAYER_H__

#include "layers.h"

// Recurrent Layer have a connection to itself, which, for now, 
// is treated specially. 

class RecurrentLayer
{
public:

protected:

public:
   RecurrentLayer(int layer_size=1, std::string name="layer"); // allows for default constructor
   ~RecurrentLayer();
   RecurrentLayer(const RecurrentLayer&);
   const RecurrentLayer& operator=(const RecurrentLayer&);
};

//----------------------------------------------------------------------

#endif
