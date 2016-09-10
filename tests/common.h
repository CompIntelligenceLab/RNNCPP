#include <stdio.h>
#include <math.h>
#include <string>
#include <assert.h>
#include <iostream>
//#include <fstream>
#include "model.h"
#include "activations.h"
#include "connection.h"
#include "optimizer.h"
#include "objective.h"
#include "layers.h"
#include "recurrent_layer.h"
#include "dense_layer.h"
#include "out_layer.h"
#include "lstm_layer.h"
#include "gmm_layer.h"
#include "input_layer.h"
#include "print_utils.h"


using namespace arma;
using namespace std;

WEIGHT dLdw(1,1);
BIAS dLdb(1);

#include "common.cpp"
