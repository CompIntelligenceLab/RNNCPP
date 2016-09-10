#include "../common.h"

void testRecurrentModelBias1(int nb_batch=1)
{
	printf("\n\n\n");
	printf("=============== BEGIN test_recurrent_model_bias1  =======================\n");
/***
	Simplest possible network: two nodes with the identity activation. 
	seq_len = 2
	nb_batch = 1
	This allows testing via simple matrix-multiplication

                 w1
	    input ---------> rdense --> loss    (loss is attached to the output layer)
***/
	int input_dim = 1; // predict works with input_dim > 1
	Model* m  = new Model(); // argument is input_dim of model
	m->setSeqLen(2); // runs (but who knows whether correct) with seq_len > 1

	// I am not sure that batchSize and nb_batch are the same thing
	m->setBatchSize(nb_batch);
	assert(m->getBatchSize() == nb_batch);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(1, "input_layer");
	Layer* dense = new RecurrentLayer(1, "rdense");
	Layer* out   = new OutLayer(1, "out");  // Dimension of out_layer must be 1.
	                                       // Automate this at a later time

	m->add(0,     input);
	m->add(input, dense);
	//printf("m->layers size: %d\n", m->getLayers().size()); exit(0);
	//m->add(dense, out); 
	//m->add(dense,  out); // No weights and no recursion. It is only there to connect with the loss function. 
	                     // There is a connection from dense to out. 
						 // Waste of memory if dimensionality is high (even if using identity matrix). 

	dense->setActivation(new Identity());
	input->setActivation(new Identity());
	//out->setActivation(new Identity());

	m->addInputLayer(input);
	m->addOutputLayer(dense);
	//runModelRecurrent(m);
}
//----------------------------------------------------------------------
int main()
{
	testRecurrentModelBias1(1);
}
