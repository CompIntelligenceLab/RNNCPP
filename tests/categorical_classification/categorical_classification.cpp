#include <math.h>
#include <unordered_map>
#include <string>
#include <cstdlib>
#include <set>
#include "globals.h"
#include "../common.h"

/* implementation of Karpathy's char-rnn. 
   Need on-hot vectors. 

  Probably 65 or so characters + punctuation. 
  Need a translator from characters to hotvec. 

  E.g.  'a' ==> 100000....00000
  E.g.  'b' ==> 010000....00000
    ..........
	    ',' ==> 000......000001

*/

//----------------------------------------------------------------------
int checkErrors(Model* m, VF2D_F& pred, VF2D_F& exact) 
{
	int batch_size = m->getBatchSize();
	int seq_len = 1;
	int nb_errors = 0;

	for (int b=0; b < batch_size; b++) {
		int label = pred[b](0,0) > pred[b](1,0) ? 1 : 0;
		int exact_label = exact[b](0,0) > exact[b](1,0) ? 1 : 0;
		if (exact_label != label) nb_errors++;
	}

	return nb_errors;
}
//----------------------------------------------------------------------
void getNextGroupOfData(Model* m, bool reset, VF2D& X_train, VF2D& Y_train, 
	VF2D_F& net_inputs, VF2D_F& net_exact)
{
	static int base=0; 
	if (reset) base = 0;

	int batch_size = m->getBatchSize();
	int input_dim = m->getInputDim();
	int nb_classes = 2;
	// sequence length is 1 (not a recurrent network)

	// shuffle or not? Ideally, yes. 

	// does memory get released? 

	for (int b=0; b < batch_size; b++) {
		for (int i=0; i < input_dim; i++) {
			net_inputs[b](i,0) = X_train(base+b,i);
		}
		for (int i=0; i < nb_classes; i++) {
			net_exact[b](i,0)  = Y_train(base+b,i);
		}
	}

	base += batch_size;
}
//----------------------------------------------------------------------
REAL computeFullLossFunction(Model* m, int nb_samples, VF2D& X_train, VF2D& Y_train, 
	VF2D_F& net_inputs, VF2D_F& net_exact)
{
	LOSS loss;
	Objective* objective = m->getObjective();

	REAL sum = 0.;
	int batch_size = m->getBatchSize();
	int seq_len = m->getSeqLen();
	int total_nb_errors = 0;

	for (int i=0; i < nb_samples; i++) {
		bool base_reset = true; 
		getNextGroupOfData(m, base_reset, X_train, Y_train, net_inputs, net_exact);
		VF2D_F pred = m->predictViaConnectionsBias(net_inputs);
		objective->computeLoss(net_exact, pred);
		loss = objective->getLoss();
		//printf("loss.n_rows= %d\n", loss.n_rows);
		//printf("loss[0].n_rows= %d\n", loss[0].n_rows);
		//printf("loss[0].n_cols= %d\n", loss[0].n_cols);
		for (int b=0; b < batch_size; b++) {
			for (int s=0; s < seq_len; s++) {
				sum += loss[b](s);
			}
		}
		int nb_errors = checkErrors(m, pred, net_exact);
		total_nb_errors += nb_errors;
	}
	printf("x sum= %f, total_nb_errors= %d\n", sum, total_nb_errors);
	return 0.;
}
//----------------------------------------------------------------------
void getDataset(Model* m, VF2D*& X_train, VF2D*& Y_train)
{
// Create two concentric circles. Labels 0 are inside, label 1 is outside. 
	int nb_points = 1000; 
	REAL radius = 0.9;
	REAL frac   = 0.8;  // separate training and testing
	int input_size = m->getInputDim();

	// construct random pairs of points (mean 0, std 1)
	VF2D pts = arma::randu<VF2D>(nb_points, input_size); // uniform[0,1]
	VF2D pts1 = arma::sqrt(pts % pts);
	colvec radii = arma::sum(pts1, 1);
	arma::umat labels = radii < radius;  // 0 or 1, unsigned int
	//printf("labels: %d, %d\n", labels.n_rows, labels.n_cols);

	// Create 1-hot vectors from labels: 0 label --> (0,1), 1 label --> 1,0

	VF2D onehot(nb_points, 2);
	for (int i=0; i < nb_points; i++) {
		// rows can be considered as 1D vectors even though they are not
		if (labels[i] == 0) {
			onehot(i,0) = 1.;
			onehot(i,1) = 0.;
		} else {
			onehot(i,0) = 0.;
			onehot(i,1) = 1.;
		}
	}

	X_train = new VF2D(pts);
	Y_train = new VF2D(onehot); 
}
//----------------------------------------------------------------------
void categoricalClassification(Globals* g) 
{
	Model* m = new Model();
	m->setSeqLen(g->seq_len); // ignore seq_len in Globals
	m->setInputDim(g->input_dim);
	m->setBatchSize(g->batch_size);
	m->nb_epochs = g->nb_epochs;
	m->setInitializationType(g->initialization_type);
	m->setLearningRate(g->learning_rate);
	m->layer_size = g->layer_size;
	m->init_weight_rms = g->init_weight_rms;
	printf("init rms= %f\n", m->init_weight_rms);

	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;
	int nb_epochs = m->nb_epochs;
	Activation* activation = m->activations[0];
	//activation->print("");exit(0);

	//--------------------------------------

	// CONSTRUCT MODEL

	m->setObjective(new CrossEntropy()); 

	m->setInputDim(2);
	int input_dim = m->getInputDim();
	int nb_classes = 2;

	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer* d1    = new DenseLayer(layer_size, "rdense");
	Layer* d2    = new DenseLayer(layer_size, "rdense");
	Layer* d3    = new DenseLayer(nb_classes, "rdense");

	// Softmax is included in the calculation of the cross-entropy

	m->add(0, input);
	m->add(input, d1);
	m->add(d1, d2);
	m->add(d2, d3);

	input->setActivation(new Identity()); // memory leak on Identity
	d1->setActivation(new Tanh());
	d2->setActivation(new Tanh());
	d3->setActivation(new Identity()); // softmax applied as part of objective function

	m->addInputLayer(input);
	m->addOutputLayer(d3);

	//m->printSummary();
	//m->print();
	//exit(0);
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing
	BIAS& b1 = d1->getBias();
	BIAS& b2 = d2->getBias();
	BIAS& b3 = d3->getBias();
	// b1 = 0.1 generates a single scalar. Do not know why. 
	b1 = 0.1 * arma::ones<BIAS>(size(b1));
	b2 = 0.1 * arma::ones<BIAS>(size(b2));
	b3 = 0.1 * arma::ones<BIAS>(size(b3));

	// End of model
	// -----------------------------
	// Run model
	//std::vector<VF2D_F> net_inputs, net_exact;
	VF2D_F net_inputs, net_exact;
	int batch_size = m->getBatchSize();
	int seq_len = 1;
	U::createMat(net_inputs, batch_size, input_dim, seq_len);
	U::createMat(net_exact,  batch_size, input_dim, seq_len);
	U::print(net_inputs, "net_inputs");

	bool reset;
	nb_epochs = 100;
	nb_epochs = 2000;

	VF2D* X_train; 
	VF2D* Y_train;
	getDataset(m, X_train, Y_train);
	U::print(*X_train, "X_train");

	int nb_samples = (X_train->n_rows / batch_size - 1);
	printf("nb_samples= %d\n", nb_samples);

	for (int i=0; i < 16; i++) {
		printf("X_train: %d, %f, %f,    Y_train: %%f, %f\n", i, (*X_train)(i,0), (*X_train)(i,1),  (*Y_train)(i,0), (*Y_train)(i,1));
	}

	for (int e=0; e < nb_epochs; e++) {
		printf("*** epoch %d ****\n", e);
		reset = true;

		if (e % 100 == 0) {
			reset = true;
			REAL loss = computeFullLossFunction(m, nb_samples, *X_train, *Y_train, net_inputs, net_exact);
			//exit(0);
		}

		for (int i=0; i < nb_samples; i++) {
    		getNextGroupOfData(m, reset, *X_train, *Y_train, net_inputs, net_exact);
			// Need a way to exit getNext... when all characters are processed
			m->trainOneBatch(net_inputs, net_exact);
			reset = false;
		}

		printf("\n\n");
	}

	delete input;
	delete d1;
	delete d2;
	delete d3;
	delete X_train;
	delete Y_train;
}

//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -r is_recursive

	Globals* g = processArguments(argc, argv);
	categoricalClassification(g);
}
//----------------------------------------------------------------------
