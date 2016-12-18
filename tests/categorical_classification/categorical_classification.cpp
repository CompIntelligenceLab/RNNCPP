#include <math.h>
#include <unordered_map>
#include <string>
#include <cstdlib>
#include <set>
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
		//printf("label= %d, exact_label= %d\n", label, exact_label);
		if (exact_label != label) nb_errors++;
	}

	return nb_errors;
	exit(0);
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
void categoricalClassification(Model* m) 
{
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

	m->printSummary();
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
	nb_epochs = 5000;

	VF2D* X_train; 
	VF2D* Y_train;
	getDataset(m, X_train, Y_train);
	U::print(*X_train, "X_train");

	int nb_samples = (X_train->n_rows / batch_size - 1);
	printf("nb_samples= %d\n", nb_samples);

	for (int i=0; i < 32; i++) {
		printf("X_train: %d, %f, %f,    Y_train: %%f, %f\n", i, (*X_train)(i,0), (*X_train)(i,1),  (*Y_train)(i,0), (*Y_train)(i,1));
	}

	for (int e=0; e < nb_epochs; e++) {
		printf("*** epoch %d ****\n", e);
		reset = true;

		for (int i=0; i < nb_samples; i++) {
			printf("%d ", i);
			//U::print(net_inputs, "net_inputs");
    		getNextGroupOfData(m, reset, *X_train, *Y_train, net_inputs, net_exact);
			//net_inputs.print("net_inputs");
			//net_exact.print("net_exact");
			// Need a way to exit getNext... when all characters are processed
			m->trainOneBatch(net_inputs, net_exact);
			reset = false;
		}

		if (e % 10 == 0) {
			reset = true;
			for (int i=0; i < nb_samples; i++) {
				//U::print(net_inputs, "net_inputs");
    			getNextGroupOfData(m, reset, *X_train, *Y_train, net_inputs, net_exact);
				VF2D_F pred = m->predictViaConnectionsBias(net_inputs);
		        int nb_errors = checkErrors(m, pred, net_exact);
				printf("nb_errors= %d\n", nb_errors);
				//exit(0);
			}
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

	Model* m = processArguments(argc, argv);
	categoricalClassification(m);
}
//----------------------------------------------------------------------
