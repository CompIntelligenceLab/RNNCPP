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
void getNextGroupOfChars(Model* m, bool& reset, std::string input_data,
        VF2D_F& net_inputs, VF2D_F& net_exact,
		std::unordered_map<char, int>& c_int,
		std::vector<char>& int_c,
		VF1D_F& hot)
{
	//printf("reset=%d\n", reset);
	static int base; // keep value across invocations
	if (reset) base = 0;

	int batch_size = m->getBatchSize();
	int input_dim = m->getInputDim();
	int seq_len = m->getSeqLen();

	VF2D_F vf2d; 
	VF2D_F vf2d_exact;

	int nb_chars = input_dim;

	// Assume batch_size = 1
	//if (batch_size != 1) { printf("batch_size should be 1! batch_size=1 is untested! \n"); }


	for (int b=0; b < batch_size; b++) {
		for (int s=0; s < seq_len; s++) {
			//printf("base+s*nb_chars, input_data: %d, %d\n", base+s*nb_chars, input_data.size());
			//printf("      nb_chars= %d\n", nb_chars);
			int which_char = c_int.at(input_data[base + s*nb_chars]);
			for (int i=0; i < nb_chars; i++) {   // one-hot vectors
				net_inputs[b](i, s)       = hot[which_char][i];
			}
			which_char = c_int.at(input_data[base + (s+1)*nb_chars]);
			for (int i=0; i < nb_chars; i++) {   // one-hot vectors
				// hot vectors not really required, if I take shortcuts to compute loss function
				net_exact[b](i, s) = hot[which_char][i];
			}
		}
	}
	//printf("enter: base= %d\n", base);
	base += seq_len * nb_chars * batch_size;
	//printf("exit: base= %d\n", base);
	//printf("input_data size: %d\n", input_data.size());
	return;
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
		//getNextGroupOfData(m, base_reset, X_train, Y_train, net_inputs, net_exact);
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
		//int nb_errors = checkErrors(m, pred, net_exact);
		//total_nb_errors += nb_errors;
	}
	//printf("x sum= %f, total_nb_errors= %d\n", sum, total_nb_errors);
	return 0.;
}
//----------------------------------------------------------------------

void charRNN(Model* m) 
{
	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;
	int nb_epochs = m->nb_epochs;

// Read Input Data
	//FILE* fd = fopen("input.txt", "r");
	ifstream fd;
	fd.open("input.txt");
	stringstream strStream;
	strStream << fd.rdbuf();
	std::string input_data = strStream.str();
	//cout << input_data << endl;

	// How many distinct characters? 
	std::set<char> char_set;

	// Collect unique characters
	for (int i=0; i < input_data.size(); i++) {
		char_set.insert(input_data[i]);
	}
	printf("set size: %d\n", char_set.size());

	int nb_chars = char_set.size();
	std::set<char>::iterator si;
	std::unordered_map<char, int> c_int;
	std::vector<char> int_c; //, char> int_c;
	int_c.resize(nb_chars);

	int i=0; 
	for (si=char_set.begin(); si != char_set.end(); si++) {
		//printf("char_set = %c\n", *si);
		c_int[*si] = i;
		int_c[i] = *si;
		i++;
	}

	i=0; 
	for (si=char_set.begin(); si != char_set.end(); si++) {
		c_int.at(*si) = i;
		int_c[i] = *si;
		i++; 
	}
	//   hot[3] = 0010000...0000;

	VF1D_F hot(nb_chars);  // VI: 

	for (int i=0; i < nb_chars; i++) {
		hot[i] = VF1D(nb_chars);
		hot[i].zeros();
		hot[i][i] = 1.;
	}
	//for (int i=0; i < nb_chars; i++) {
		//printf("%d\n", hot[13][i]);
	//}
	//--------------------------------------

	// CONSTRUCT MODEL

	m->setObjective(new CrossEntropy()); 

	m->setInputDim(nb_chars);
	int input_dim = nb_chars;
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer* d1    = new DenseLayer(layer_size, "rdense");
	Layer* d2    = new DenseLayer(input_dim, "rdense");


	// Softmax is included in the calculation of the cross-entropy

	m->add(0, input);
	m->add(input, d1);
	m->add(d1, d1, true);
	m->add(d1, d2);

	input->setActivation(new Identity());// Original
	//input->setActivation(new Tanh());
	//d1->setActivation(new Tanh());
	d1->setActivation(new ReLU());
	//d1->setActivation(new Tanh());
	d2->setActivation(new Identity()); // original

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	m->printSummary();
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing
	BIAS& b1 = d1->getBias();
	BIAS& b2 = d2->getBias();
	// b1 = 0.1 generates a single scalar. Do not know why. 
	b1 = 0.1 * arma::ones<BIAS>(size(b1));
	b2 = 0.1 * arma::ones<BIAS>(size(b2));

	m->setStateful(true);


	// End of model
	// -----------------------------
	// Run model

	bool reset;
	int nb_samples;
	int seq_len = m->getSeqLen();
	int batch_size = m->getBatchSize();
	nb_samples = input_data.size() / (batch_size * seq_len * nb_chars);
	printf("nb_samples= %d\n", nb_samples); //exit(0);
	printf("batch_size= %d, seq_len= %d, nb_chars= %d\n", batch_size, seq_len, nb_chars);
	//nb_samples = 200;
	//nb_epochs = 100;
	Objective* objective = m->getObjective();

	VF2D_F net_inputs, net_exact;
	U::createMat(net_inputs, batch_size, input_dim, seq_len);
	U::createMat(net_exact, batch_size, input_dim, seq_len);

	for (int e=0; e < nb_epochs; e++) {
		printf("*** epoch %d ****\n", e);
		m->resetState();
		reset = true;

		for (int i=0; i < nb_samples; i++) {
			//printf("sample %d ", i+1);
    		getNextGroupOfChars(m, reset, input_data, net_inputs, net_exact, c_int, int_c, hot);
			// Need a way to exit getNext... when all characters are processed
			reset = false;
			m->trainOneBatch(net_inputs, net_exact);
		}
		printf("\n\n");
		//exit(0);
	}

	delete input;
	delete d1;
	delete d2;
}

//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -r is_recursive

	Model* m = processArguments(argc, argv);
	charRNN(m);
}
//----------------------------------------------------------------------
