#include <math.h>
#include <unordered_map>
#include <string>
#include <cstdlib>
#include <random>
#include <set>

// for discrete distributions
#include <iostream>
#include <map>

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

How to transfer weights from model M1 to model M2?
  
     M2.setWeightsAndBiases(M1);
*/

//----------------------------------------------------------------------

int discrete_sample(VF2D_F& x)
{
// code modified from code obtained on the net

    // Setup the random bits
	Softmax soft; 

	// ideally, these (rd and gen) should be called only once
    std::random_device rd;
    std::mt19937 gen(rd()); 

	VF2D_F y = soft(x);
	//y.print("softmax");

    // Setup the weights (in this case linearly weighted)
	int sz = y[0].n_rows;
    std::vector<REAL> weights(sz); // sz=65=nb_chars
    for(int i=0; i < sz; i++) {
        weights[i] = y[0](i,0);
    }

    // Create the distribution with those weights
    std::discrete_distribution<> d(weights.begin(), weights.end());

    // use the distribution and print the results.
	int ix = d(gen);
	return ix;
}
//----------------------------------------------------------------------
void sample(Model* mi, int which_char, 
		std::unordered_map<char, int>& c_int,
		std::vector<char>& int_c, VF1D_F& hot)
{
	VF2D_F x(1), y(1);
	std::vector<int> message;

	//printf("ENTER SAMPLE, which_char= %d\n", which_char);
	x(0) = hot[which_char];
	message.push_back(which_char);

	mi->resetState();

	for (int i=0; i < 50; i++) {
		x = mi->predictViaConnectionsBias(x);
		int id;
		which_char = discrete_sample(x);
		x(0) = hot[which_char];
		//printf("id= %d\n", which_char);
		//U::print(hot, "hot");
		//for (int j=0; j < 6; j++) {
			//printf("hot[%d] = %f\n", j, x(0)(j,0));
		//}
		message.push_back(which_char);
	}

	printf("message: ");
	for (int i=0; i < message.size(); i++) {
		printf("%c", int_c[message[i]]);
	}
	printf("\n\n");
}
//----------------------------------------------------------------------
// returns c_int(last character processed)
int getNextGroupOfChars(Model* m, bool& reset, std::string input_data,
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

	int which_char;

	// Check that we won't go beyond character string
	printf("input_data.size= %d\n", input_data.size());
	if (((base + seq_len*batch_size + 2)) >= input_data.size()) {
		return -1;
	}

	for (int b=0; b < batch_size; b++) {
		for (int s=0; s < seq_len; s++) {
			//printf("base+s*nb_chars, input_data: %d, %d\n", base+s*nb_chars, input_data.size());
			//printf("      nb_chars= %d\n", nb_chars);
			which_char = c_int.at(input_data[base + s]);
			//printf("which_char= %d, ", which_char);
			for (int i=0; i < nb_chars; i++) {   // one-hot vectors
				net_inputs[b](i, s)       = hot[which_char][i];
			}
			//printf("x base= %d\n", base);
			which_char = c_int.at(input_data[base + (s+1)]);
			//printf("which_char_exact= %d\n", which_char);
			for (int i=0; i < nb_chars; i++) {   // one-hot vectors
				// hot vectors not really required, if I take shortcuts to compute loss function
				net_exact[b](i, s) = hot[which_char][i];
			}
		}
	}
	//printf("enter: base= %d\n", base);
	base += seq_len * batch_size;
	//printf("exit: base= %d\n", base);
	//printf("input_data size: %d\n", input_data.size());
	return which_char;
}
//----------------------------------------------------------------------
Model* createModel(Globals* g, int batch_size, int seq_len, int input_dim, int layer_size) 
{
	printf("*********** enter createModel ***********\n");
	// Not working

	//Model* m = new Model(*m_old);
	Model* m = new Model();
	m->setSeqLen(seq_len); // ignore seq_len in Globals
	m->setInputDim(input_dim);
	m->setBatchSize(batch_size);
	m->nb_epochs = g->nb_epochs;
	m->setInitializationType(g->initialization_type);
	m->setLearningRate(g->learning_rate);
	m->layer_size = g->layer_size;
	m->init_weight_rms = g->init_weight_rms;
	//printf("init rms= %f\n", m->init_weight_rms);

	m->setObjective(new CrossEntropy()); 

	Layer* input = new InputLayer(m->getInputDim(), "input_layer");
	Layer* d1    = new DenseLayer(m->layer_size,    "rdense");
	//wght.print("update: w3");
	Layer* d2    = new DenseLayer(m->getInputDim(), "rdense");

	// Softmax is included in the calculation of the cross-entropy

	m->add(0, input);
	m->add(input, d1);
	m->add(d1, d1, true);  // recursive
	m->add(d1, d2);

	input->setActivation(new Identity());// Original
	//input->setActivation(new Tanh());
	//d1->setActivation(new Tanh());
	//d1->setActivation(new ReLU());
	d1->setActivation(new Tanh()); // as Karpathy python code
	d2->setActivation(new Identity()); // original

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	//m->printSummary();
	//printf("====   after printSummary ====\n");
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing
	BIAS& b1 = d1->getBias();
	BIAS& b2 = d2->getBias();

	#if 1
	// initialize weights deterministically (same as Karparthy for debugging)

	Connection* con = m->getConnection(input, d1);
	WEIGHT& w1 = con->getWeight();
	//con->printSummary("clist[c]"); // GE
	//U::print(w1, "w1");
	//U::print(w2, "w2");
	//printf("w1: %d,%d\n", w1.n_rows, w1.n_cols);
	//printf("w2: %d,%d\n", w2.n_rows, w2.n_cols);
	for (int i=0; i < w1.n_rows; i++) {
	for (int j=0; j < w1.n_cols; j++) {
		//printf("i,j= %d, %d\n", i, j);
		w1(i,j) = .3 / (1.+i+j);
	}}
	con->computeWeightTranspose();

	con = m->getConnection(d1, d2);
	WEIGHT& w2 = con->getWeight();
	for (int i=0; i < w2.n_rows; i++) {
	for (int j=0; j < w2.n_cols; j++) {
		w2(i,j) = .3 / (1.+i+j);
	}}
	con->computeWeightTranspose();

	WEIGHT& w3 = m->getConnection(d1, d1)->getWeight();
	//U::print(w3,"w3");
	//m->getConnection(d1, d1)->freeze();  // freeze w3
	w3.zeros();
	w3 = w3 + 0.3;
	//w3.print("w3");
	#endif


	// b1 = 0.1 generates a single scalar. Do not know why. 
	b1 = 0.0 * arma::ones<BIAS>(size(b1));
	b2 = 0.0 * arma::ones<BIAS>(size(b2));

	//m->print("MODEL PRINTOUT\n");
	//printf("====   after print ====\n");

	return m;
}
//----------------------------------------------------------------------

void charRNN(Globals* g) 
{
	int layer_size = g->layer_size;
	int is_recurrent = g->is_recurrent;
	int nb_epochs = g->nb_epochs;

// Read Input Data
	//string file_name = "input.txt";
	string file_name = "fox.txt";

	ifstream fd;
	fd.open(file_name);
	stringstream strStream;
	strStream << fd.rdbuf();
	std::string input_data = strStream.str();
	//cout << input_data << endl;

	// How many distinct characters? 
	std::set<char> char_set;

	// Collect unique characters
	// skip last character which is "\0" or some other non-printable character
	for (int i=0; i < input_data.size()-1; i++) {
		//printf("char: %c\n", input_data[i]);
		char_set.insert(input_data[i]);
	}
	printf("set size: %d\n", char_set.size());

	int nb_chars = char_set.size();
	std::set<char>::iterator si;
	std::unordered_map<char, int> c_int;
	std::vector<char> int_c; //, char> int_c;
	int_c.resize(nb_chars);

	// Put the characters in a particular order: brown
	std::string brown= "brown";

	int i=0; 
	#if 0
	// Original
	for (si=char_set.begin(); si != char_set.end(); si++) {
		printf("char= %c\n", *si);
		//printf("char_set = %c\n", *si);
		c_int[*si] = i;
		int_c[i] = *si;
		i++;
	}
	#endif

	// Used for testing (GE)
	// I want the character order to be the same in Karpathy's code and this one. 
	// "brown" has 5 different characters. 
	for (int i=0; i < brown.size(); i++) {
		printf("brown[%d]= %c\n", i, brown[i]);
		c_int[brown[i]] = i;
		int_c[i] = brown[i];
	}

	i=0; 
	#if 0
	for (si=char_set.begin(); si != char_set.end(); si++) {
		c_int.at(*si) = i;
		int_c[i] = *si;
		i++; 
	}
	#endif

	// Return to generic approach once code matches Karpathy's
	for (int i=0; i < brown.size(); i++) {
		char c = int_c[i];
		printf("char[%d] = %c\n", i, c);
	}
	//   hot[3] = 0010000...0000;

	VF1D_F hot(nb_chars);  // VI: 

	for (int i=0; i < nb_chars; i++) {
		hot[i] = VF1D(nb_chars);
		hot[i].zeros();
		hot[i][i] = 1.;
	}
	//--------------------------------------

	// CONSTRUCT MODEL
	//m->setInputDim(nb_chars);
	int input_dim = nb_chars;
	Model* m_train = createModel(g, g->batch_size, g->seq_len, input_dim, g->layer_size);
	// What could be wrong with this line? 
	Model* m_pred = createModel(g,             1,          1, input_dim, g->layer_size);
	Model* m = m_train;

	m->setStateful(true);


	// End of model
	// -----------------------------
	// Run model

	bool reset;
	int nb_samples;
	int seq_len = m->getSeqLen();
	int batch_size = m->getBatchSize();
	nb_samples = input_data.size() / (batch_size * seq_len);
	printf("nb_samples= %d\n", nb_samples); //exit(0);
	printf("batch_size= %d, seq_len= %d, nb_chars= %d\n", batch_size, seq_len, nb_chars);
	//nb_samples = 200;
	//nb_epochs = 100;
	Objective* objective = m->getObjective();

	VF2D_F net_inputs, net_exact;
	//int input_dim = m_train->getInputDim();
	U::createMat(net_inputs, batch_size, input_dim, seq_len);
	U::createMat(net_exact, batch_size, input_dim, seq_len);

	int count = 0;
	int which_char;

	for (int e=0; e < nb_epochs; e++) {
		which_char = c_int.at(input_data[0]);
		printf("*** epoch %d ****\n", e);
		m->resetState();
		reset = true;

		for (int i=0; i < nb_samples; i++) {
		printf("\n=====================================\n");
		printf("========== NEW SAMPLE === iter %d ======\n", i);
			#if 0
			// ADD BACK WHEN CODE WORKS
			if (count % 10 == 0) {
				printf("TRAIN, nb_epochs: %d, iter: %d, ", e, count);
				m_pred->setWeightsAndBiases(m);
				sample(m_pred, which_char, c_int, int_c, hot);
			}
			#endif

			printf("Model m state: %d\n", m->getStateful());
    		which_char = getNextGroupOfChars(m, reset, input_data, net_inputs, net_exact, c_int, int_c, hot);
			if (which_char < 0) break;
			// Need a way to exit getNext... when all characters are processed
			reset = false;

			if (count == 2) exit(0); // TEMPORARY
		#if 0
		printf("------------\n");
		for (int s=0; s < seq_len; s++) {
		for (int j=0; j < 5; j++) {
			printf("s=%d, net_inputs[%d]= %f, net_exact[%d]= %f\n", s, j, net_inputs(0)(j,s), j, net_exact(0)(j,s));
		}}
		#endif
			m->trainOneBatch(net_inputs, net_exact);
			count++;

		}
		printf("\n\n");
	}

	//delete input;
	//delete d1;
	//delete d2;
	//delete m_train;
	//delete m_pred;
}

//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -r is_recursive

	Globals* g = processArguments(argc, argv);
	charRNN(g);
}
//----------------------------------------------------------------------
