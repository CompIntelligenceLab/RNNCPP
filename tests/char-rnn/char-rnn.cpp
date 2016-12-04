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
void getNextGroupOfChars(Model* m, bool reset, std::string input_data,
        std::vector<VF2D_F>& net_inputs, 
		std::vector<VF2D_F>& net_exact,
		std::unordered_map<char, int>& c_int,
		std::vector<char>& int_c,
		arma::field<VI>& hot)
{
	static int base; // keep value across invocations
	if (reset) base = 0;

	int nb_batch = m->getBatchSize();
	int input_dim = m->getInputDim();
	int seq_len = m->getSeqLen();
	printf("input_dim= %d (=65?)\n", input_dim);

	VF2D_F vf2d; 
	VF2D_F vf2d_exact;

	int nb_chars = input_dim;

	U::createMat(vf2d, nb_batch, input_dim, seq_len);
	U::createMat(vf2d_exact, nb_batch, input_dim, seq_len);

	// Assume nb_batch = 1
	if (nb_batch != 1) { printf("nb_batch should be 1\n"); exit(1); }

	printf("input_data.size= %d\n", input_data.size());

			for (int s=0; s < seq_len; s++) {
				for (int i=0; i < nb_chars; i++) {   // one-hot vectors
					int ii = c_int.at(input_data[base + s*nb_chars]);
					vf2d[0](i, s)       = hot[ii][i];
					ii = c_int.at(input_data[base + s*nb_chars+1]);
					vf2d_exact[0](i, s) = hot[ii][i];
				}
			}
		//net_inputs.push_back(vf2d);
		//net_exact.push_back(vf2d_exact);
	base += seq_len * nb_chars;
	printf("base= %d\n", base);
	net_inputs.push_back(vf2d);
	net_exact.push_back(vf2d_exact);
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

	arma::field<VI> hot(nb_chars);  // VI: 

	i=0; 
	for (si=char_set.begin(); si != char_set.end(); si++) {
		c_int.at(*si) = i;
		int_c[i] = *si;
		i++; 
	}
	//   hot[3] = 0010000...0000;

	for (int i=0; i < nb_chars; i++) {
		hot[i] = VI(nb_chars);
		hot[i].zeros();
		hot[i][i] = 1;
	}
	for (int i=0; i < nb_chars; i++) {
		printf("%d\n", hot[13][i]);
	}
	//--------------------------------------

	// CONSTRUCT MODEL

	m->setObjective(new CrossEntropy()); 

	m->setInputDim(nb_chars);
	int input_dim = nb_chars;
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer* d1    = new DenseLayer(layer_size, "rdense");
	Layer* d2    = new DenseLayer(nb_chars, "rdense");

	m->add(0, input);
	m->add(input, d1);
	m->add(d1, d1, true);
	m->add(d1, d2);

	input->setActivation(new Identity());
	d1->setActivation(new Tanh());
	d2->setActivation(new Softmax());

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	m->printSummary();
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing

	// End of model
	// ----------
	// Run model
	std::vector<VF2D_F> net_inputs, net_exact;
	//int nb_samples = getCharData(m, net_inputs, net_exact, input_data, c_int, int_c, hot);
	//printf("nb_samples= %d\n", nb_samples);

	bool reset;
	int nb_samples = 2;

	for (int e=0; e < nb_epochs; e++) {
		m->resetState();
		reset = true;

		for (int i=0; i < nb_samples-1; i++) {
			printf("before train\n");
    		getNextGroupOfChars(m, reset, input_data, net_inputs, net_exact, c_int, int_c, hot);
			reset = false;
			m->trainOneBatch(net_inputs[i], net_exact[i]);
			exit(0); // CHECK backprop
		}
	}
}

//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -r is_recursive

	Model* m = processArguments(argc, argv);
	charRNN(m);
}
//----------------------------------------------------------------------
