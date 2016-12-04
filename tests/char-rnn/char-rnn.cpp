#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>
#include <unordered_map>
#include <set>

/* implementation of Karpathy's char-rnn. 
   Need on-hot vectors. 

  Probably 65 or so characters + punctuation. 
  Need a translator from characters to hotvec. 

  E.g.  'a' ==> 100000....00000
  E.g.  'b' ==> 010000....00000
    ..........
	    ',' ==> 000......000001

*/

void charRNN(Model* m) 
{
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
	arma::field<VI> hot(nb_chars);  // VI: 
	std::unordered_map<char, int> c_int;
	std::unordered_map<int, char> int_c;
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

	m->setObjective(new MeanSquareError()); 
	m->getObjective()->setErrorType(m->obj_err_type);

	int layer_size = 100;
	int input_dim = 1;
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer* d1    = new DenseLayer(layer_size, "rdense");
	Layer* d2    = new DenseLayer(nb_chars, "rdense");

	m->add(0, input);
	m->add(input, d1);
	m->add(d1, d1, true);
	m->add(d1, d2, true);

	input->setActivation(new Identity());
	d1->setActivation(new Tanh());
	d2->setActivation(new Softmax());

	m->addInputLayer(input);
	m->addOutputLayer(d2);

	m->printSummary();
	m->connectionOrderClean(); // no print statements

	m->initializeWeights(); // be initialized after freezing

	// End of model
}

//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -r is_recursive

	Model* m = processArguments(argc, argv);
	charRNN(m);
}
//----------------------------------------------------------------------
