#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>
#include <unordered_map>

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
	int nb_chars = 65;
	arma::field<VI> hot(nb_chars);  // VI: 
	std::unordered_map<char, int> c_int;
	std::unordered_map<int, char> int_c;
	//   hot[3] = 0010000...0000;

	for (int i=0; i < nb_chars; i++) {
		hot[i] = VI(65);
		hot[i].zeros();
		hot[i][i] = 1;
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
