
#include <stdio.h>
#include <math.h>
#include "activations.h"
#include "optimizer.h"
#include "weights.h"

int main() {
	Optimizer* opt = new Optimizer("optGE");
	Optimizer opt1(*opt);
	Optimizer opt2 = *opt;

	opt->print("\n--> print opt\n");
	opt1.print("\n--> print opt1\n");
	opt2.print("\n--> print opt2\n");

	Weights* weight = new Weights(1,1,"wghtGE");
	Weights w1(*weight);
	Weights w2 = *weight;

	weight->print("\n --> print wght\n");
	w1.print("\n --> print w1\n");
	w2.print("\n --> print w2\n");
}
