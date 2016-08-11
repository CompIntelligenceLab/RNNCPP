
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include "input.h"
#include "typedefs.h"

int main() 
{
	printf("\n----- input.cpp ---------\n");
	std::string file = "test.csv";

	Input input;

	VF2D x = input.load1DSine(100, 5, 10); 


	//VF1D in = input.read1D(file);

	/*
	VF2D x = input.loadFromCSV(file);

	printf("Data read via loadFromCSV\n");
	for (int i=0; i < x.n_rows; i++) {
		for (int j=0; j < x.n_cols; j++) {
			// does not work with x[i,j]. STRANGE. 
			printf("x[%d,%d] = %f  ", i, j, x(i,j));
		}
		printf("\n");
	}
	*/
}
