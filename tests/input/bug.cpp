
#include <stdio.h>
#include <armadillo>

int main() 
{
	printf("\n\n\n");
	printf("=============== BEGIN bug =======================\n");
	arma::Mat<REAL> x(2,2);

	x(0,0) = 3;
	x(0,1) = 4;
	x(1,0) = 5;
	x(1,1) = 6;

	// [] opereators can have only a single number. 
	//x[0,0] = 3;
	//x[0,1] = 4;
	//x[1,0] = 5;
	//x[1,1] = 6;

	printf("Data read via loadFromCSV\n");
	for (int i=0; i < x.n_rows; i++) {
		for (int j=0; j < x.n_cols; j++) {
			// Use of x[i,j] is incorrect!
			//printf("x[%d,%d] = %f, x(%d,%d)=%f  ", i, j, x[i,j], i, j, x(i,j));
			printf("x[%d,%d] = %f", i, j, x(i,j));
		}
		printf("\n");
	}
}
