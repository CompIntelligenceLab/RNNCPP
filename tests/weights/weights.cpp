
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "print_utils.h"
#include "activations.h"
#include "optimizer.h"
#include "connection.h"
#include "layers.h"

int main() 
{
	printf("\n\n\n");
	printf("=============== BEGIN weights  =======================\n");
	Connection w1(5,6); // arguments: inputs, outputs, weight(output, input)
	Connection w2(5,6); // WEIGHT 6,5

	for (int j=0; j < 5; j++) {  // inputs, 2nd arg
	for (int i=0; i < 6; i++) {
		printf("i,j= %d, %d\n", i,j);
		w1(i,j) = (i+1) + (j+1);
		w2(i,j) = i-j;
	}}

	WEIGHT& ww = w1.getWeight();
	U::print(ww, "ww");
	ww.print("ww");

	// assume single batch
	Connection w3(w1); // + w2; (works fine)
	for (int i=0; i < 2; i++) {
	for (int j=0; j < 3; j++) {
		printf("1. w1,w3(w1)= %f, %f\n", w1(i,j), w3(i,j));
	}}


	Connection w4(w1); 
	printf("w4 = w1+w2+w1+w2\n");
	//w4 = w1 + w2 + w1 + w2; // works

	//printf("===========================\n");
	//w1.print("connection w1");
	w4 = w1 + w2 + w1 + w2;
	//printf("===========================\n");

	for (int i=0; i < 2; i++) {
	for (int j=0; j < 3; j++) {
		printf("2. w1,w2= %f, %f, w4=w1+w2+w1+w2= %f\n", w1(i,j), w2(i,j), w4(i,j));
	}}

	//w4.getWeight().print("w4");

	//exit(0);

	// I do not the way weights are initialized. Ideally, weights should be initialized
	//    as weights(outputs, inputs). But they are not. 
	Connection w10(6,3); // arguments: inputs, outputs, weight(output, input)
	for (int j=0; j < 6; j++) {
	for (int i=0; i < 3; i++) {
		w10(i,j) = i+j;
	}}

	w10.getWeight().print("w10");
	w1.getWeight().print("w1");


	// ===============================
	//Test multiplication

	// w1(5,6), w10(6,3). Therefore, 
	//w3 = w1 * w10;
	w3 = w10 * w1;

	for (int i=0; i < 2; i++) {
	for (int j=0; j < 3; j++) {
		printf("3. w1,w10= %f, %f, w3=w1*w10= %f\n", w1(i,j), w10(i,j), w3(i,j));
	}}


	//==============================
	// Multiplication of w * x  (Connection * VF2D_F)

	VF2D_F x(3);
	//VF2D_F y(3);

	for (int i=0; i < 3; i++) {
		x[i] = VF2D(2,4); // dim, seq_len
		//y[i] = VF2D(3,4); // layer[k], seq_len

		for (int p=0; p < 2; p++) {
		for (int q=0; q < 3; q++) {
			x[i](p,q) = (REAL) (p+q)*(i+1);
			printf("4. x[%d](%d,%d)= %f\n", i, p, q, x[i](p,q));
		}}
	}

	//-----------------
	Connection w11(2,3);  // layer[k], layer[k-1]

	U::print(x, "tests_weights, x");
	w11.printSummary("w11"); 
	//printf("x dims: b: %d, (%d, %d)\n", x.n_rows, x[0].n_rows, x[0].n_cols);
	//printf("w11 rows/cols: %d, %d\n", w11.getNRows(), w11.getNCols());

	for (int i=0; i < 3; i++) {
	for (int j=0; j < 2; j++) {
		w11(i,j) = (REAL) (i+2*j);
		printf("5. w11(%d,%d)= %f\n", i, j, w11(i,j));
	}}

	w11.getWeight().print("w11");
	x.print("x");
	//exit(0);


	VF2D_F y = w11 * x;  // w11(3,2) * x(3)(2,4) ==> x(3)(3,4) // w11[layer(l), layer(l-1))
	VF2D_F tst(y); tst[0].zeros(); tst[1].zeros(); tst[2].zeros();

	// Multiply by hand for testing
	for (int b=0; b < x.n_rows; b++) {
		for (int s=0; s < x[0].n_cols; s++) {
			for (int l=0; l < w11.getNRows(); l++) {
				REAL m = 0.;
				for (int i=0; i < x[0].n_rows; i++) {
					m += w11(l, i) * x[b](i, s);
				}
				tst[b](l,s) = m;
			}
		}
	}

	U::print(y, "tests_weights, y");
	//printf("y batch: %d,  dims: %d, %d\n", y.n_rows, y[0].n_rows, y[0].n_cols);

	printf("y = w11 * x\n");
	U::print(y, "tests_weights, y");

	for (int i=0; i < 3; i++) {
		for (int p=0; p < 3; p++) {
		for (int q=0; q < 4; q++) {
			printf("6. y[%d](%d,%d)= %f, exact: %f\n", i, p, q, y[i](p,q), tst[i](p,q));  
		}}
	}

	// FIX code to check calculations and return success or not when computing norms. 
	exit(0);
}
