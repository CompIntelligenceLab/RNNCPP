
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "activations.h"
#include "optimizer.h"
#include "connection.h"
#include "layers.h"

int main() 
{
	Connection w1(5,6);
	Connection w2(5,6);

	for (int i=0; i < 5; i++) {
	for (int j=0; j < 6; j++) {
		printf("i,j= %d, %d\n", i,j);
		w1(i,j) = i+j;
		w2(i,j) = i-j;
	}}

	// assume single batch
	Connection w3(w1);; // + w2; (works fine)
	for (int i=0; i < 2; i++) {
	for (int j=0; j < 3; j++) {
		printf("w1,w3(w1)= %f, %f\n", w1(i,j), w3(i,j));
	}}


	printf("w3 = w1+w2+w1+w2\n");
	w3 = w1 + w2 + w1 + w2; // works
	exit(0);

	for (int i=0; i < 2; i++) {
	for (int j=0; j < 3; j++) {
		printf("w1,w2= %f, %f, w3=w1+w2= %f\n", w1(i,j), w2(i,j), w3(i,j));
	}}

	Connection w10(6,5);
	for (int j=0; j < 5; j++) {
	for (int i=0; i < 6; i++) {
		w10(i,j) = i+j;
	}}


	// ===============================
	//Test multiplication

	w3 = w1 * w10;

	for (int i=0; i < 2; i++) {
	for (int j=0; j < 3; j++) {
		printf("w3=w1*w10= %f\n", w3(i,j));
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
			x[i](p,q) = (float) (p+q)*(i+1);
			printf("x[%d](%d,%d)= %f\n", i, p, q, x[i](p,q));
		}}
	}

	//-----------------
	Connection w11(3,2);  // layer[k], layer[k-1]

	printf("x dims: b: %d, (%d, %d)\n", x.n_rows, x[0].n_rows, x[0].n_cols);
	printf("w11 rows/cols: %d, %d\n", w11.getNRows(), w11.getNCols());

	for (int i=0; i < 3; i++) {
	for (int j=0; j < 2; j++) {
		w11(i,j) = (float) (i+2*j);
		printf("w11(%d,%d)= %f\n", i, j, w11(i,j));
	}}


	VF2D_F y = w11 * x;  // w11(3,2) * x(3)(2,4) ==> x(3)(3,4) // w11[layer(l), layer(l-1))
	VF2D_F tst(y); tst[0].zeros(); tst[1].zeros(); tst[2].zeros();

	// Multiply by hand for testing
	for (int b=0; b < x.n_rows; b++) {
		for (int s=0; s < x[0].n_cols; s++) {
			for (int l=0; l < w11.getNRows(); l++) {
				float m = 0.;
				for (int i=0; i < x[0].n_rows; i++) {
					m += w11(l, i) * x[b](i, s);
				}
				tst[b](l,s) = m;
			}
		}
	}

	printf("y batch: %d,  dims: %d, %d\n", y.n_rows, y[0].n_rows, y[0].n_cols);

	printf("y = w11 * x\n");
	printf("y.size() = %d\n", y.size()); // should be (3, ...)
	printf("print y dim: %d, %d\n", y[0].n_rows, y[0].n_cols);

	for (int i=0; i < 3; i++) {
		for (int p=0; p < 3; p++) {
		for (int q=0; q < 4; q++) {
			printf("y[%d](%d,%d)= %f, exact: %f\n", i, p, q, y[i](p,q), tst[i](p,q));  // ==> Index out of bounds
		}}
	}
}
