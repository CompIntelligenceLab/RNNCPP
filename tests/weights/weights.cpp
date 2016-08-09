
#include <stdio.h>
#include <math.h>
#include "activations.h"
#include "optimizer.h"
#include "weights.h"
#include "layers.h"

int main() 
{
	Weights w1(5,6);
	Weights w2(5,6);

	for (int i=0; i < 5; i++) {
	for (int j=0; j < 6; j++) {
		printf("i,j= %d, %d\n", i,j);
		w1(i,j) = i+j;
		w2(i,j) = i-j;
	}}

	// assume single batch
	Weights w3(w1);; // + w2; (works fine)
	for (int i=0; i < 2; i++) {
	for (int j=0; j < 3; j++) {
		printf("w1,w3(w1)= %f, %f\n", w1(i,j), w3(i,j));
	}}


	w3 = w1 + w2; // works

	for (int i=0; i < 2; i++) {
	for (int j=0; j < 3; j++) {
		printf("w1,w2= %f, %f, w3=w1+w2= %f\n", w1(i,j), w2(i,j), w3(i,j));
	}}

	Weights w10(6,5);
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

}
