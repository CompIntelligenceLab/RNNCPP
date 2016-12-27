#ifndef __PRINT_UTILS_H__
#define __PRINT_UTILS_H__

#include "typedefs.h"

class Model;

class U
{
public:
	static void print(VF3D x, std::string msg="");
	static void print(VF2D_F x, std::string msg="");
	static void print(VF2D_F x, int val1, std::string msg);
	static void print(VF2D x, std::string msg="");
	static void print(VF2D x, int val1, std::string msg);
	static void print(VF1D_F x, std::string msg="");
	static void print(VF1D_F x, int val1, std::string msg);
	static void print(VF1D x, int val1, std::string msg);
	static void print(VF1D x, std::string msg="");
	static void print(LOSS x, std::string msg="");

	static void createMat(VF2D_F& mat, int nb_batch, int nb_rows, int nb_cols);
    static void createMat(VF1D_F& mat, int nb_batch, int nb_rows);
	static void matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec);
	static void matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec, int seq);
	static void matmul(VF2D& prod, const VF2D& mat, const VF2D_F& vec, int seq);
	static void matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec, int from, int to);
	static void zeros(VF2D_F& mat);
	static void t(VF2D_F& mat, VF2D_F& transpose);
	static void leftTriad(VF2D_F& prod, VF2D_F& a, VF2D_F& b, VF2D_F& c);
	static void leftTriad(VF2D_F& prod, VF2D_F& a, VF2D_F& b, VF2D_F& c, int from, int to);
	static void rightTriad(VF2D_F& prod, const VF2D& a, const VF2D_F& b, const VF2D_F& c);
	static void rightTriad(VF2D_F& prod, const VF2D& a, const VF2D_F& b, const VF2D_F& c, int from, int to);

	static void printRecurrentLayerLoopInputs(Model *m);
	static void printInputs(Model *m);
	static void printLayerInputs(Model *m);
	static void printOutputs(Model *m);
	static void printWeights(Model* m);
	static void printWeightDeltas(Model* m);
	static void printLayerBiases(Model *m);
	static void printBiases(Model *m);
	static void printBiasDeltas(Model *m);
	static void printPreviousState(Model* m);
};

#endif
