#ifndef __PRINT_UTILS_H__
#define __PRINT_UTILS_H__

#include "typedefs.h"

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

	static void createMat(VF2D_F& mat, int nb_batch, int nb_rows, int nb_cols);
	static void matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec);
	static void matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec, int seq);
	static void matmul(VF2D& prod, const VF2D& mat, const VF2D_F& vec, int seq);
	static void matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec, int from, int to);
	static void zeros(VF2D_F& mat);
	static void t(VF2D_F& mat, VF2D_F& transpose);
	static void leftTriad(VF2D_F& prod, VF2D_F& a, VF2D_F& b, VF2D_F& c);
	static void leftTriad(VF2D_F& prod, VF2D_F& a, VF2D_F& b, VF2D_F& c, int from, int to);
	static void rightTriad(VF2D_F& prod, VF2D_F& a, VF2D_F& b, VF2D_F& c);
	static void rightTriad(VF2D_F& prod, VF2D_F& a, VF2D_F& b, VF2D_F& c, int from, int to);
};

#endif
