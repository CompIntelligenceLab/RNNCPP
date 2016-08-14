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
	static void print(VF1D x, std::string msg="");
	static void print(VF1D x, int val1, std::string msg);
};

#endif
