#include <iostream>
#include "print_utils.h"

using namespace std;

//----------------------------------------------------------------------
void U::print(VF3D x, std::string msg /*""*/)
{
	cout << msg << ",  shape: (" << x.n_rows << ", " << x.n_cols << ", " << x.n_slices << ")" << endl;
}
//----------------------------------------------------------------------
void U::print(VF2D_F x, std::string msg /*""*/)
{
	cout << msg << ",  field size: " << x.n_rows << ", shape: " << x[0].n_rows << ", " << x[0].n_cols << endl;
} 
//----------------------------------------------------------------------
void U::print(VF2D_F x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);
	cout << buf << ", field size: " << x.n_rows << ", shape: (" << x[0].n_rows << ", " << x[0].n_cols << ")" << endl;
}
//----------------------------------------------------------------------
void U::print(VF2D x, std::string msg /*""*/)
{
	cout << msg << ", shape: " << x.n_rows << ", " << x.n_cols << endl;
} 
//----------------------------------------------------------------------
void U::print(VF2D x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);
	cout << buf << ", shape: (" << x.n_rows << ", " << x.n_cols << ")" << endl;
}
//----------------------------------------------------------------------
void U::print(VF1D x, std::string msg /*""*/)
{
	cout << msg << ", shape: " << x.n_rows << endl;
} 
//----------------------------------------------------------------------
void U::print(VF1D x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);
	cout << buf << ", shape: (" << x.n_rows << endl;
}
//----------------------------------------------------------------------
