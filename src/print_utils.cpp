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
	int n_rows = (int) x[0].n_rows;
	int n_cols = (int) x[0].n_cols;

	string same_size = "same size";

	// check that all matrics are the same size
	if (x.n_rows > 1) {
		for (int i=1; i < x.n_rows; i++) {
			if (x[i].n_rows != n_rows || x[i].n_cols != n_cols)  {
				same_size = "different sizes";
				break;
			}
		}
	}

	cout << msg << ",  field size: " << x.n_rows << ", shape: (" << x[0].n_rows << ", " << x[0].n_cols << ")" 
	     << ", " << same_size << endl;
} 
//----------------------------------------------------------------------
void U::print(VF2D_F x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);

	int n_rows = (int) x[0].n_rows;
	int n_cols = (int) x[0].n_cols;

	string same_size = "same size";

	// check that all matrics are the same size
	if (x.n_rows > 1) {
		for (int i=1; i < x.n_rows; i++) {
			if (x[i].n_rows != n_rows || x[i].n_cols != n_cols)  {
				same_size = "different sizes";
				break;
			}
		}
	}

	cout << buf << ", field size: " << x.n_rows << ", shape: (" << x[0].n_rows << ", " << x[0].n_cols << ")" 
	     << ", " << same_size << endl;
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
	cout << buf << ", shape: (" << x.n_rows << ")" << endl;
}
//----------------------------------------------------------------------
void U::print(VF1D_F x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);

	int n_rows = (int) x[0].n_rows;

	string same_size = "same size";

	// check that all matrics are the same size
	if (x.n_rows > 1) {
		for (int i=1; i < x.n_rows; i++) {
			if (x[i].n_rows != n_rows) {
				same_size = "different sizes";
				break;
			}
		}
	}

	cout << buf << ", field size: " << x.n_rows << ", shape: (" << x[0].n_rows <<  ")" 
	     << same_size << endl;
}
//----------------------------------------------------------------------
void U::print(VF1D_F x, std::string msg /*""*/)
{

	int n_rows = (int) x[0].n_rows;

	string same_size = "same size";

	// check that all matrics are the same size
	if (x.n_rows > 1) {
		for (int i=1; i < x.n_rows; i++) {
			if (x[i].n_rows != n_rows) {
				same_size = "different sizes";
				break;
			}
		}
	}

	cout << msg << ", field size: " << x.n_rows << ", shape: (" << x[0].n_rows <<  ")" 
	     << same_size << endl;
}
