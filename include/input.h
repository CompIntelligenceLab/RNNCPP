#ifndef __INPUT_H__
#define __INPUT_H__

#include <vector>
#include <string>
#include "typedefs.h"

// Class to help input data into the code. 
// Read input from one of several file formats. The user can write Python 
// code and write out the data in any one of several formats. 
// The Input class will convert the data to VF2D format (nb_elements, dim). 
// "dim" : dimensionality of the data (how many streams are input into the network 
//

class Input
{
private:
	static int counter;
	std::string name;
	bool print_verbose;

public:
  	Input(std::string name="input");
  	~Input();
  	Input(const Input&); 
  	const Input& operator=(const Input&); 
  	void print(std::string msg=std::string());

  	//VF1D read1D(std::string filename);
	VF2D loadFromCSV(const std::string& filename);
	VF2D load1DSine(int nb_pts, int period, int nb_pts_per_period);
};

#endif
