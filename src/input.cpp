#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include "input.h"

using namespace std;

int Input::counter = 0;

Input::Input(std::string name /*input*/)
{
	char cname[80];
	if (strlen(cname) > 80) {
		printf("Activation::Activation : cname array too small\n");
		exit(1);
	}
	sprintf(cname, "%s%d", name.c_str(), counter);
	this->name = cname;
	printf("Layer constructor (%s)\n", this->name.c_str());
	name = "input";

	printf("Input constructor (%s)\n", name.c_str());
}

Input::~Input()
{}

Input::Input(const Input& in) : name(in.name), print_verbose(in.print_verbose)
{
	name = in.name + "c";
	printf("Input copy constructor (%s)\n", this->name.c_str());
}

const Input& Input::operator=(const Input& in)
{
	if (this != &in) {
		name = in.name + "=";
		print_verbose = in.print_verbose;
	}

	printf("Input operator= (%s)\n", this->name.c_str());
	return (*this);
}

void Input::print(std::string msg)
{
}

#if 0
VF1D Input::read1D(std::string filename)
{
	std::ifstream file (filename.c_str()); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
	std::string value;
	while ( file.good() ) {
    	getline ( file, value, ',' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
    	printf("value= %s\n", value.c_str());
    	std::cout << std::string( value, 1, value.length()-2 ); // display value removing the first and the last character from it
	}
}
#endif


VF2D Input::loadFromCSV(const std::string& filename)
{
    std::ifstream       file( filename.c_str() );
    std::vector< std::vector<std::string> >   matrix;
    std::vector< std::vector<float> >   matrix_f;
    std::vector<std::string>   row;
    std::string                line;
    std::string                cell;
	VF2D* table;

    while( file )
    {
        std::getline(file,line);
        std::stringstream lineStream(line);
        row.clear();

        while( std::getline( lineStream, cell, ',' ) )
            row.push_back( cell );

        if( !row.empty() )
            matrix.push_back( row );
    }

    for( int i=0; i<int(matrix.size()); i++ )
    {
        for( int j=0; j<int(matrix[i].size()); j++ )
            std::cout << matrix[i][j] << " ";

        std::cout << std::endl;
    }

	int nb_rows = matrix.size();
	int nb_cols = matrix[0].size();
	printf("nb_rows= %d, nb_cols= %d\n", nb_rows, nb_cols);
	table = new VF2D(nb_rows, nb_cols);
	VF2D t(nb_rows, nb_cols);

	for (int i=0; i < nb_rows; i++) {
		for (int j=0; j < nb_cols; j++) {
			const char* st = matrix[i][j].c_str();
			//printf("st= %s\n", st);
			float f = strtof(st, 0);
			//printf("f= %f\n", f);
			(*table)(i,j) = f; //strtof(st, 0);
			//t(i,j) = f; //strtof(st, 0);
			printf("input::table[%d, %d] = %f\n", i, j, (*table)(i,j));
		}
	}

	// table is not producing correct values. 

	return t;
}

VF2D Input::load1DSine(int nb_pts, int nb_periods, int nb_pts_per_period)
{
	float pi2 = 2.*acos(-1.);
	float dx = 1. / nb_pts_per_period;
	int dim = 2;
	VF2D data(nb_pts, dim);  

	for (int i=0; i < nb_pts; i++) {
		data(i,1) = cos(pi2*i*dx);
		data(i,0) = i*dx;
		printf("x, data[%d]= %f, %f\n", i, data(i,0), data(i,1));
	}

	// Returning by reference will create an error
	//printf("sizeof data= %d", sizeof(data));
	cout << "sizeof data= " << sizeof(data) << endl;
	return data;
}

