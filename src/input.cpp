#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "input.h"

Input::Input()
{
	printf("Input constructor\n");
}

Input::~Input()
{}

Input::Input(const Input& in)
{
	printf("Input copy constructor\n");
}

const Input& Input::operator=(const Input& in)
{}

void Input::print(std::string msg)
{
}

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
			t(i,j) = f; //strtof(st, 0);
			printf("input::table[%d, %d] = %f\n", i, j, (*table)(i,j));
		}
	}

	// table is not producing correct values. 

	return t;
}
