#include <stdio.h>
#include <stdlib.h>

// Must make sure that I compile without C++11 

#define N 100000000
class Array
{
public:
	int n;
	int* f;

public:
	Array() {
	    n = N;
		f = new int[N];
		for (int i=0; i < N; i++) {
			f[i] = i;
		}
		printf("constructor\n");
	}

	~Array() {
		printf("destructor\n");
		delete [] f;
	}

	Array& operator+(const Array& a) const {
		Array* tmp =new Array();
		for (int i=0; i < N; i++) {
			tmp->f[i] = a.f[i] + this->f[i];
		}
		return *tmp;
	}
};

int main() 
{	
	Array* a1 = new Array();
	Array* a2 = new Array();

	
	for (int i=0; i < 100; i++) {
		printf("i= %d\n", i);
		Array a6 = *a1 + *a2 + *a1 + *a2;  // 3 constructors + 1 destructor
		//Array& a6 = *a1 + *a2 + *a1 + *a2; // 3 consturctors, NO destructors

		// defining Array operator+(...) 
		//Array a6 = *a1 + *a2 + *a1 + *a2;  // 3 constructors + 3 destructor
	}
	exit(0);
}
