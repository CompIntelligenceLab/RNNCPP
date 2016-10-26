#include <stdio.h>
#include <math.h>

// Finite-difference derivatives vs exact derivatives of the function
// y = x^n  as a function of the increment eps and single vs double precision

//#define SINGLE

#ifdef SINGLE
typedef float REAL;
#else
typedef double REAL;
#endif

REAL f(REAL x, int n)
{
	REAL y = 1.000;
	for (int i=0; i < n; i++) 
		y *= x;

	//return y;
	return pow(x, n);
}

void derivative(REAL x, REAL eps, int n)
{
	REAL y = f(x, n);
	REAL deriv = (f(x+eps,n) - f(x-eps,n)) / (2.*eps);
	REAL exact_deriv = n*pow(x,n-1);

	REAL abs_err;
	REAL rel_err;

	abs_err = deriv - exact_deriv;
	rel_err = abs_err / exact_deriv;

	printf("eps= %e, rel_err= %14.7e\n", eps, rel_err);
}

int main()
{
	REAL eps = 1.e-3;
	REAL x;
	int n; 

	n = 200; // exponent
	x = 0.98;

	while (eps > 1.e-15) {
		derivative(x, eps, n);
		eps /= 2.;
	}
}
