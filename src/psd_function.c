#include <math.h>
#include <stdio.h>
#define DELTA 1

void psd_function(double *x, double *y, int n){
    int i=0;
    for(i=0;i<n;i++, x++, y++)
        *y=1./(1.+pow(M_PI*(*x),2));
}
