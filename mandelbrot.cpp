#include <stdio.h>
#include <pycuda-complex.hpp>
#include <math.h>
#include <stdint.h>

typedef   pycuda::complex<double> pyComplex;
__device__ float norma(pyComplex z)
{
    return norm(z);
}

__global__ void mandelbrot(uint8_t *m, double x0, double y0,double dx, double dy, double power)
{
    int n_x = blockDim.x*gridDim.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int idy = threadIdx.y + blockDim.y*blockIdx.y;
    int threadId = idy*n_x+idx;

    pyComplex c(x0+dx*idx,y0+dy*idy);
    pyComplex z(x0+dx*idx,y0+dy*idy);
    int h = 0;
    float R = 2.0;
    while(h<255 && norma(z)<R){
        z=pow(z,power)+c;
        h+=1;
    }
    m[threadId]=h;
}