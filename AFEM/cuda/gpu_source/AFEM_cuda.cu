

#include <stdio.h>


#include <math.h>

#include <iostream>
#include "AFEM_cuda.cuh"



__global__ void hello_gpuprint(int num){
	printf("gpu print %d", &num);
}



__global__ void vecAdd(double *a, double *b, double *c, int n)
{
	// Get our global thread ID
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Make sure we do not go out of bounds
	if (id < n)
		c[id] = a[id] + b[id];
}

void cuda_tools::hello(){
	// Size of vectors
	int n = 100000;

	// Host input vectors
	double *h_a;
	double *h_b;
	//Host output vector
	double *h_c;

	// Device input vectors
	double *d_a;
	double *d_b;
	//Device output vector
	double *d_c;

	// Size, in bytes, of each vector
	size_t bytes = n*sizeof(double);

	// Allocate memory for each vector on host
	h_a = (double*)malloc(bytes);
	h_b = (double*)malloc(bytes);
	h_c = (double*)malloc(bytes);

	// Allocate memory for each vector on GPU
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	int i;
	// Initialize vectors on host
	for (i = 0; i < n; i++) {
		h_a[i] = sin(i)*sin(i);
		h_b[i] = cos(i)*cos(i);
	}

	// Copy host vectors to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	int blockSize, gridSize;

	// Number of threads in each thread block
	blockSize = 1024;

	// Number of thread blocks in grid
	gridSize = (int)ceil((float)n / blockSize);

	// Execute the kernel
	vecAdd << <gridSize, blockSize >> >(d_a, d_b, d_c, n);

	// Copy array back to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Sum up vector c and print result divided by n, this should equal 1 within error
	double sum = 0;
	for (i = 0; i<n; i++)
		sum += h_c[i];
	printf("final result: %f\n", sum / n);

	// Release device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Release host memory
	free(h_a);
	free(h_b);
	free(h_c);

}

//__global__ void make_K_cuda3d(double *E_vector, int *nodesInElem_device, double *x_vector, double *y_vector, double *z_vector, int *displaceInElem_device, float *d_A_dense, int *numnodes);//3D
//__global__ void make_K_cuda2d(double *K, int *nodesInElem, double *x_vector, double *y_vector, int *displaceInElem_device, float *d_A_dense, int numnodes, double thickness,double young_E,double nu,double alpha,double beta1,double beta2, double rho, double dt,double c_xi,int numE);//2D
//__global__ void make_global_K(void); 





