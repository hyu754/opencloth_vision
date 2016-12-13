

#include <stdio.h>


#include <math.h>

#include <iostream>
#include "AFEM_cuda.cuh"

#define nodesinelemX(node,el,nodesPerElem) (node + nodesPerElem*el) //the first entry is the element # the second entry would be the element number and the last one is the number of nodes/element
#define threeD21D(row_d,col_d,el_d,width_d,depth_d) (row_d+width_d*(col_d+depth_d*el_d)) //
#define nodesDisplacementX(dof,node,dimension) (dof + node*dimension)
#define IDX2C(i,j,ld) (((j)*(ld))+( i )) 

__global__ void hello_gpuprint(int num){
	printf("gpu print %d", &num);
}



//This is for the local K matrix
//NOTE:::: nu and E are not initilized
__device__ inline float atomicAdda(float* address, double value)

{

	float ret = atomicExch(address, 0.0f);

	float old = ret + (float)value;

	while ((old = atomicExch(address, old)) != 0.0f)

	{

		old = atomicExch(address, 0.0f) + old;

	}

	return ret;

};
__global__ void make_K_cuda3d(double *E_vector, int *nodesInElem, double *x_vector, double *y_vector, double *z_vector, int *displaceInElem_device, float *d_A_dense, int *numnodes) {
	//int x = threadIdx.x + blockIdx.x*blockDim.x; //if we have a 3D problem then this will go from 0 to 11
#if 1
	int row;
	int dummy_node;
	int loop_node;
	int dummy_row;
	int dummy_col;
	int DOF[12];
	int counter;
	int offset = threadIdx.x + blockIdx.x*blockDim.x; // offset will essentaillay be the element counter
	//int y_offset = threadIdx.y + blockIdx.y*blockDim.y;

	int max_limit = 12 * 12 * 4374;
	double E = 20000.0;
	double nu = 0.49;
	double x14 = x_vector[nodesInElem[nodesinelemX(0, offset, 4)]] - x_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double x24 = x_vector[nodesInElem[nodesinelemX(1, offset, 4)]] - x_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double x34 = x_vector[nodesInElem[nodesinelemX(2, offset, 4)]] - x_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double y14 = y_vector[nodesInElem[nodesinelemX(0, offset, 4)]] - y_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double y24 = y_vector[nodesInElem[nodesinelemX(1, offset, 4)]] - y_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double y34 = y_vector[nodesInElem[nodesinelemX(2, offset, 4)]] - y_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double z14 = z_vector[nodesInElem[nodesinelemX(0, offset, 4)]] - z_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double z24 = z_vector[nodesInElem[nodesinelemX(1, offset, 4)]] - z_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double z34 = z_vector[nodesInElem[nodesinelemX(2, offset, 4)]] - z_vector[nodesInElem[nodesinelemX(3, offset, 4)]];

	//std::cout << x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * 34) + z14*(x24*y34 - y24*x34) << std::endl;
	double det_J = (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34));

	double J_bar11 = (y24*z34 - z24*y34) / det_J;
	double J_bar12 = (z14*y34 - y14*z34) / det_J;
	double J_bar13 = (y14*z24 - z14*y24) / det_J;
	double J_bar21 = (z24*x34 - x24*z34) / det_J;
	double J_bar22 = (x14*z34 - z14*x34) / det_J;
	double J_bar23 = (z14*x24 - x14*z24) / det_J;
	double J_bar31 = (x24*y34 - y24*x34) / det_J;
	double J_bar32 = (y14*x34 - x14*y34) / det_J;
	double J_bar33 = (x14*y24 - y14*x24) / det_J;

	double J_star1 = -(J_bar11 + J_bar12 + J_bar13);
	double J_star2 = -(J_bar21 + J_bar22 + J_bar23);
	double J_star3 = -(J_bar31 + J_bar32 + J_bar33);



	E_vector[offset * 144 + 0] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 1] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 2] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 3] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 4] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 5] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 6] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 7] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 8] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 9] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 10] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 11] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 12] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 13] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 14] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 15] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 16] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 17] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 18] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 19] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 20] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 21] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 22] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 23] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 24] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 25] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 26] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 27] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 28] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 29] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 30] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 31] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 32] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 33] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 34] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 35] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 36] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 37] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 38] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 39] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 40] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 41] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 42] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 43] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 44] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 45] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 46] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 47] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 48] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 49] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 50] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 51] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 52] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 53] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 54] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 55] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 56] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 57] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 58] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 59] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 60] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 61] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 62] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 63] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 64] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 65] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 66] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 67] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 68] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 69] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 70] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 71] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 72] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 73] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 74] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 75] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 76] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 77] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 78] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 79] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 80] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 81] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 82] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 83] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 84] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 85] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 86] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 87] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 88] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 89] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 90] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 91] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 92] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 93] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 94] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 95] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 96] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 97] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 98] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 99] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 100] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 101] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 102] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 103] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 104] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 105] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 106] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 107] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 108] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 109] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 110] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 111] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 112] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 113] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 114] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 115] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 116] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 117] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 118] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 119] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 120] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 121] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 122] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 123] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 124] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 125] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 126] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 127] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 128] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 129] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 130] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 131] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 132] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 133] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 134] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 135] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 136] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 137] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 138] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 139] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 140] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 141] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 142] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	E_vector[offset * 144 + 143] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);


	counter = 0;
	//The two loops are responsible for finding the DOF (or q_i) for each element
	for (int npe = 0; npe < 4; npe++){
		dummy_node = nodesInElem[nodesinelemX(npe, offset, 4)]; // The row of the matrix we looking at will be k_th element and npe (nodes per element) 	
		for (int dof = 0; dof < 3; dof++){

			DOF[counter] = displaceInElem_device[nodesDisplacementX(dof, dummy_node, 3)];
			counter++;
		}
	}

	//we will use atomic add because we will be writting to a single location multiple times (perhaps) 
	for (int c = 0; c < 12; c++){
		for (int r = 0; r < 12; r++){

			//d_A_dense[IDX2C(DOF[c], DOF[r], 3000)] = d_A_dense[IDX2C(DOF[c], DOF[r], 3000)] + E_vector[offset * 144 + c*12+r];
			atomicAdda(&(d_A_dense[IDX2C(DOF[c], DOF[r], 3 * (*numnodes))]), E_vector[offset * 144 + c * 12 + r]);
			//IDX2C(DOF[c], DOF[r], 3000)
			//K[IDX2C(DOF[r], DOF[c], numP*dim)] = K[IDX2C(DOF[r], DOF[c], numP*dim)] + E[k][r][c];
		}
	}

#endif // 0




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





