

#include <stdio.h>


#include <math.h>

#include <iostream>
#include "AFEM_cuda.cuh"
#include "cuda.h"
#include <cuda_runtime.h>

#define nodesinelemX(node,el,nodesPerElem) (node + nodesPerElem*el) //the first entry is the element # the second entry would be the element number and the last one is the number of nodes/element
#define threeD21D(row_d,col_d,el_d,width_d,depth_d) (row_d+width_d*(col_d+depth_d*el_d)) //
#define nodesDisplacementX(dof,node,dimension) (dof + node*dimension)
#define IDX2C(i,j,ld) (((j)*(ld))+( i )) 


__global__ void gpu_print_vec(AFEM::element *in_vec){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	printf("%d", in_vec[x].nodes_in_elem[0]);
}
__device__ void print_kernel() {
	printf("Hello from block");
}
__device__ void find_Jacobian_and_localK(AFEM::element *in_element){
	
	float x14 = in_element->position_info[0].x - in_element->position_info[3].x;
	float x24 = in_element->position_info[1].x - in_element->position_info[3].x;
	float x34 = in_element->position_info[2].x - in_element->position_info[3].x;
	float y14 = in_element->position_info[0].y - in_element->position_info[3].y;
	float y24 = in_element->position_info[1].y - in_element->position_info[3].y;
	float y34 = in_element->position_info[2].y - in_element->position_info[3].y;
	float z14 = in_element->position_info[0].z - in_element->position_info[3].z;
	float z24 = in_element->position_info[1].z - in_element->position_info[3].z;
	float z34 = in_element->position_info[2].z - in_element->position_info[3].z;

	//std::cout << x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * 34) + z14*(x24*y34 - y24*x34) << std::endl;
	float det_J = (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34));
	float J_bar11 = (y24*z34 - z24*y34) / det_J;
	float J_bar12 = (z14*y34 - y14*z34) / det_J;
	float J_bar13 = (y14*z24 - z14*y24) / det_J;
	float J_bar21 = (z24*x34 - x24*z34) / det_J;
	float J_bar22 = (x14*z34 - z14*x34) / det_J;
	float J_bar23 = (z14*x24 - x14*z24) / det_J;
	float J_bar31 = (x24*y34 - y24*x34) / det_J;
	float J_bar32 = (y14*x34 - x14*y34) / det_J;
	float J_bar33 = (x14*y24 - y14*x24) / det_J;

	float J_star1 = -(J_bar11 + J_bar12 + J_bar13);
	float J_star2 = -(J_bar21 + J_bar22 + J_bar23);
	float J_star3 = -(J_bar31 + J_bar32 + J_bar33);

	in_element->Jacobian = det_J;

	

	float E = 100000.0;
	float nu = 0.49;
	

	in_element->local_K[  0] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	//in_element->local_K[0] = det_J;
	in_element->local_K[  1] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  2] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  3] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  4] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  5] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  6] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  7] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  8] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  9] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  10] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  11] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  12] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  13] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  14] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  15] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  16] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  17] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  18] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  19] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  20] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  21] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  22] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  23] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  24] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  25] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  26] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  27] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  28] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  29] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  30] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  31] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  32] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  33] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  34] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  35] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  36] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  37] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  38] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  39] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  40] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  41] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  42] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  43] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  44] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  45] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  46] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  47] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  48] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  49] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  50] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  51] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  52] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  53] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  54] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  55] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  56] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  57] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  58] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  59] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  60] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  61] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  62] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  63] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  64] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  65] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  66] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  67] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  68] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  69] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  70] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  71] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  72] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  73] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  74] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  75] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  76] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  77] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  78] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  79] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  80] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  81] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  82] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  83] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  84] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  85] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  86] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  87] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  88] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  89] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  90] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  91] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  92] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  93] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  94] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  95] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  96] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  97] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  98] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  99] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  100] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  101] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  102] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  103] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  104] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  105] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  106] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  107] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  108] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  109] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  110] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  111] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  112] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  113] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  114] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  115] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  116] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  117] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  118] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  119] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  120] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  121] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  122] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  123] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  124] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  125] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  126] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  127] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  128] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[  129] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  130] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  131] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  132] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  133] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  134] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  135] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  136] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  137] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  138] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  139] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  140] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  141] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  142] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[  143] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);

	//return (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34));
}


//Atomic add
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
__global__ void gpu_make_K(
	AFEM::element *in_vec,
	int numElem,
	int numNodes, 
	float *K_d
	)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < numElem){
		find_Jacobian_and_localK(&in_vec[x]);

		//K_d[x] = (in_vec[x]).local_K[0];
		int DOF[12];
		int counter = 0;
		//The two loops are responsible for finding the DOF (or q_i) for each element
		for (int npe = 0; npe < 4; npe++){
			//dummy_node = nodesInElem[nodesinelemX(npe, offset, 4)]; // The row of the matrix we looking at will be k_th element and npe (nodes per element) 	
			for (int dof = 0; dof < 3; dof++){

				DOF[counter] = in_vec[x].position_info[npe].displacement_index[dof];
				counter++;
			}
		}

		for (int c = 0; c < 12; c++){
			for (int r = 0; r < 12; r++){

				//d_A_dense[IDX2C(DOF[c], DOF[r], 3000)] = d_A_dense[IDX2C(DOF[c], DOF[r], 3000)] + E_vector[offset * 144 + c*12+r];
				atomicAdda(&(K_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]),in_vec[x].local_K[c*12+r]);
				//IDX2C(DOF[c], DOF[r], 3000)
				//K[IDX2C(DOF[r], DOF[c], numP*dim)] = K[IDX2C(DOF[r], DOF[c], numP*dim)] + E[k][r][c];
			}
		}
		//printf("hi");
	}
	
}

//Resets the K matrix to zero
__global__ void reset_K(
	float *K_d,
	int numNodes,
	int dim

){
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < numNodes*dim*numNodes*dim){
		K_d[x] = 0;
	}
}


//Allocates the cpu and gpu memory, and then copy necessary data to them
void cuda_tools::allocate_copy_CUDA_geometry_data(AFEM::element *in_array, int num_elem, int num_nodes, int dim){
	//cpu allocation of memeory

	K_h = (float*)malloc( sizeof(*K_h)*dim*num_nodes*dim*num_nodes);


	//cuda allocation of memory
	cudaMalloc((void**)&elem_array_d, sizeof(AFEM::element) *num_elem); //element array
	cudaMalloc((void**)&K_d, sizeof(*K_d)*dim*num_nodes*dim*num_nodes); //final global K matrix container


	//cuda copy of memory from host to device
	cudaMemcpy(elem_array_d, in_array, sizeof(AFEM::element) *num_elem, cudaMemcpyHostToDevice);
	cudaMemset(K_d, 0, sizeof(*K_d)*dim*num_nodes*dim*num_nodes); //initialize the vector K_d to zero
	
}


void cuda_tools::copy_data_from_cuda(int num_nodes,int dim){

	
	
	
}


void cuda_tools::make_K(int num_elem,int num_nodes){
	int blocks, threads;
	if (num_elem <= 256){
		blocks = 16;
		threads = 16;
	}
	else {
		blocks = (num_elem + 1) / 256;
		threads = 256;
	}
	gpu_make_K << <blocks, threads >> > (elem_array_d, num_elem,num_nodes, K_d);
	cudaMemcpy(K_h, K_d, 100, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 1; i++){
		std::cout << K_h[0] << " ";
	}
	//cudaMemset(K_d, 0, sizeof(*K_d)*dim*num_nodes*dim*num_nodes); //initialize the vector K_d to zero

	std::cout << std::endl;
	reset_K << <blocks, threads >> >( K_d,num_nodes, 3);

}





cuda_tools::~cuda_tools(){
	free(K_h);
	cudaFree(K_d);

}

__global__ void vecAdd(double *a, double *b, double *c, int n)
{
	// Get our global thread ID
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Make sure we do not go out of bounds
	if (id < n)
		c[id] = a[id] + b[id];
}

void hello(){
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





