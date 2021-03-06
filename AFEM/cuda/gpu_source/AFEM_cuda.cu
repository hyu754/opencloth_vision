

#include <stdio.h>
#include <math.h>
#include <iostream>
#include "AFEM_cuda.cuh"
#include "cuda.h"
#include <cuda_runtime.h>
#include <fstream>

//#define DYNAMIC
#define nodesinelemX(node,el,nodesPerElem) (node + nodesPerElem*el) //the first entry is the element # the second entry would be the element number and the last one is the number of nodes/element
#define threeD21D(row_d,col_d,el_d,width_d,depth_d) (row_d+width_d*(col_d+depth_d*el_d)) //
#define nodesDisplacementX(dof,node,dimension) (dof + node*dimension)
#define IDX2C(i,j,ld) (((j)*(ld))+( i )) 
__device__ float _dd_ = 10000;
__device__ int node_max = 0;
//Atomic add for global K matrix assembly
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
//__device__ void find_localM(AFEM::element *in_element){
//	float det_J = in_element->Jacobian;
//	float rho = 1000.0; //in_element->density;
//	in_element->local_M[0] = det_J*rho / 3;
//	in_element->local_M[1] = 0;
//	in_element->local_M[2] = 0;
//	in_element->local_M[3] = det_J*rho / 4;
//	in_element->local_M[4] = 0;
//	in_element->local_M[5] = 0;
//	in_element->local_M[6] = det_J*rho / 4;
//	in_element->local_M[7] = 0;
//	in_element->local_M[8] = 0;
//	in_element->local_M[9] = -det_J*rho / 3;
//	in_element->local_M[10] = 0;
//	in_element->local_M[11] = 0;
//	in_element->local_M[12] = 0;
//	in_element->local_M[13] = det_J*rho / 3;
//	in_element->local_M[14] = 0;
//	in_element->local_M[15] = 0;
//	in_element->local_M[16] = det_J*rho / 4;
//	in_element->local_M[17] = 0;
//	in_element->local_M[18] = 0;
//	in_element->local_M[19] = det_J*rho / 4;
//	in_element->local_M[20] = 0;
//	in_element->local_M[21] = 0;
//	in_element->local_M[22] = -det_J*rho / 3;
//	in_element->local_M[23] = 0;
//	in_element->local_M[24] = 0;
//	in_element->local_M[25] = 0;
//	in_element->local_M[26] = det_J*rho / 3;
//	in_element->local_M[27] = 0;
//	in_element->local_M[28] = 0;
//	in_element->local_M[29] = det_J*rho / 4;
//	in_element->local_M[30] = 0;
//	in_element->local_M[31] = 0;
//	in_element->local_M[32] = det_J*rho / 4;
//	in_element->local_M[33] = 0;
//	in_element->local_M[34] = 0;
//	in_element->local_M[35] = -det_J*rho / 3;
//	in_element->local_M[36] = det_J*rho / 4;
//	in_element->local_M[37] = 0;
//	in_element->local_M[38] = 0;
//	in_element->local_M[39] = det_J*rho / 3;
//	in_element->local_M[40] = 0;
//	in_element->local_M[41] = 0;
//	in_element->local_M[42] = det_J*rho / 4;
//	in_element->local_M[43] = 0;
//	in_element->local_M[44] = 0;
//	in_element->local_M[45] = -det_J*rho / 3;
//	in_element->local_M[46] = 0;
//	in_element->local_M[47] = 0;
//	in_element->local_M[48] = 0;
//	in_element->local_M[49] = det_J*rho / 4;
//	in_element->local_M[50] = 0;
//	in_element->local_M[51] = 0;
//	in_element->local_M[52] = det_J*rho / 3;
//	in_element->local_M[53] = 0;
//	in_element->local_M[54] = 0;
//	in_element->local_M[55] = det_J*rho / 4;
//	in_element->local_M[56] = 0;
//	in_element->local_M[57] = 0;
//	in_element->local_M[58] = -det_J*rho / 3;
//	in_element->local_M[59] = 0;
//	in_element->local_M[60] = 0;
//	in_element->local_M[61] = 0;
//	in_element->local_M[62] = det_J*rho / 4;
//	in_element->local_M[63] = 0;
//	in_element->local_M[64] = 0;
//	in_element->local_M[65] = det_J*rho / 3;
//	in_element->local_M[66] = 0;
//	in_element->local_M[67] = 0;
//	in_element->local_M[68] = det_J*rho / 4;
//	in_element->local_M[69] = 0;
//	in_element->local_M[70] = 0;
//	in_element->local_M[71] = -det_J*rho / 3;
//	in_element->local_M[72] = det_J*rho / 4;
//	in_element->local_M[73] = 0;
//	in_element->local_M[74] = 0;
//	in_element->local_M[75] = det_J*rho / 4;
//	in_element->local_M[76] = 0;
//	in_element->local_M[77] = 0;
//	in_element->local_M[78] = det_J*rho / 3;
//	in_element->local_M[79] = 0;
//	in_element->local_M[80] = 0;
//	in_element->local_M[81] = -det_J*rho / 3;
//	in_element->local_M[82] = 0;
//	in_element->local_M[83] = 0;
//	in_element->local_M[84] = 0;
//	in_element->local_M[85] = det_J*rho / 4;
//	in_element->local_M[86] = 0;
//	in_element->local_M[87] = 0;
//	in_element->local_M[88] = det_J*rho / 4;
//	in_element->local_M[89] = 0;
//	in_element->local_M[90] = 0;
//	in_element->local_M[91] = det_J*rho / 3;
//	in_element->local_M[92] = 0;
//	in_element->local_M[93] = 0;
//	in_element->local_M[94] = -det_J*rho / 3;
//	in_element->local_M[95] = 0;
//	in_element->local_M[96] = 0;
//	in_element->local_M[97] = 0;
//	in_element->local_M[98] = det_J*rho / 4;
//	in_element->local_M[99] = 0;
//	in_element->local_M[100] = 0;
//	in_element->local_M[101] = det_J*rho / 4;
//	in_element->local_M[102] = 0;
//	in_element->local_M[103] = 0;
//	in_element->local_M[104] = det_J*rho / 3;
//	in_element->local_M[105] = 0;
//	in_element->local_M[106] = 0;
//	in_element->local_M[107] = -det_J*rho / 3;
//	in_element->local_M[108] = -det_J*rho / 3;
//	in_element->local_M[109] = 0;
//	in_element->local_M[110] = 0;
//	in_element->local_M[111] = -det_J*rho / 3;
//	in_element->local_M[112] = 0;
//	in_element->local_M[113] = 0;
//	in_element->local_M[114] = -det_J*rho / 3;
//	in_element->local_M[115] = 0;
//	in_element->local_M[116] = 0;
//	in_element->local_M[117] = det_J*rho / 2;
//	in_element->local_M[118] = 0;
//	in_element->local_M[119] = 0;
//	in_element->local_M[120] = 0;
//	in_element->local_M[121] = -det_J*rho / 3;
//	in_element->local_M[122] = 0;
//	in_element->local_M[123] = 0;
//	in_element->local_M[124] = -det_J*rho / 3;
//	in_element->local_M[125] = 0;
//	in_element->local_M[126] = 0;
//	in_element->local_M[127] = -det_J*rho / 3;
//	in_element->local_M[128] = 0;
//	in_element->local_M[129] = 0;
//	in_element->local_M[130] = det_J*rho / 2;
//	in_element->local_M[131] = 0;
//	in_element->local_M[132] = 0;
//	in_element->local_M[133] = 0;
//	in_element->local_M[134] = -det_J*rho / 3;
//	in_element->local_M[135] = 0;
//	in_element->local_M[136] = 0;
//	in_element->local_M[137] = -det_J*rho / 3;
//	in_element->local_M[138] = 0;
//	in_element->local_M[139] = 0;
//	in_element->local_M[140] = -det_J*rho / 3;
//	in_element->local_M[141] = 0;
//	in_element->local_M[142] = 0;
//	in_element->local_M[143] = det_J*rho / 2;
//}
__device__ void find_Jacobian_localK_localM(AFEM::element *in_element, const AFEM::position_3D *in_pos){

	/*float x14 = in_element->position_info[0].x - in_element->position_info[3].x;
	float x24 = in_element->position_info[1].x - in_element->position_info[3].x;
	float x34 = in_element->position_info[2].x - in_element->position_info[3].x;*/

	//four node positions
	AFEM::position_3D n1, n2, n3, n4;
	n1 = in_pos[in_element->nodes_in_elem[0]];
	n2 = in_pos[in_element->nodes_in_elem[1]];
	n3 = in_pos[in_element->nodes_in_elem[2]];
	n4 = in_pos[in_element->nodes_in_elem[3]];


	float x14 = n1.x - n4.x;
	float x24 = n2.x - n4.x;
	float x34 = n3.x - n4.x;

	float y14 = n1.y - n4.y;
	float y24 = n2.y - n4.y;
	float y34 = n3.y - n4.y;


	float z14 = n1.z - n4.z;
	float z24 = n2.z - n4.z;
	float z34 = n3.z - n4.z;



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

	in_element->volume = det_J / 6.0;

	float E = 50000.0;
	float nu = 0.49;

#if 1

	in_element->local_K[0] = 0.166666666666667*E*J_bar11*J_bar11*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[1] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[2] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[3] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[4] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[5] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[6] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[7] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[8] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[9] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[10] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[11] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[12] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[13] = 0.166666666666667*E*J_bar11*J_bar11*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[14] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[15] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[16] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[17] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[18] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[19] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[20] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[21] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[22] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[23] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[24] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[25] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[26] = 0.166666666666667*E*J_bar11*J_bar11*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[27] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[28] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[29] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[30] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[31] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[32] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[33] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[34] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[35] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[36] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[37] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[38] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[39] = 0.166666666666667*E*J_bar12*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[40] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[41] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[42] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[43] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[44] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[45] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[46] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[47] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[48] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[49] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[50] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[51] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[52] = 0.166666666666667*E*J_bar12*J_bar12*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[53] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[54] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[55] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[56] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[57] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[58] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[59] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[60] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[61] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[62] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[63] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[64] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[65] = 0.166666666666667*E*J_bar12*J_bar12*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[66] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[67] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[68] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[69] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[70] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[71] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[72] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[73] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[74] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[75] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[76] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[77] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[78] = 0.166666666666667*E*J_bar13*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[79] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[80] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[81] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[82] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[83] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[84] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[85] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[86] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[87] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[88] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[89] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[90] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[91] = 0.166666666666667*E*J_bar13*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[92] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[93] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[94] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[95] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[96] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[97] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[98] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[99] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[100] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[101] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[102] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[103] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[104] = 0.166666666666667*E*J_bar13*J_bar13*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[105] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[106] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[107] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[108] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[109] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[110] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[111] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[112] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[113] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[114] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[115] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[116] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[117] = 0.166666666666667*E*J_star1*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[118] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[119] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[120] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[121] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[122] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[123] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[124] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[125] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[126] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[127] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[128] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[129] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[130] = 0.166666666666667*E*J_star1*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[131] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[132] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[133] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[134] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[135] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[136] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[137] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[138] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[139] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[140] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[141] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[142] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[143] = 0.166666666666667*E*J_star1*J_star1*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2*det_J*(-2 * nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);

#else


	in_element->local_K[0] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[1] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[2] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[3] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[4] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[5] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[6] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[7] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[8] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[9] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[10] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[11] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[12] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[13] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[14] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[15] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[16] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[17] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[18] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[19] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[20] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[21] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[22] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[23] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[24] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[25] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[26] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[27] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[28] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[29] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[30] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[31] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[32] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[33] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[34] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[35] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[36] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[37] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[38] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[39] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[40] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[41] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[42] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[43] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[44] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[45] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[46] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[47] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[48] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[49] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[50] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[51] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[52] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[53] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[54] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[55] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[56] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[57] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[58] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[59] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[60] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[61] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[62] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[63] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[64] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[65] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[66] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[67] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[68] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[69] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[70] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[71] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[72] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[73] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[74] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[75] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[76] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[77] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[78] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[79] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[80] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[81] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[82] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[83] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[84] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[85] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[86] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[87] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[88] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[89] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[90] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[91] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[92] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[93] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[94] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[95] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[96] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[97] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[98] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[99] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[100] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[101] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[102] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[103] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[104] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[105] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[106] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[107] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[108] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[109] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[110] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[111] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[112] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[113] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[114] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[115] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[116] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[117] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[118] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[119] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[120] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[121] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[122] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[123] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[124] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[125] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[126] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[127] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[128] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
	in_element->local_K[129] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[130] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[131] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[132] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[133] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[134] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[135] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[136] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[137] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[138] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[139] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[140] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
	in_element->local_K[141] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[142] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
	in_element->local_K[143] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);

#endif // 0
	float rho = 1000.0;
	/*in_element->local_M[0] = 0.166666666666667*det_J*(J_bar11*J_bar11*rho + J_bar21*J_bar21*rho + J_bar31*J_bar31*rho);
	in_element->local_M[1] = 0.166666666666667*J_bar11*J_bar21*det_J*rho;
	in_element->local_M[2] = 0.166666666666667*J_bar11*J_bar31*det_J*rho;
	in_element->local_M[3] = 0.166666666666667*det_J*(J_bar11*J_bar12*rho + J_bar21*J_bar22*rho + J_bar31*J_bar32*rho);
	in_element->local_M[4] = 0.166666666666667*J_bar12*J_bar21*det_J*rho;
	in_element->local_M[5] = 0.166666666666667*J_bar12*J_bar31*det_J*rho;
	in_element->local_M[6] = 0.166666666666667*det_J*(J_bar11*J_bar13*rho + J_bar21*J_bar23*rho + J_bar31*J_bar33*rho);
	in_element->local_M[7] = 0.166666666666667*J_bar13*J_bar21*det_J*rho;
	in_element->local_M[8] = 0.166666666666667*J_bar13*J_bar31*det_J*rho;
	in_element->local_M[9] = 0.166666666666667*det_J*(J_bar11*J_star1*rho + J_bar21*J_star2*rho + J_bar31*J_star3*rho);
	in_element->local_M[10] = 0.166666666666667*J_bar21*J_star1*det_J*rho;
	in_element->local_M[11] = 0.166666666666667*J_bar31*J_star1*det_J*rho;
	in_element->local_M[12] = 0.166666666666667*J_bar11*J_bar21*det_J*rho;
	in_element->local_M[13] = 0.166666666666667*det_J*(J_bar11*J_bar11*rho + J_bar21*J_bar21*rho + J_bar31*J_bar31*rho);
	in_element->local_M[14] = 0.166666666666667*J_bar21*J_bar31*det_J*rho;
	in_element->local_M[15] = 0.166666666666667*J_bar11*J_bar22*det_J*rho;
	in_element->local_M[16] = 0.166666666666667*det_J*(J_bar11*J_bar12*rho + J_bar21*J_bar22*rho + J_bar31*J_bar32*rho);
	in_element->local_M[17] = 0.166666666666667*J_bar22*J_bar31*det_J*rho;
	in_element->local_M[18] = 0.166666666666667*J_bar11*J_bar23*det_J*rho;
	in_element->local_M[19] = 0.166666666666667*det_J*(J_bar11*J_bar13*rho + J_bar21*J_bar23*rho + J_bar31*J_bar33*rho);
	in_element->local_M[20] = 0.166666666666667*J_bar23*J_bar31*det_J*rho;
	in_element->local_M[21] = 0.166666666666667*J_bar11*J_star2*det_J*rho;
	in_element->local_M[22] = 0.166666666666667*det_J*(J_bar11*J_star1*rho + J_bar21*J_star2*rho + J_bar31*J_star3*rho);
	in_element->local_M[23] = 0.166666666666667*J_bar31*J_star2*det_J*rho;
	in_element->local_M[24] = 0.166666666666667*J_bar11*J_bar31*det_J*rho;
	in_element->local_M[25] = 0.166666666666667*J_bar21*J_bar31*det_J*rho;
	in_element->local_M[26] = 0.166666666666667*det_J*(J_bar11*J_bar11*rho + J_bar21*J_bar21*rho + J_bar31*J_bar31*rho);
	in_element->local_M[27] = 0.166666666666667*J_bar11*J_bar32*det_J*rho;
	in_element->local_M[28] = 0.166666666666667*J_bar21*J_bar32*det_J*rho;
	in_element->local_M[29] = 0.166666666666667*det_J*(J_bar11*J_bar12*rho + J_bar21*J_bar22*rho + J_bar31*J_bar32*rho);
	in_element->local_M[30] = 0.166666666666667*J_bar11*J_bar33*det_J*rho;
	in_element->local_M[31] = 0.166666666666667*J_bar21*J_bar33*det_J*rho;
	in_element->local_M[32] = 0.166666666666667*det_J*(J_bar11*J_bar13*rho + J_bar21*J_bar23*rho + J_bar31*J_bar33*rho);
	in_element->local_M[33] = 0.166666666666667*J_bar11*J_star3*det_J*rho;
	in_element->local_M[34] = 0.166666666666667*J_bar21*J_star3*det_J*rho;
	in_element->local_M[35] = 0.166666666666667*det_J*(J_bar11*J_star1*rho + J_bar21*J_star2*rho + J_bar31*J_star3*rho);
	in_element->local_M[36] = 0.166666666666667*det_J*(J_bar11*J_bar12*rho + J_bar21*J_bar22*rho + J_bar31*J_bar32*rho);
	in_element->local_M[37] = 0.166666666666667*J_bar11*J_bar22*det_J*rho;
	in_element->local_M[38] = 0.166666666666667*J_bar11*J_bar32*det_J*rho;
	in_element->local_M[39] = 0.166666666666667*det_J*(J_bar12*J_bar12*rho + J_bar22*J_bar22*rho + J_bar32*J_bar32*rho);
	in_element->local_M[40] = 0.166666666666667*J_bar12*J_bar22*det_J*rho;
	in_element->local_M[41] = 0.166666666666667*J_bar12*J_bar32*det_J*rho;
	in_element->local_M[42] = 0.166666666666667*det_J*(J_bar12*J_bar13*rho + J_bar22*J_bar23*rho + J_bar32*J_bar33*rho);
	in_element->local_M[43] = 0.166666666666667*J_bar13*J_bar22*det_J*rho;
	in_element->local_M[44] = 0.166666666666667*J_bar13*J_bar32*det_J*rho;
	in_element->local_M[45] = 0.166666666666667*det_J*(J_bar12*J_star1*rho + J_bar22*J_star2*rho + J_bar32*J_star3*rho);
	in_element->local_M[46] = 0.166666666666667*J_bar22*J_star1*det_J*rho;
	in_element->local_M[47] = 0.166666666666667*J_bar32*J_star1*det_J*rho;
	in_element->local_M[48] = 0.166666666666667*J_bar12*J_bar21*det_J*rho;
	in_element->local_M[49] = 0.166666666666667*det_J*(J_bar11*J_bar12*rho + J_bar21*J_bar22*rho + J_bar31*J_bar32*rho);
	in_element->local_M[50] = 0.166666666666667*J_bar21*J_bar32*det_J*rho;
	in_element->local_M[51] = 0.166666666666667*J_bar12*J_bar22*det_J*rho;
	in_element->local_M[52] = 0.166666666666667*det_J*(J_bar12*J_bar12*rho + J_bar22*J_bar22*rho + J_bar32*J_bar32*rho);
	in_element->local_M[53] = 0.166666666666667*J_bar22*J_bar32*det_J*rho;
	in_element->local_M[54] = 0.166666666666667*J_bar12*J_bar23*det_J*rho;
	in_element->local_M[55] = 0.166666666666667*det_J*(J_bar12*J_bar13*rho + J_bar22*J_bar23*rho + J_bar32*J_bar33*rho);
	in_element->local_M[56] = 0.166666666666667*J_bar23*J_bar32*det_J*rho;
	in_element->local_M[57] = 0.166666666666667*J_bar12*J_star2*det_J*rho;
	in_element->local_M[58] = 0.166666666666667*det_J*(J_bar12*J_star1*rho + J_bar22*J_star2*rho + J_bar32*J_star3*rho);
	in_element->local_M[59] = 0.166666666666667*J_bar32*J_star2*det_J*rho;
	in_element->local_M[60] = 0.166666666666667*J_bar12*J_bar31*det_J*rho;
	in_element->local_M[61] = 0.166666666666667*J_bar22*J_bar31*det_J*rho;
	in_element->local_M[62] = 0.166666666666667*det_J*(J_bar11*J_bar12*rho + J_bar21*J_bar22*rho + J_bar31*J_bar32*rho);
	in_element->local_M[63] = 0.166666666666667*J_bar12*J_bar32*det_J*rho;
	in_element->local_M[64] = 0.166666666666667*J_bar22*J_bar32*det_J*rho;
	in_element->local_M[65] = 0.166666666666667*det_J*(J_bar12*J_bar12*rho + J_bar22*J_bar22*rho + J_bar32*J_bar32*rho);
	in_element->local_M[66] = 0.166666666666667*J_bar12*J_bar33*det_J*rho;
	in_element->local_M[67] = 0.166666666666667*J_bar22*J_bar33*det_J*rho;
	in_element->local_M[68] = 0.166666666666667*det_J*(J_bar12*J_bar13*rho + J_bar22*J_bar23*rho + J_bar32*J_bar33*rho);
	in_element->local_M[69] = 0.166666666666667*J_bar12*J_star3*det_J*rho;
	in_element->local_M[70] = 0.166666666666667*J_bar22*J_star3*det_J*rho;
	in_element->local_M[71] = 0.166666666666667*det_J*(J_bar12*J_star1*rho + J_bar22*J_star2*rho + J_bar32*J_star3*rho);
	in_element->local_M[72] = 0.166666666666667*det_J*(J_bar11*J_bar13*rho + J_bar21*J_bar23*rho + J_bar31*J_bar33*rho);
	in_element->local_M[73] = 0.166666666666667*J_bar11*J_bar23*det_J*rho;
	in_element->local_M[74] = 0.166666666666667*J_bar11*J_bar33*det_J*rho;
	in_element->local_M[75] = 0.166666666666667*det_J*(J_bar12*J_bar13*rho + J_bar22*J_bar23*rho + J_bar32*J_bar33*rho);
	in_element->local_M[76] = 0.166666666666667*J_bar12*J_bar23*det_J*rho;
	in_element->local_M[77] = 0.166666666666667*J_bar12*J_bar33*det_J*rho;
	in_element->local_M[78] = 0.166666666666667*det_J*(J_bar13*J_bar13*rho + J_bar23*J_bar23*rho + J_bar33*J_bar33*rho);
	in_element->local_M[79] = 0.166666666666667*J_bar13*J_bar23*det_J*rho;
	in_element->local_M[80] = 0.166666666666667*J_bar13*J_bar33*det_J*rho;
	in_element->local_M[81] = 0.166666666666667*det_J*(J_bar13*J_star1*rho + J_bar23*J_star2*rho + J_bar33*J_star3*rho);
	in_element->local_M[82] = 0.166666666666667*J_bar23*J_star1*det_J*rho;
	in_element->local_M[83] = 0.166666666666667*J_bar33*J_star1*det_J*rho;
	in_element->local_M[84] = 0.166666666666667*J_bar13*J_bar21*det_J*rho;
	in_element->local_M[85] = 0.166666666666667*det_J*(J_bar11*J_bar13*rho + J_bar21*J_bar23*rho + J_bar31*J_bar33*rho);
	in_element->local_M[86] = 0.166666666666667*J_bar21*J_bar33*det_J*rho;
	in_element->local_M[87] = 0.166666666666667*J_bar13*J_bar22*det_J*rho;
	in_element->local_M[88] = 0.166666666666667*det_J*(J_bar12*J_bar13*rho + J_bar22*J_bar23*rho + J_bar32*J_bar33*rho);
	in_element->local_M[89] = 0.166666666666667*J_bar22*J_bar33*det_J*rho;
	in_element->local_M[90] = 0.166666666666667*J_bar13*J_bar23*det_J*rho;
	in_element->local_M[91] = 0.166666666666667*det_J*(J_bar13*J_bar13*rho + J_bar23*J_bar23*rho + J_bar33*J_bar33*rho);
	in_element->local_M[92] = 0.166666666666667*J_bar23*J_bar33*det_J*rho;
	in_element->local_M[93] = 0.166666666666667*J_bar13*J_star2*det_J*rho;
	in_element->local_M[94] = 0.166666666666667*det_J*(J_bar13*J_star1*rho + J_bar23*J_star2*rho + J_bar33*J_star3*rho);
	in_element->local_M[95] = 0.166666666666667*J_bar33*J_star2*det_J*rho;
	in_element->local_M[96] = 0.166666666666667*J_bar13*J_bar31*det_J*rho;
	in_element->local_M[97] = 0.166666666666667*J_bar23*J_bar31*det_J*rho;
	in_element->local_M[98] = 0.166666666666667*det_J*(J_bar11*J_bar13*rho + J_bar21*J_bar23*rho + J_bar31*J_bar33*rho);
	in_element->local_M[99] = 0.166666666666667*J_bar13*J_bar32*det_J*rho;
	in_element->local_M[100] = 0.166666666666667*J_bar23*J_bar32*det_J*rho;
	in_element->local_M[101] = 0.166666666666667*det_J*(J_bar12*J_bar13*rho + J_bar22*J_bar23*rho + J_bar32*J_bar33*rho);
	in_element->local_M[102] = 0.166666666666667*J_bar13*J_bar33*det_J*rho;
	in_element->local_M[103] = 0.166666666666667*J_bar23*J_bar33*det_J*rho;
	in_element->local_M[104] = 0.166666666666667*det_J*(J_bar13*J_bar13*rho + J_bar23*J_bar23*rho + J_bar33*J_bar33*rho);
	in_element->local_M[105] = 0.166666666666667*J_bar13*J_star3*det_J*rho;
	in_element->local_M[106] = 0.166666666666667*J_bar23*J_star3*det_J*rho;
	in_element->local_M[107] = 0.166666666666667*det_J*(J_bar13*J_star1*rho + J_bar23*J_star2*rho + J_bar33*J_star3*rho);
	in_element->local_M[108] = 0.166666666666667*det_J*(J_bar11*J_star1*rho + J_bar21*J_star2*rho + J_bar31*J_star3*rho);
	in_element->local_M[109] = 0.166666666666667*J_bar11*J_star2*det_J*rho;
	in_element->local_M[110] = 0.166666666666667*J_bar11*J_star3*det_J*rho;
	in_element->local_M[111] = 0.166666666666667*det_J*(J_bar12*J_star1*rho + J_bar22*J_star2*rho + J_bar32*J_star3*rho);
	in_element->local_M[112] = 0.166666666666667*J_bar12*J_star2*det_J*rho;
	in_element->local_M[113] = 0.166666666666667*J_bar12*J_star3*det_J*rho;
	in_element->local_M[114] = 0.166666666666667*det_J*(J_bar13*J_star1*rho + J_bar23*J_star2*rho + J_bar33*J_star3*rho);
	in_element->local_M[115] = 0.166666666666667*J_bar13*J_star2*det_J*rho;
	in_element->local_M[116] = 0.166666666666667*J_bar13*J_star3*det_J*rho;
	in_element->local_M[117] = 0.166666666666667*det_J*(J_star1*J_star1*rho + J_star2*J_star2*rho + J_star3*J_star3*rho);
	in_element->local_M[118] = 0.166666666666667*J_star1*J_star2*det_J*rho;
	in_element->local_M[119] = 0.166666666666667*J_star1*J_star3*det_J*rho;
	in_element->local_M[120] = 0.166666666666667*J_bar21*J_star1*det_J*rho;
	in_element->local_M[121] = 0.166666666666667*det_J*(J_bar11*J_star1*rho + J_bar21*J_star2*rho + J_bar31*J_star3*rho);
	in_element->local_M[122] = 0.166666666666667*J_bar21*J_star3*det_J*rho;
	in_element->local_M[123] = 0.166666666666667*J_bar22*J_star1*det_J*rho;
	in_element->local_M[124] = 0.166666666666667*det_J*(J_bar12*J_star1*rho + J_bar22*J_star2*rho + J_bar32*J_star3*rho);
	in_element->local_M[125] = 0.166666666666667*J_bar22*J_star3*det_J*rho;
	in_element->local_M[126] = 0.166666666666667*J_bar23*J_star1*det_J*rho;
	in_element->local_M[127] = 0.166666666666667*det_J*(J_bar13*J_star1*rho + J_bar23*J_star2*rho + J_bar33*J_star3*rho);
	in_element->local_M[128] = 0.166666666666667*J_bar23*J_star3*det_J*rho;
	in_element->local_M[129] = 0.166666666666667*J_star1*J_star2*det_J*rho;
	in_element->local_M[130] = 0.166666666666667*det_J*(J_star1*J_star1*rho + J_star2*J_star2*rho + J_star3*J_star3*rho);
	in_element->local_M[131] = 0.166666666666667*J_star2*J_star3*det_J*rho;
	in_element->local_M[132] = 0.166666666666667*J_bar31*J_star1*det_J*rho;
	in_element->local_M[133] = 0.166666666666667*J_bar31*J_star2*det_J*rho;
	in_element->local_M[134] = 0.166666666666667*det_J*(J_bar11*J_star1*rho + J_bar21*J_star2*rho + J_bar31*J_star3*rho);
	in_element->local_M[135] = 0.166666666666667*J_bar32*J_star1*det_J*rho;
	in_element->local_M[136] = 0.166666666666667*J_bar32*J_star2*det_J*rho;
	in_element->local_M[137] = 0.166666666666667*det_J*(J_bar12*J_star1*rho + J_bar22*J_star2*rho + J_bar32*J_star3*rho);
	in_element->local_M[138] = 0.166666666666667*J_bar33*J_star1*det_J*rho;
	in_element->local_M[139] = 0.166666666666667*J_bar33*J_star2*det_J*rho;
	in_element->local_M[140] = 0.166666666666667*det_J*(J_bar13*J_star1*rho + J_bar23*J_star2*rho + J_bar33*J_star3*rho);
	in_element->local_M[141] = 0.166666666666667*J_star1*J_star3*det_J*rho;
	in_element->local_M[142] = 0.166666666666667*J_star2*J_star3*det_J*rho;
	in_element->local_M[143] = 0.166666666666667*det_J*(J_star1*J_star1*rho + J_star2*J_star2*rho + J_star3*J_star3*rho);*/
	

#if 1
	in_element->local_M[0] = det_J*rho / 3;
	in_element->local_M[1] = 0.0;
	in_element->local_M[2] = 0.0;
	in_element->local_M[3] = det_J*rho / 4;
	in_element->local_M[4] = 0.0;
	in_element->local_M[5] = 0.0;
	in_element->local_M[6] = det_J*rho / 4;
	in_element->local_M[7] = 0.0;
	in_element->local_M[8] = 0.0;
	in_element->local_M[9] = -det_J*rho / 3;
	in_element->local_M[10] = 0.0;
	in_element->local_M[11] = 0.0;
	in_element->local_M[12] = 0.0;
	in_element->local_M[13] = det_J*rho / 3;
	in_element->local_M[14] = 0.0;
	in_element->local_M[15] = 0.0;
	in_element->local_M[16] = det_J*rho / 4;
	in_element->local_M[17] = 0.0;
	in_element->local_M[18] = 0.0;
	in_element->local_M[19] = det_J*rho / 4;
	in_element->local_M[20] = 0.0;
	in_element->local_M[21] = 0.0;
	in_element->local_M[22] = -det_J*rho / 3;
	in_element->local_M[23] = 0.0;
	in_element->local_M[24] = 0.0;
	in_element->local_M[25] = 0.0;
	in_element->local_M[26] = det_J*rho / 3;
	in_element->local_M[27] = 0.0;
	in_element->local_M[28] = 0.0;
	in_element->local_M[29] = det_J*rho / 4;
	in_element->local_M[30] = 0.0;
	in_element->local_M[31] = 0.0;
	in_element->local_M[32] = det_J*rho / 4;
	in_element->local_M[33] = 0.0;
	in_element->local_M[34] = 0.0;
	in_element->local_M[35] = -det_J*rho / 3;
	in_element->local_M[36] = det_J*rho / 4;
	in_element->local_M[37] = 0.0;
	in_element->local_M[38] = 0.0;
	in_element->local_M[39] = det_J*rho / 3;
	in_element->local_M[40] = 0.0;
	in_element->local_M[41] = 0.0;
	in_element->local_M[42] = det_J*rho / 4;
	in_element->local_M[43] = 0.0;
	in_element->local_M[44] = 0.0;
	in_element->local_M[45] = -det_J*rho / 3;
	in_element->local_M[46] = 0.0;
	in_element->local_M[47] = 0.0;
	in_element->local_M[48] = 0.0;
	in_element->local_M[49] = det_J*rho / 4;
	in_element->local_M[50] = 0.0;
	in_element->local_M[51] = 0.0;
	in_element->local_M[52] = det_J*rho / 3;
	in_element->local_M[53] = 0.0;
	in_element->local_M[54] = 0.0;
	in_element->local_M[55] = det_J*rho / 4;
	in_element->local_M[56] = 0.0;
	in_element->local_M[57] = 0.0;
	in_element->local_M[58] = -det_J*rho / 3;
	in_element->local_M[59] = 0.0;
	in_element->local_M[60] = 0.0;
	in_element->local_M[61] = 0.0;
	in_element->local_M[62] = det_J*rho / 4;
	in_element->local_M[63] = 0.0;
	in_element->local_M[64] = 0.0;
	in_element->local_M[65] = det_J*rho / 3;
	in_element->local_M[66] = 0.0;
	in_element->local_M[67] = 0.0;
	in_element->local_M[68] = det_J*rho / 4;
	in_element->local_M[69] = 0.0;
	in_element->local_M[70] = 0.0;
	in_element->local_M[71] = -det_J*rho / 3;
	in_element->local_M[72] = det_J*rho / 4;
	in_element->local_M[73] = 0.0;
	in_element->local_M[74] = 0.0;
	in_element->local_M[75] = det_J*rho / 4;
	in_element->local_M[76] = 0.0;
	in_element->local_M[77] = 0.0;
	in_element->local_M[78] = det_J*rho / 3;
	in_element->local_M[79] = 0.0;
	in_element->local_M[80] = 0.0;
	in_element->local_M[81] = -det_J*rho / 3;
	in_element->local_M[82] = 0.0;
	in_element->local_M[83] = 0.0;
	in_element->local_M[84] = 0.0;
	in_element->local_M[85] = det_J*rho / 4;
	in_element->local_M[86] = 0.0;
	in_element->local_M[87] = 0.0;
	in_element->local_M[88] = det_J*rho / 4;
	in_element->local_M[89] = 0.0;
	in_element->local_M[90] = 0.0;
	in_element->local_M[91] = det_J*rho / 3;
	in_element->local_M[92] = 0.0;
	in_element->local_M[93] = 0.0;
	in_element->local_M[94] = -det_J*rho / 3;
	in_element->local_M[95] = 0.0;
	in_element->local_M[96] = 0.0;
	in_element->local_M[97] = 0.0;
	in_element->local_M[98] = det_J*rho / 4;
	in_element->local_M[99] = 0.0;
	in_element->local_M[100] = 0.0;
	in_element->local_M[101] = det_J*rho / 4;
	in_element->local_M[102] = 0.0;
	in_element->local_M[103] = 0.0;
	in_element->local_M[104] = det_J*rho / 3;
	in_element->local_M[105] = 0.0;
	in_element->local_M[106] = 0.0;
	in_element->local_M[107] = -det_J*rho / 3;
	in_element->local_M[108] = -det_J*rho / 3;
	in_element->local_M[109] = 0.0;
	in_element->local_M[110] = 0.0;
	in_element->local_M[111] = -det_J*rho / 3;
	in_element->local_M[112] = 0.0;
	in_element->local_M[113] = 0.0;
	in_element->local_M[114] = -det_J*rho / 3;
	in_element->local_M[115] = 0.0;
	in_element->local_M[116] = 0.0;
	in_element->local_M[117] = det_J*rho / 2;
	in_element->local_M[118] = 0.0;
	in_element->local_M[119] = 0.0;
	in_element->local_M[120] = 0.0;
	in_element->local_M[121] = -det_J*rho / 3;
	in_element->local_M[122] = 0.0;
	in_element->local_M[123] = 0.0;
	in_element->local_M[124] = -det_J*rho / 3;
	in_element->local_M[125] = 0.0;
	in_element->local_M[126] = 0.0;
	in_element->local_M[127] = -det_J*rho / 3;
	in_element->local_M[128] = 0.0;
	in_element->local_M[129] = 0.0;
	in_element->local_M[130] = det_J*rho / 2;
	in_element->local_M[131] = 0.0;
	in_element->local_M[132] = 0.0;
	in_element->local_M[133] = 0.0;
	in_element->local_M[134] = -det_J*rho / 3;
	in_element->local_M[135] = 0.0;
	in_element->local_M[136] = 0.0;
	in_element->local_M[137] = -det_J*rho / 3;
	in_element->local_M[138] = 0.0;
	in_element->local_M[139] = 0.0;
	in_element->local_M[140] = -det_J*rho / 3;
	in_element->local_M[141] = 0.0;
	in_element->local_M[142] = 0.0;
	in_element->local_M[143] = det_J*rho / 2;
#else 
	in_element->local_M[0] = det_J*rho / 6;
	in_element->local_M[1] = 0.0;
	in_element->local_M[2] = 0.0;
	in_element->local_M[3] = 0.0;
	in_element->local_M[4] = 0.0;
	in_element->local_M[5] = 0.0;
	in_element->local_M[6] = 0.0;
	in_element->local_M[7] = 0.0;
	in_element->local_M[8] = 0.0;
	in_element->local_M[9] = 0.0;
	in_element->local_M[10] = 0.0;
	in_element->local_M[11] = 0.0;
	in_element->local_M[12] = 0.0;
	in_element->local_M[13] = det_J*rho / 6;
	in_element->local_M[14] = 0.0;
	in_element->local_M[15] = 0.0;
	in_element->local_M[16] = 0.0;
	in_element->local_M[17] = 0.0;
	in_element->local_M[18] = 0.0;
	in_element->local_M[19] = 0.0;
	in_element->local_M[20] = 0.0;
	in_element->local_M[21] = 0.0;
	in_element->local_M[22] = 0.0;
	in_element->local_M[23] = 0.0;
	in_element->local_M[24] = 0.0;
	in_element->local_M[25] = 0.0;
	in_element->local_M[26] = det_J*rho / 6;
	in_element->local_M[27] = 0.0;
	in_element->local_M[28] = 0.0;
	in_element->local_M[29] = 0.0;
	in_element->local_M[30] = 0.0;
	in_element->local_M[31] = 0.0;
	in_element->local_M[32] = 0.0;
	in_element->local_M[33] = 0.0;
	in_element->local_M[34] = 0.0;
	in_element->local_M[35] = 0.0;
	in_element->local_M[36] = 0.0;
	in_element->local_M[37] = 0.0;
	in_element->local_M[38] = 0.0;
	in_element->local_M[39] = det_J*rho / 6;
	in_element->local_M[40] = 0.0;
	in_element->local_M[41] = 0.0;
	in_element->local_M[42] = 0.0;
	in_element->local_M[43] = 0.0;
	in_element->local_M[44] = 0.0;
	in_element->local_M[45] = 0.0;
	in_element->local_M[46] = 0.0;
	in_element->local_M[47] = 0.0;
	in_element->local_M[48] = 0.0;
	in_element->local_M[49] = 0.0;
	in_element->local_M[50] = 0.0;
	in_element->local_M[51] = 0.0;
	in_element->local_M[52] = det_J*rho / 6;
	in_element->local_M[53] = 0.0;
	in_element->local_M[54] = 0.0;
	in_element->local_M[55] = 0.0;
	in_element->local_M[56] = 0.0;
	in_element->local_M[57] = 0.0;
	in_element->local_M[58] = 0.0;
	in_element->local_M[59] = 0.0;
	in_element->local_M[60] = 0.0;
	in_element->local_M[61] = 0.0;
	in_element->local_M[62] = 0.0;
	in_element->local_M[63] = 0.0;
	in_element->local_M[64] = 0.0;
	in_element->local_M[65] = det_J*rho / 6;
	in_element->local_M[66] = 0.0;
	in_element->local_M[67] = 0.0;
	in_element->local_M[68] = 0.0;
	in_element->local_M[69] = 0.0;
	in_element->local_M[70] = 0.0;
	in_element->local_M[71] = 0.0;
	in_element->local_M[72] = 0.0;
	in_element->local_M[73] = 0.0;
	in_element->local_M[74] = 0.0;
	in_element->local_M[75] = 0.0;
	in_element->local_M[76] = 0.0;
	in_element->local_M[77] = 0.0;
	in_element->local_M[78] = det_J*rho / 6;
	in_element->local_M[79] = 0.0;
	in_element->local_M[80] = 0.0;
	in_element->local_M[81] = 0.0;
	in_element->local_M[82] = 0.0;
	in_element->local_M[83] = 0.0;
	in_element->local_M[84] = 0.0;
	in_element->local_M[85] = 0.0;
	in_element->local_M[86] = 0.0;
	in_element->local_M[87] = 0.0;
	in_element->local_M[88] = 0.0;
	in_element->local_M[89] = 0.0;
	in_element->local_M[90] = 0.0;
	in_element->local_M[91] = det_J*rho / 6;
	in_element->local_M[92] = 0.0;
	in_element->local_M[93] = 0.0;
	in_element->local_M[94] = 0.0;
	in_element->local_M[95] = 0.0;
	in_element->local_M[96] = 0.0;
	in_element->local_M[97] = 0.0;
	in_element->local_M[98] = 0.0;
	in_element->local_M[99] = 0.0;
	in_element->local_M[100] = 0.0;
	in_element->local_M[101] = 0.0;
	in_element->local_M[102] = 0.0;
	in_element->local_M[103] = 0.0;
	in_element->local_M[104] = det_J*rho / 6;
	in_element->local_M[105] = 0.0;
	in_element->local_M[106] = 0.0;
	in_element->local_M[107] = 0.0;
	in_element->local_M[108] = 0.0;
	in_element->local_M[109] = 0.0;
	in_element->local_M[110] = 0.0;
	in_element->local_M[111] = 0.0;
	in_element->local_M[112] = 0.0;
	in_element->local_M[113] = 0.0;
	in_element->local_M[114] = 0.0;
	in_element->local_M[115] = 0.0;
	in_element->local_M[116] = 0.0;
	in_element->local_M[117] = det_J*rho / 6;
	in_element->local_M[118] = 0.0;
	in_element->local_M[119] = 0.0;
	in_element->local_M[120] = 0.0;
	in_element->local_M[121] = 0.0;
	in_element->local_M[122] = 0.0;
	in_element->local_M[123] = 0.0;
	in_element->local_M[124] = 0.0;
	in_element->local_M[125] = 0.0;
	in_element->local_M[126] = 0.0;
	in_element->local_M[127] = 0.0;
	in_element->local_M[128] = 0.0;
	in_element->local_M[129] = 0.0;
	in_element->local_M[130] = det_J*rho / 6;
	in_element->local_M[131] = 0.0;
	in_element->local_M[132] = 0.0;
	in_element->local_M[133] = 0.0;
	in_element->local_M[134] = 0.0;
	in_element->local_M[135] = 0.0;
	in_element->local_M[136] = 0.0;
	in_element->local_M[137] = 0.0;
	in_element->local_M[138] = 0.0;
	in_element->local_M[139] = 0.0;
	in_element->local_M[140] = 0.0;
	in_element->local_M[141] = 0.0;
	in_element->local_M[142] = 0.0;
	in_element->local_M[143] = det_J*rho / 6;
#endif // 0
	//for (int b = 0; b < 144; b++){
	//	in_element->local_M[b] = in_element->
	//}
	//return (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34));
	float b1 = 0.0;
	float b2 = -(9.81 *1000.0)*(det_J / 6) ;
	float b3 = 0.0;
//	b1  = b2;
	in_element->f_body[0] = b1;
	in_element->f_body[1] = b2;// b2*det_J / 2;
	in_element->f_body[2] = b3;

	in_element->f_body[3] = b1;
	in_element->f_body[4] = b2;// b2*det_J / 2;
	in_element->f_body[5] = b3;

	in_element->f_body[6] = b1;
	in_element->f_body[7] = b2;// b2*det_J / 2;
	in_element->f_body[8] = b3;

	in_element->f_body[9] =  b1;
	in_element->f_body[10] = -b2;// *det_J / 2;
	in_element->f_body[11] = b3;

	//in_element->f_body[0] =0;
	//in_element->f_body[1] =0.01;// b2*det_J / 2;
	//in_element->f_body[2] = 0;

	//in_element->f_body[3] = 0;
	//in_element->f_body[4] = 0.01;// b2*det_J / 2;
	//in_element->f_body[5] = 0;

	//in_element->f_body[6] =0;
	//in_element->f_body[7] = 0.01;// b2*det_J / 2;
	//in_element->f_body[8] = 0;

	//in_element->f_body[9] = 0;
	//in_element->f_body[10] =- 0.01;// *det_J / 2;
	//in_element->f_body[11] = 0;


}



__global__ void gpu_make_K(AFEM::element *in_vec, const AFEM::position_3D *in_pos, int numElem, int numNodes, float *K_d, float *M_d, float *f_d)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < numElem){
		find_Jacobian_localK_localM(&in_vec[x], in_pos);
		//find_localM(&in_vec[x]);
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
				atomicAdda(&(K_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), in_vec[x].local_K[c * 12 + r]);

				/*if (DOF[c] == DOF[r]){

				} */
				//atomicAdda(&(M_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), in_vec[x].local_M[c * 12 + r]);
				atomicAdda(&(M_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), in_vec[x].volume*1000.0 / 4.0);//
				//IDX2C(DOF[c], DOF[r], 3000)
				//K[IDX2C(DOF[r], DOF[c], numP*dim)] = K[IDX2C(DOF[r], DOF[c], numP*dim)] + E[k][r][c];
				
			}
			//atomicAdda(&(f_d[DOF[c]]), in_vec[x].f_body[c]);
			/*if (x == 800){
				atomicAdda(&(f_d[DOF[c]]), in_vec[x].f_body[c]);
				}*/
			//atomicAdda(&(f_d[DOF[c]]), in_vec[x].f_body[c]);
			
			atomicAdda(&(f_d[DOF[c]]), in_vec[x].f_body[c]);
			
		}
		//printf("hi");
	}

}



__global__ void gpu_make_K_new(AFEM::element *in_vec, AFEM::position_3D *in_pos, int numElem, int numNodes, float *K_d, float *M_d, float *f_d)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < numElem){
		find_Jacobian_localK_localM(&in_vec[x], in_pos);
		//find_localM(&in_vec[x]);
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
				atomicAdda(&(K_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), in_vec[x].local_K[c * 12 + r]);

				/*if (DOF[c] == DOF[r]){

				} */

				//if ((DOF[c] == 2130) && (DOF[r] == 2130)){

				//	atomicAdda(&(K_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), -1);
				//}
				//else if ((DOF[c] == 2131) && (DOF[r] == 2131)) {
				//	atomicAdda(&(K_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), -1);
				//}
				//else if ((DOF[c] == 2132) && (DOF[r] == 2132)){
				//	atomicAdda(&(K_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), -1);

				//}
				//atomicAdda(&(M_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), in_vec[x].local_M[c * 12 + r]);
				atomicAdda(&(M_d[IDX2C(DOF[c], DOF[r], 3 * (numNodes))]), in_vec[x].volume*1000.0 / 4.0);//
				//IDX2C(DOF[c], DOF[r], 3000)
				//K[IDX2C(DOF[r], DOF[c], numP*dim)] = K[IDX2C(DOF[r], DOF[c], numP*dim)] + E[k][r][c];

			}
			//atomicAdda(&(f_d[DOF[c]]), in_vec[x].f_body[c]);
			/*if (x == 800){
			atomicAdda(&(f_d[DOF[c]]), in_vec[x].f_body[c]);
			}*/
			//atomicAdda(&(f_d[DOF[c]]), in_vec[x].f_body[c]);

			atomicAdda(&(f_d[DOF[c]]), in_vec[x].f_body[c]);

		}
		//printf("hi");
	}

}


//Resets the K matrix to zero
__global__ void reset_K_GPU(float *K_d, float *M_d, int numNodes, int dim){
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < numNodes*dim*numNodes*dim){
		K_d[x] = 0;
		M_d[x] = 0;
	}
}

//reset f external vector
//Resets the K matrix to zero
__global__ void reset_f_GPU(float *f_d, int numNodes, int dim){
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < numNodes*dim){
		f_d[x] = 0;
		//M_d[x] = 0;
	}
}
__global__ void update_position_vector(AFEM::position_3D *pos_in, float *u_dot_in, float dt, int numNodes, int dim){
	int x = threadIdx.x + blockIdx.x *blockDim.x;

	if (x < numNodes){
		pos_in[x].x += dt*u_dot_in[pos_in[x].displacement_index[0]];
		pos_in[x].y += dt*u_dot_in[pos_in[x].displacement_index[1]];
		pos_in[x].z += dt*u_dot_in[pos_in[x].displacement_index[2]];
	
	}
}


__global__ void update_position_vector_static(AFEM::position_3D *pos_in,const AFEM::position_3D *pos_initial,  float *u_dot_in, float dt, int numNodes, int dim){
	int x = threadIdx.x + blockIdx.x *blockDim.x;

	if (x < numNodes){
		/*pos_in[x].x = pos_initial[x].x +u_dot_in[pos_in[x].displacement_index[0]];
		pos_in[x].y = pos_initial[x].y+u_dot_in[pos_in[x].displacement_index[1]];
		pos_in[x].z = pos_initial[x].z +u_dot_in[pos_in[x].displacement_index[2]];
*/
		pos_in[x].x = u_dot_in[pos_in[x].displacement_index[0]];
		pos_in[x].y = u_dot_in[pos_in[x].displacement_index[1]];
		pos_in[x].z =  u_dot_in[pos_in[x].displacement_index[2]];

	}
}
__global__ void update_Geo_CUDA(AFEM::element *in_vec, AFEM::position_3D *pos_in, float *x_solution, int numElem){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < numElem){
		for (int npe = 0; npe < 4; npe++){
			//dummy_node = nodesInElem[nodesinelemX(npe, offset, 4)]; // The row of the matrix we looking at will be k_th element and npe (nodes per element) 	
			/*for (int dof = 0; dof < 3; dof++){
				if (dof == 0){
				in_vec[x].position_info[npe].x += x_solution[in_vec[x].position_info[npe].displacement_index[dof]];
				}
				else if (dof == 1){
				in_vec[x].position_info[npe].y += x_solution[in_vec[x].position_info[npe].displacement_index[dof]];
				}
				else if (dof == 2){
				in_vec[x].position_info[npe].z += x_solution[in_vec[x].position_info[npe].displacement_index[dof]];
				}
				}*/

			//in_vec[npe].position_info
		}

	}


}

//Change the K_d matrix and the f matrix so that they have the necessary BC

__global__ void gpu_stationary_BC(float *K_d, float *f_d, AFEM::stationary *stat_d, int numstationary, int numnodes, int dim){
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < numstationary){



		for (int i = 0; i < 3; i++){

			for (int n = 0; n < numnodes*dim; n++){
				K_d[IDX2C(n, stat_d[x].displacement_index[i], 3 * (numnodes))] = 0.0f;
			}

			for (int n = 0; n < numnodes*dim; n++){
				K_d[IDX2C(stat_d[x].displacement_index[i], n, 3 * (numnodes))] = 0.0f;
			}
			K_d[IDX2C(stat_d[x].displacement_index[i], stat_d[x].displacement_index[i], 3 * (numnodes))] = 1.0f;
		    f_d[stat_d[x].displacement_index[i]] = 0.0f;
		}

	}

	/*if (x < numstationary){



		for (int i = 0; i < 3; i++){

		for (int n = 0; n < numnodes*dim; n++){
		K_d[IDX2C(stat_d[x].displacement_index[i], n, 3 * (numnodes))] = 0.0f;
		}
		K_d[IDX2C(stat_d[x].displacement_index[i], stat_d[x].displacement_index[i], 3 * (numnodes))] = 1.0f;
		f_d[stat_d[x].displacement_index[i]] = 0.0f;
		}

		}*/

	//if (x < numstationary){



	//	for (int i = 0; i < 3; i++){

	//		/*for (int n = 0; n < numnodes*dim; n++){
	//			K_d[IDX2C(n, stat_d[x].displacement_index[i], 3 * (numnodes))] = 0.0f;
	//		}
	//		K_d[IDX2C(stat_d[x].displacement_index[i], stat_d[x].displacement_index[i], 3 * (numnodes))] = 1.0f;*/
	//		f_d[stat_d[x].displacement_index[i]] = 0.0f;
	//	}

	//}


}


__global__ void gpu_stationary_BC_new(float *K_d, float *f_d, AFEM::stationary *stat_d, AFEM::position_3D *initial_pos,int numstationary, int numnodes, int dim){
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < numstationary){



		for (int i = 0; i < 3; i++){

			for (int n = 0; n < numnodes*dim; n++){
				K_d[IDX2C(n, stat_d[x].displacement_index[i], 3 * (numnodes))] = 0.0f;
			}

			for (int n = 0; n < numnodes*dim; n++){
				K_d[IDX2C(stat_d[x].displacement_index[i], n, 3 * (numnodes))] = 0.0f;
			}
			K_d[IDX2C(stat_d[x].displacement_index[i], stat_d[x].displacement_index[i], 3 * (numnodes))] = 1.0f;

			if (i == 0){
				f_d[stat_d[x].displacement_index[i]] = initial_pos[stat_d[x].node_number].x;
			}
			else if (i == 1){
				f_d[stat_d[x].displacement_index[i]] = initial_pos[stat_d[x].node_number].y;
			}
			else if (i == 2){
				f_d[stat_d[x].displacement_index[i]] = initial_pos[stat_d[x].node_number].z;
			}

		}

	}

	/*if (x < numstationary){



	for (int i = 0; i < 3; i++){

	for (int n = 0; n < numnodes*dim; n++){
	K_d[IDX2C(stat_d[x].displacement_index[i], n, 3 * (numnodes))] = 0.0f;
	}
	K_d[IDX2C(stat_d[x].displacement_index[i], stat_d[x].displacement_index[i], 3 * (numnodes))] = 1.0f;
	f_d[stat_d[x].displacement_index[i]] = 0.0f;
	}

	}*/

	//if (x < numstationary){



	//	for (int i = 0; i < 3; i++){

	//		/*for (int n = 0; n < numnodes*dim; n++){
	//			K_d[IDX2C(n, stat_d[x].displacement_index[i], 3 * (numnodes))] = 0.0f;
	//		}
	//		K_d[IDX2C(stat_d[x].displacement_index[i], stat_d[x].displacement_index[i], 3 * (numnodes))] = 1.0f;*/
	//		f_d[stat_d[x].displacement_index[i]] = 0.0f;
	//	}

	//}


}




__global__ void gpu_make_f(float *f_d, int numnodes, AFEM::position_3D *pos_info, int dim, int first){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < numnodes){
		if (first == 1){
			f_d[pos_info[x].displacement_index[0]] += 0.0; //x
			f_d[pos_info[x].displacement_index[1]] = -0.9 / 10.0; //y
			f_d[pos_info[x].displacement_index[2]] += 0.0; //z
		}
		else{
			f_d[pos_info[x].displacement_index[0]] += 0.0; //x
			f_d[pos_info[x].displacement_index[1]] = -0.9 / 10.0; //y
			f_d[pos_info[x].displacement_index[2]] += 0.0; //z
		}
	}
}

__global__ void find_dx(float *dx_in, AFEM::position_3D *initial_pos, AFEM::position_3D *new_pos, int numnodes){
	int x = threadIdx.x + blockIdx.x *blockDim.x;

	if (x < numnodes){

		dx_in[new_pos[x].displacement_index[0]] = new_pos[x].x - initial_pos[x].x;
		dx_in[new_pos[x].displacement_index[1]] = new_pos[x].y - initial_pos[x].y;
		dx_in[new_pos[x].displacement_index[2]] = new_pos[x].z - initial_pos[x].z;

		/*initial_pos[x].x = new_pos[x].x;
		initial_pos[x].y = new_pos[x].y;
		initial_pos[x].z = new_pos[x].z;*/

	}
}

__global__ void find_dx_new(float *dx_in, AFEM::position_3D *initial_pos, AFEM::position_3D *new_pos, int numnodes){
	int x = threadIdx.x + blockIdx.x *blockDim.x;

	if (x < numnodes){

		dx_in[new_pos[x].displacement_index[0]] = initial_pos[x].x;
		dx_in[new_pos[x].displacement_index[1]] = initial_pos[x].y;
		dx_in[new_pos[x].displacement_index[2]] = initial_pos[x].z;

		/*initial_pos[x].x = new_pos[x].x;
		initial_pos[x].y = new_pos[x].y;
		initial_pos[x].z = new_pos[x].z;*/

	}
}

//find the vector value of dt*f_ext - dt*K*(u(t)-u(0))+dt*dt K*u_dot(t)
//In the code I have set:
//a= dt*f_ext
//b= dt*K*(u(t)-u(0))
//c= dt*dt K*u_dot(t)
//so RHS = a-b+c
//And LHS: = M-dt*dt* K
__global__ void find_A_b_dynamic(float *K_in, float *dx_in, float *u_dot, float *f_ext, float *RHS, float *M_in, float *LHS, int num_nodes, float dt, float rm, float rk, int dim,AFEM::stationary *stationary, int num_station){
	int x = threadIdx.x + blockIdx.x *blockDim.x;

	if (x < num_nodes*dim){
#ifdef DYNAMIC
		float a, b, c;
		b = c = 0;
		a = f_ext[x];
		for (int i = 0; i < num_nodes*dim; i++){
			/*if (i == x){
				M_in[IDX2C(i, x, 3 * (num_nodes))] =
				}*/
			b = b + K_in[IDX2C(i, x, 3 * (num_nodes))] * dx_in[i];
			//c = c + K_in[IDX2C(i, x, 3 * (num_nodes))] * u_dot[i]; 
			//origional
			//float c1 = dt*rm*M_in[IDX2C(i, x, 3 * num_nodes)] + (dt *rk-dt*dt)*K_in[IDX2C(i, x, 3 * num_nodes)];

			c = c + M_in[IDX2C(i, x, 3 * num_nodes)] * u_dot[i];

			//Origional
			//LHS[IDX2C(i, x, 3 * (num_nodes))] = (1.0-dt*rm)*M_in[IDX2C(i, x, 3 * (num_nodes))] - (dt*rk+dt*dt)*K_in[IDX2C(i, x, 3 * (num_nodes))];
			LHS[IDX2C(i, x, 3 * (num_nodes))] = M_in[IDX2C(i, x, 3 * (num_nodes))] + (dt*dt)*K_in[IDX2C(i, x, 3 * (num_nodes))];
			/*if (i == x){

			}
			else{
			LHS[IDX2C(i, x, 3 * (num_nodes))] = (dt*dt)*K_in[IDX2C(i, x, 3 * (num_nodes))];
			}*/

		}
		a = a*dt;
		b = b*dt;


		RHS[x] = a - b + c;
		//RHS[x] = f_ext[x];  

#else
		float a,b;
		b = 0;
		//a = f_ext[x];
		for (int i = 0; i < num_nodes*dim; i++){
			/*if (i == x){
			M_in[IDX2C(i, x, 3 * (num_nodes))] =
			}*/

			//Origional
			//LHS[IDX2C(i, x, 3 * (num_nodes))] = (1.0-dt*rm)*M_in[IDX2C(i, x, 3 * (num_nodes))] - (dt*rk+dt*dt)*K_in[IDX2C(i, x, 3 * (num_nodes))];
			
			b = b + K_in[IDX2C(i, x, 3 * (num_nodes))] * dx_in[i];


			LHS[IDX2C(i, x, 3 * (num_nodes))] = K_in[IDX2C(i, x, 3 * (num_nodes))];
			/*if ((i == 2130) &(x == 2130)){
				LHS[IDX2C(i, x, 3 * (num_nodes))] = LHS[IDX2C(i, x, 3 * (num_nodes))] - 1;
				RHS[i] = RHS[i] - dx_in[i] ;
			}
			else if ((i == 2131) &(x == 2131))
			{
				LHS[IDX2C(i, x, 3 * (num_nodes))] = LHS[IDX2C(i, x, 3 * (num_nodes))] - 1;
				RHS[i] = RHS[i] - dx_in[i];
			}
			else if ((i == 2132) &(x == 2132)){
				LHS[IDX2C(i, x, 3 * (num_nodes))] = LHS[IDX2C(i, x, 3 * (num_nodes))] - 1;
				RHS[i] = RHS[i] - dx_in[i];
			}*/
			

			/*if (x == i){
				LHS[IDX2C(i, x, 3 * (num_nodes))] = LHS[IDX2C(i, x, 3 * (num_nodes))] - 1;
				RHS[i] = RHS[i] - dx_in[i];
			}*/
			/*if (i == x){
			
			}
			else{
			LHS[IDX2C(i, x, 3 * (num_nodes))] = (dt*dt)*K_in[IDX2C(i, x, 3 * (num_nodes))];
			}*/

		}
	

		RHS[x] = b;
		//RHS[x] = a ;
		//RHS[x] = f_ext[x];

	


		/*	for (int nnn = 2131; nnn < 2131 + 1; nnn++){
				if (nnn == x){
					LHS[IDX2C(nnn, nnn, 3 * (num_nodes))] = LHS[IDX2C(nnn, nnn, 3 * (num_nodes))] + alpha;


					RHS[nnn] = RHS[nnn] + alpha* (dx_in[nnn] + 0.05);
				}
			}


			for (int nnn = 2134; nnn < 2134 + 1; nnn++){
				if (nnn == x){
					LHS[IDX2C(nnn, nnn, 3 * (num_nodes))] = LHS[IDX2C(nnn, nnn, 3 * (num_nodes))] + alpha;


					RHS[nnn] = RHS[nnn] + alpha* (dx_in[nnn] + 0.05);
				}
			}*/

		
#endif // DYNAMIC

	}
	
	if (x == 0){


		float alpha = 18000000;
		//_dd_ += 1000;*/
		/*LHS[IDX2C(2130, 2130, 3 * (num_nodes))] = LHS[IDX2C(2130, 2130, 3 * (num_nodes))] + alpha;
		LHS[IDX2C(2131, 2131, 3 * (num_nodes))] = LHS[IDX2C(2131, 2131, 3 * (num_nodes))] + alpha;
		LHS[IDX2C(2132, 2132, 3 * (num_nodes))] = LHS[IDX2C(2132, 2132, 3 * (num_nodes))] + alpha;
		RHS[2130] = RHS[2130] + alpha*dx_in[2130];
		RHS[2131] = RHS[2131] + alpha*dx_in[2131];
		RHS[2132] = RHS[2132] + alpha*dx_in[2132];
		*/

		for (int nnn = 0; nnn < num_station; nnn++){
			for (int i = 0; i < 3; i++){
				int station_array = stationary[nnn].displacement_index[i];

				LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] = LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] + alpha;


				RHS[station_array] = RHS[station_array] + alpha* (dx_in[station_array]);



			}
		}
		/*int station_array ;
		
		station_array = 463 * 3 + 1;
		LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] = LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] + alpha;


		RHS[station_array] = RHS[station_array] + alpha* (dx_in[station_array] + 0.01);

		station_array = 464 *3 + 1;
		LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] = LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] + alpha;
		RHS[station_array] = RHS[station_array] + alpha* (dx_in[station_array] + 0.01 );

		station_array = 465 * 3 + 1;
		LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] = LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] + alpha;
		RHS[station_array] = RHS[station_array] + alpha* (dx_in[station_array] + 0.01 );


		station_array = 466 * 3 + 1;
		LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] = LHS[IDX2C(station_array, station_array, 3 * (num_nodes))] + alpha;
		RHS[station_array] = RHS[station_array] + alpha* (dx_in[station_array] + 0.01 );

		dd += 0.004;*/	
		//for (int lll = 0; lll < num_station; lll++){

			int dof_counter = 0;
			int node_num = 555;
			for (int dof = node_num * 3; dof < ((node_num * 3) + 3); dof++){



				LHS[IDX2C(dof, dof, 3 * (num_nodes))] = LHS[IDX2C(dof, dof, 3 * (num_nodes))] + alpha;

				if (dof_counter == 0){
					RHS[dof] = RHS[dof] + alpha* (dx_in[dof]);//sudo_force_vec[i].fx);
				}
				else if (dof_counter == 1){
					RHS[dof] = RHS[dof] + alpha* (dx_in[dof]-0.151);//sudo_force_vec[i].fy);

				}
				else if (dof_counter == 2){

					RHS[dof] = RHS[dof] + alpha* (dx_in[dof]);//sudo_force_vec[i].fz);
				}



				dof_counter++;

			}
		//}

		node_max++;

	}

}


//updates the u_dot vector, so u_dot(t+dt) = u_dot(t)+du
__global__ void update_u_dot_vector(float *u_dot_pre, float *u_dot_sln, int numNodes, int dim){
	int x = threadIdx.x + blockIdx.x *blockDim.x;

	if (x < numNodes*dim){
		u_dot_pre[x] = u_dot_sln[x];// LOL u_dot_pre[x]+
	}


}
//
//__global__ void find_A_dynamic(float *K_in, float *M_in, float *LHS, int numnodes, int dim){
//	int x = threadIdx.x + blockIdx.x *blockDim.x;
//
//	if (x < numnodes*dim){
//		for (int i = 0; i < numnodes*dim; i++){
//			LHS
//		}
//
//	}
//}


//Allocates the cpu and gpu memory, and then copy necessary data to them
void cuda_tools::allocate_copy_CUDA_geometry_data(AFEM::element *in_array_elem, AFEM::stationary *in_array_stationary, AFEM::position_3D *in_array_position, int numstationary, int num_elem, int num_nodes, int dim){
	//cpu allocation of memeory
	//K matrix
	K_h = (float*)malloc(sizeof(*K_h)*dim*num_nodes*dim*num_nodes);


	//cuda allocation of memory
	elem_array_h = in_array_elem;
	position_array_h = in_array_position;
	cudaMalloc((void**)&elem_array_d, sizeof(AFEM::element) *num_elem); //element array
	//cudaMalloc((void**)&element_array_initial_d, sizeof(AFEM::element) *num_elem); //initial orientation of the element array
	cudaMalloc((void**)&K_d, sizeof(*K_d)*dim*num_nodes*dim*num_nodes); //final global K matrix container
	cudaMalloc((void**)&M_d, sizeof(*M_d)*dim*num_nodes*dim*num_nodes); //Global mass M matrix container
	cudaMalloc((void**)&stationary_array_d, sizeof(*stationary_array_d)*numstationary); //stationary vector
	cudaMalloc((void**)&f_d, sizeof(*f_d)*dim*num_nodes); //force vector 
	cudaMalloc((void**)&position_array_d, sizeof(AFEM::position_3D)*num_nodes);// the position vector with indicy information
	cudaMalloc((void**)&position_array_initial_d, sizeof(AFEM::position_3D)*num_nodes);//initial position vector with indice 
	cudaMalloc((void**)&dx_d, sizeof(*dx_d)*num_nodes*dim); // vector for u(t)-u(0)
	cudaMalloc((void**)&Kdx_d, sizeof(*Kdx_d)*num_nodes*dim); //result vector of Kdx_d
	cudaMalloc((void**)&u_dot_d, sizeof(*u_dot_d)*num_nodes*dim); //velocity of the displacement field
	cudaMalloc((void**)&RHS, sizeof(*RHS)*num_nodes*dim); // allocating the vector for the RHS
	cudaMalloc((void**)&LHS, sizeof(*LHS)*num_nodes*dim*num_nodes*dim);

	//cuda copy of memory from host to device
	cudaMemcpy(elem_array_d, in_array_elem, sizeof(AFEM::element) *num_elem, cudaMemcpyHostToDevice);
	//cudaMemcpy(element_array_initial_d, in_array_elem, sizeof(AFEM::element) *num_elem, cudaMemcpyHostToDevice);
	cudaMemcpy(stationary_array_d, in_array_stationary, sizeof(AFEM::stationary)*numstationary, cudaMemcpyHostToDevice);
	cudaMemcpy(position_array_d, in_array_position, sizeof(AFEM::position_3D)*num_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(position_array_initial_d, in_array_position, sizeof(AFEM::position_3D)*num_nodes, cudaMemcpyHostToDevice);
	//initialize the global matricies to zero
	cudaMemset(K_d, 0.0, sizeof(*K_d)*dim*num_nodes*dim*num_nodes);
	cudaMemset(M_d, 0.0, sizeof(*M_d)*dim*num_nodes*dim*num_nodes);
	cudaMemset(LHS, 0.0, sizeof(*LHS)*dim*num_nodes*dim*num_nodes);


	//initialize the force to be 0
	cudaMemset(f_d, 0.0, sizeof(*f_d)*dim*num_nodes);

	//initialize x(t)-x(0) to be 0
	cudaMemset(dx_d, 0.0, sizeof(*dx_d)*dim*num_nodes);

	//initialize u_dot to be 0
	cudaMemset(u_dot_d, 0.0, sizeof(*u_dot_d)*dim*num_nodes);




	//Allocating num nodes and num elem information into the class
	Nnodes = num_nodes;
	Nelems = num_elem;
	Nstationary = numstationary;
}


void cuda_tools::copy_data_from_cuda(AFEM::element *elem_array_ptr, AFEM::position_3D *pos_array_ptr){
	cudaMemcpy(elem_array_h, elem_array_d, sizeof(AFEM::element) *Nelems, cudaMemcpyDeviceToHost);
	cudaMemcpy(position_array_h, position_array_d, sizeof(AFEM::position_3D) *Nnodes, cudaMemcpyDeviceToHost);

	//std::cout << elem_array_h[0].position_info[0].x << std::endl;

	elem_array_ptr = elem_array_h;
	pos_array_ptr = position_array_h;

}

//Host wrapper to call gpu_make_K
void cuda_tools::make_K(int num_elem, int num_nodes){
	int blocks, threads;
	if (num_elem <= 256){
		blocks = 1;
		threads = num_elem;
	}
	else {
		blocks = (num_elem + 256) / 256;
		threads = 256;
	}

#ifdef DYNAMIC
	gpu_make_K << <blocks, threads >> > (elem_array_d, position_array_d, num_elem, num_nodes, K_d, M_d, f_d);
#else
	gpu_make_K << <blocks, threads >> > (elem_array_d, position_array_initial_d, num_elem, num_nodes, K_d, M_d, f_d);
#endif
	//cudaMemset(K_d, 0, sizeof(*K_d)*dim*num_nodes*dim*num_nodes); //initialize the vector K_d to zero

	//std::cout << std::endl;
	//

}
int first = 1;

void cuda_tools::make_f(int num_nodes, int dim){
	int blocks, threads;
	if (num_nodes <= 256){
		blocks = 1;
		threads = num_nodes;
	}
	else {
		blocks = (num_nodes + 256) / 256;
		threads = 256;
	}
	if (first){

		gpu_make_f << <blocks, threads >> >(f_d, num_nodes, position_array_d, dim, first);
		first = 0;
	}
	else {
		gpu_make_f << <blocks, threads >> >(f_d, num_nodes, position_array_d, dim, 0);
	}


}

void cuda_tools::stationary_BC(int num_elem, int num_nodes, int num_stationary, int dim){
	int blocks, threads;
	if (num_stationary <= 256){
		blocks = 1;
		threads = num_stationary;
	}
	else {
		blocks = (num_stationary + 256) / 256;
		threads = 256;
	}
	//gpu_stationary_BC << <blocks, threads >> >(LHS, RHS, stationary_array_d, num_stationary, num_nodes, dim);

#ifdef DYNAMIC
	gpu_stationary_BC << <blocks, threads >> >(LHS, RHS, stationary_array_d, num_stationary, num_nodes, dim);

#else
	gpu_stationary_BC_new << <blocks, threads >> >(LHS, RHS, stationary_array_d, position_array_initial_d,num_stationary, num_nodes, dim);
#endif
}


void cuda_tools::stationary_BC_f(float *zero_vec){
	int blocks, threads;
	if (Nstationary <= 256){
		blocks = 1;
		threads = Nstationary;
	}
	else {
		blocks = (Nstationary + 256) / 256;
		threads = 256;
	}
	gpu_stationary_BC << <blocks, threads >> >(LHS, zero_vec, stationary_array_d, Nstationary, Nnodes, 3);

	//stationary_BC( Nelems, Nnodes, Nstationary, 3);

}
void cuda_tools::reset_K(int num_elem, int num_nodes){
	int blocks, threads;
	int total_size = Nnodes*Nnodes * 9;
	if (total_size <= 256){
		blocks = 1;
		threads = total_size;
	}
	else {
		blocks = (total_size + 256) / 256;
		threads = 256;
	}

	int blocks2, threads2;
	int total_size2 = Nnodes * 3;
	if (total_size2 <= 256){
		blocks2 = 1;
		threads2 = total_size2;
	}
	else {
		blocks2 = (total_size2 + 256) / 256;
		threads2 = 256;
	}


	reset_K_GPU << <blocks, threads >> >(K_d, M_d, num_nodes, 3);
	reset_f_GPU << <blocks2, threads2 >> >(f_d, num_nodes, 3);
}


void cuda_tools::update_geometry(float *u_dot_sln){
	int blocks_nodes2, threads_nodes2;
	int nnodesdim = Nnodes * 3;
	if (nnodesdim <= 256){
		blocks_nodes2 = 1;
		threads_nodes2 = nnodesdim;
	}
	else {
		blocks_nodes2 = (nnodesdim + 256) / 256;
		threads_nodes2 = 256;
	}
	int blocks_nodes, threads_nodes;
	if (Nnodes <= 256){
		blocks_nodes = 1;
		threads_nodes = Nnodes;
	}
	else {
		blocks_nodes = (Nnodes + 256) / 256;
		threads_nodes = 256;
	}
























	/*float *h_y = (float *)malloc(Ncols* sizeof(float));
	cudaMemcpy(h_y, u_dot_sln, Ncols* sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 1; i < 12; i++){
	std::cout << h_y[i] << " ";
	}
	free(h_y);*/
	//std::cout << std::endl;
	//stationary_BC_f(u_dot_sln);
#ifdef DYNAMIC
	update_position_vector << <blocks_nodes, threads_nodes >> >(position_array_d, u_dot_sln, dt, Nnodes, 3);
	update_u_dot_vector << <blocks_nodes2, threads_nodes2 >> >(u_dot_d, u_dot_sln, Nnodes, 3);
#else

	update_position_vector_static << <blocks_nodes, threads_nodes >> >(position_array_d, position_array_initial_d,u_dot_sln, dt, Nnodes, 3);
	//update_u_dot_vector << <blocks_nodes2, threads_nodes2 >> >(u_dot_d, u_dot_sln, Nnodes, 3);
#endif



	



	//update_Geo_CUDA << <blocks_element, threads_element >> >(elem_array_d, position_array_d,u_soln, Nelems);
}
void cuda_tools::dynamic(){
	int blocks_nodesdim, threads_nodesdim;
	if (Nnodes * 3 <= 256){
		blocks_nodesdim = 1;
		threads_nodesdim = Nnodes * 3;
	}
	else {
		blocks_nodesdim = (Nnodes * 3 + 256) / 256;
		threads_nodesdim = 256;
	}


	int blocks_nodes, threads_nodes;
	if (Nnodes <= 256){
		blocks_nodes = 1;
		threads_nodes = Nnodes;
	}
	else {
		blocks_nodes = (Nnodes + 256) / 256;
		threads_nodes = 256;
	}


	//float *K_in, float *dx_in, float *u_dot, float *f_ext, float *RHS
#ifdef DYNAMIC
	find_dx << <blocks_nodes, threads_nodes >> >(dx_d, position_array_initial_d, position_array_d, Nnodes);

#else 
	find_dx_new << <blocks_nodes, threads_nodes >> >(dx_d, position_array_initial_d, position_array_d, Nnodes);

#endif
	/*float *sln_ptr = (float *)malloc(Ncols*sizeof(float));
	cudaMemcpy(sln_ptr, f_d, Ncols*sizeof(float), cudaMemcpyDeviceToHost);


	std::ofstream output;
	output.open("SLN.txt");

	for (int i = 0; i < Ncols; i++){

	output << sln_ptr[i];

	std::cout << std::endl;
	output << std::endl;
	}

	output.close();
	free(sln_ptr);*/

	//stationary_BC_f(Nelems, Nnodes, Nstationary, 3);
	//stationary_BC_f(f_d);
	find_A_b_dynamic << <blocks_nodesdim, threads_nodesdim >> >(K_d, dx_d, u_dot_d, f_d, RHS, M_d, LHS, Nnodes, dt, 1.02, 0.002, 3, stationary_array_d, Nstationary);

	
	//float *rhs_output = (float *)malloc(Ncols* Ncols*sizeof(float));
	
	//cudaMemcpy(rhs_output, LHS, Ncols* Ncols*sizeof(float), cudaMemcpyDeviceToHost);

	//std::ofstream output_RHS;
	//output_RHS.open("LHS.txt");

	//for (int i = 0; i < Ncols; i++){

	//	//std::cout << h_y[IDX2C(j, i, Ncols)] << " ";
	//	for (int j = 0; j < Ncols; j++){
	//		output_RHS << rhs_output[IDX2C(j, i, Ncols)] << " ";
	//	}
	//	output_RHS << std::endl;


	//	//output_RHS << rhs_output[i] << std::endl;

	//	//std::cout << std::endl;
	//	//output << std::endl;
	//}

	//output_RHS.close();


	//update_geometry(d_y);
	//stationary_BC(Nelems, Nnodes, Nstationary, 3);



}



void cuda_tools::set_RHS_LHS(){


	/*LHS = M_d;
	RHS = Kdx_d;*/
}
cuda_tools::cuda_tools(){
	std::cout << "CUDA solver initialized " << std::endl;
}
cuda_tools::~cuda_tools(){
	free(K_h);
	cudaFree(K_d);

}




























#if 0

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
	for (i = 0; i < n; i++)
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






#endif // 0
