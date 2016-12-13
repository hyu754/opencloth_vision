#include "AFEM_geometry.hpp"

#ifndef AFEM_CUDA_H
#define AFEM_CUDA_H



//__global__ void make_K_cuda3d(double *E_vector, int *nodesInElem_device, double *x_vector, double *y_vector, double *z_vector, int *displaceInElem_device, float *d_A_dense, int *numnodes);//3D
//__global__ void make_K_cuda2d(double *K, int *nodesInElem, double *x_vector, double *y_vector, int *displaceInElem_device, float *d_A_dense, int numnodes, double thickness,double young_E,double nu,double alpha,double beta1,double beta2, double rho, double dt,double c_xi,int numE);//2D
//__global__ void make_global_K(void); 
class cuda_tools{
	AFEM::element *elem_array_d;
	
public:

	//Allocate data
	
	//allocate_geometry_data allocate device memory for  geometry structure
	//Input:	in_elment_vector = An array of the elements structures
	//			numElem = number of elements		
	void allocate_copy_CUDA_geometry_data(AFEM::element *, int);



	void make_K(int num_elem);





	void hello();

	//place holder for host -> device
	void host_to_device();
};


#endif //AFEM_CUDA_H
