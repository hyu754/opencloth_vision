#include "AFEM_geometry.hpp"
#include "cuda.h"

#ifndef AFEM_CUDA_H
#define AFEM_CUDA_H



//__global__ void make_K_cuda3d(double *E_vector, int *nodesInElem_device, double *x_vector, double *y_vector, double *z_vector, int *displaceInElem_device, float *d_A_dense, int *numnodes);//3D
//__global__ void make_K_cuda2d(double *K, int *nodesInElem, double *x_vector, double *y_vector, int *displaceInElem_device, float *d_A_dense, int numnodes, double thickness,double young_E,double nu,double alpha,double beta1,double beta2, double rho, double dt,double c_xi,int numE);//2D
//__global__ void make_global_K(void); 
class cuda_tools{
	//Device memory pointer for the element array
	AFEM::element *elem_array_d;

	//Device memory pointer for the global K matrix
	//This matrix will be numnodes*3*numNodes*3 = 9numNodes^2 
	double *K_d;


	//Corresponding host memory pointers
	double *K_h;
	
public:
	~cuda_tools();
	//Allocate data
	
	//allocate_geometry_data allocate device memory for  geometry structure
	//Input:	in_elment_vector = An array of the elements structures
	//			numElem = number of elements	
	//			numNodes = number of nodes
	//			dim = dimension
	void allocate_copy_CUDA_geometry_data(AFEM::element *,int,int,int);

	//Copy the data from the device memory to host
	void copy_data_from_cuda(int numNodes, int dim);



	void make_K(int num_elem,int num_nodes);





	
	//place holder for host -> device
	void host_to_device();
};


#endif //AFEM_CUDA_H
