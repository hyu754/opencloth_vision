#include "AFEM_geometry.hpp"
#include "cuda.h"


#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>


#include <cuda_runtime.h> 
#include "cublas_v2.h" 


#include <cusolverDn.h>
#include <cusparse_v2.h>

#ifndef AFEM_CUDA_H
#define AFEM_CUDA_H



//__global__ void make_K_cuda3d(double *E_vector, int *nodesInElem_device, double *x_vector, double *y_vector, double *z_vector, int *displaceInElem_device, float *d_A_dense, int *numnodes);//3D
//__global__ void make_K_cuda2d(double *K, int *nodesInElem, double *x_vector, double *y_vector, int *displaceInElem_device, float *d_A_dense, int numnodes, double thickness,double young_E,double nu,double alpha,double beta1,double beta2, double rho, double dt,double c_xi,int numE);//2D
//__global__ void make_global_K(void); 
class cuda_tools{
	//Device memory pointer for the element array
	AFEM::element *elem_array_d;

	//Host memory pointer for the element array
	AFEM::element *elem_array_h;
	
	//Host and device memory for stationary vector
	int *stationary_array_d;
	int *stationary_array_h;

	//Device memory pointer for the global K matrix
	//This matrix will be numnodes*3*numNodes*3 = 9numNodes^2 
	float *K_d;


	//Corresponding host memory pointers
	float *K_h;

	//for cuda solver

	//cuda allocations
	//----------------------------------------------------------------------------------
	int Nrows;                        // --- Number of rows
	int Ncols;                        // --- Number of columns
	int Nelems,Nnodes;
	int N;
	double duration_K;
	cusparseHandle_t handle;
	cusparseMatDescr_t descrA;
	cusparseMatDescr_t      descr_L = 0;
	float *h_A_dense;
	double *h_M_dense;
	
	float *d_A_dense;
	double *d_A_dense_double; 
	double *h_A_dense_double;

                  // --- Leading dimension of dense matrix
	int *d_nnzPerVector; 
	int *h_nnzPerVector;
	int nnz;
	int lda;

		//Memory used in cholesky factorization
	csric02Info_t info_A = 0; 
	csrsv2Info_t  info_L = 0;  
	csrsv2Info_t  info_Lt = 0; 

	float *d_A;
	int *d_A_RowIndices;
	int *d_A_ColIndices;
	
public:
	cuda_tools();
	~cuda_tools();
	//Allocate data
	
	//allocate_geometry_data allocate device memory for  geometry structure
	//Input:	in_elment_vector = An array of the elements structures
	//			numElem = number of elements	
	//			numNodes = number of nodes
	//			dim = dimension
	void allocate_copy_CUDA_geometry_data(AFEM::element *, int *in_array_stationary, int numstationary,int num_elem, int num_nodes, int dim);

	//Copy the data from the device memory to host
	void copy_data_from_cuda(AFEM::element *elem_array_ptr);

	
	//A wrapper function that makes the K matrix on the GPU
	void make_K(int num_elem,int num_nodes);


	//A wrapper function that resets the value of K (device) when for the next simulation
	void reset_K(int num_elem,int num_nodes);

	void stationary_BC(void);

	
	//place holder for host -> device
	void host_to_device();



	//QR solver
	int QR_solver(void);

	//before calling cholesky we need to initialize the gpu variables, only once needed.
	void initialize_cholesky_variables(int numnodes,int numelem,int dim);
	void cholesky();

	//after solving for cholesky, this function will update the geometry
	void update_geometry( float *u_sln);
};


#endif //AFEM_CUDA_H
