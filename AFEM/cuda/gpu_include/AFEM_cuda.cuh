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
	
	//device pointer for stationary vector
	AFEM::stationary *stationary_array_d;



	//Device and host pointer for position vector
	AFEM::position_3D *position_array_d;
	AFEM::position_3D *position_array_h;
	//Device pointer for the initial position vector
	AFEM::position_3D *position_array_initial_d;
	//int *stationary_array_h;

	//Device and host memory pointer for the global K matrix
	//This matrix will be numnodes*3*numNodes*3 = 9numNodes^2 
	float *K_d,*K_h;

	//Device and host memory pointer for global M matrix
	float *M_d, *M_h;

	

	//Device pointer for f vector
	float *f_d;

	//Device pointer for the difference in the nodal positions ie. (x(t)-x(0))
	float *dx_d;

	//Device pointer for the matrix multiplication result K*dx
	float *Kdx_d;

	//Device pointer velocity of displacements field
	float *u_dot_d;

	//for cuda solver
	//RHS and LHS pointers for cholesky solver
	float *RHS, *LHS;


	//dt for dynamic
	float dt = 1.0;
	//float dt = 1.0;
	//cuda allocations
	//----------------------------------------------------------------------------------
	int Nrows;                        // --- Number of rows
	int Ncols;                        // --- Number of columns
	int Nelems,Nnodes,Nstationary;
	//int N;
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

	//for CG
	//Variables to use
	int M = 0, N = 0;// nz = 0, *I = NULL, *J = NULL;
	float *val = NULL;
	const float tol = 1e-7f;
	const int max_iter = 1820;
	float *x;
	float *rhs;
	float a, b, na, r0, r1;
	int *d_col, *d_row;
	float *d_val, *d_x, dot;
	float *d_r, *d_p, *d_Ax;
	int k;
	float alpha, beta, alpham1;

	cudaDeviceProp deviceProp;
	
public:
	cuda_tools();
	~cuda_tools();
	//Allocate data
	
	//allocate_geometry_data allocate device memory for  geometry structure
	//Input:	in_elment_vector = An array of the elements structures
	//			numElem = number of elements	
	//			numNodes = number of nodes
	//			dim = dimension
	void allocate_copy_CUDA_geometry_data(AFEM::element *, AFEM::stationary *in_array_stationary, AFEM::position_3D *in_array_position, int numstationary, int num_elem, int num_nodes, int dim);

	//Copy the data from the device memory to host
	void copy_data_from_cuda(AFEM::element *elem_array_ptr,AFEM::position_3D *pos_array_ptr);

	
	//A wrapper function that makes the K matrix on the GPU
	void make_K(int num_elem,int num_nodes);

	
	//A wrapper function that resets the value of K (device) when for the next simulation
	void reset_K(int num_elem,int num_nodes);

	//A wrapper function that modifies K and f w.r.t the BC
	void stationary_BC(int num_elem, int num_nodes,int num_stationary,int dim);

	//A wrapper function to have zero in the f vector corresponding to bc
	//A wrapper function that modifies K and f w.r.t the BC
	void stationary_BC_f(float *zero_u);

	//Make f vector
	void make_f(int num_nodes, int dim);


	//place holder for host -> device
	void host_to_device();





	//before calling cholesky we need to initialize the gpu variables, only once needed.
	void initialize_cholesky_variables(int numnodes,int numelem,int dim);

	//Sets RHS and LHS memory pointers for cholesky solver
	void set_RHS_LHS();


	//Runs the cholesky solver.
	void cholesky();

	//conjugate gradient
	void cg();

	//conjugate gradient with precondition
	void cg_precond();
	//Dynamic
	void dynamic();

	//after solving for cholesky, this function will update the geometry
	void update_geometry( float *u_sln);
};






#endif //AFEM_CUDA_H
