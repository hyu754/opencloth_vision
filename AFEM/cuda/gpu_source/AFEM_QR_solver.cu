

#include <stdio.h>
#include <math.h>
#include <iostream>
#include "AFEM_cuda.cuh"
#include "Utilities.cuh"

#define IDX2C(i,j,ld) (((j)*(ld))+( i )) 
void printMatrix(int m, int n, const double*A, int lda, const char* name) { for (int row = 0; row < m; row++){ for (int col = 0; col < n; col++){ double Areg = A[row + col*lda]; printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg); } } }

void cuda_tools::initialize_cholesky_variables(int numNodes, int numElem, int dim){
	Nrows = numNodes*dim;                        // --- Number of rows
	Ncols = numNodes*dim;                        // --- Number of columns
	N = Nrows;
	cusparseSafeCall(cusparseCreate(&handle));

	//h_A_dense = (float*)malloc(Nrows*Ncols*sizeof(*h_A_dense));
	cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
	nnz = 0;                                // --- Number of nonzero elements in dense matrix
	lda = Nrows;                      // --- Leading dimension of dense matrix
	gpuErrchk(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));
	h_nnzPerVector = (int *)malloc(Nrows * sizeof(*h_nnzPerVector));


	//device side dense matrix
	gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));



	//cudaMemcpy(&numNodes,dev_numNodes , sizeof(dev_numNodes), cudaMemcpyDeviceToHost);


	cusparseSafeCall(cusparseCreateMatDescr(&descr_L));
	cusparseSafeCall(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE));
	cusparseSafeCall(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
	cusparseSafeCall(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));

	//emeory in cholesky
	cusparseSafeCall(cusparseCreateCsric02Info(&info_A));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_L));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_Lt));
}

void cuda_tools::cholesky()
{
	
#if 0
	std::ofstream writenodes("global_K.txt");

	for (int j = 0; j < N; j++){
		for (int i = 0; i < N; i++){
			writenodes << h_A_dense[IDX2C(j, i, N)] << " ";
		}
		writenodes << std::endl;
	}

	writenodes.close();
#endif // 0


	// --- Create device array and copy host array to it
	/*for (int j = 0; j < 20; j++){
	for (int i = 0; i < 20; i++){
	std::cout << h_A_dense[IDX2C(j, i, N)] << std::endl;
	}
	std::cout<<std::endl;
	}*/



	// --- Descriptor for sparse matrix A





	// --- Device side number of nonzero elements per row

	cusparseSafeCall(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, LHS, lda, d_nnzPerVector, &nnz));
	// --- Host side number of nonzero elements per row


	gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost));

	/*printf("Number of nonzero elements in dense matrix = %i\n\n", nnz);
	for (int i = 0; i < 10; ++i) printf("Number of nonzero elements in row %i = %i \n", i, h_nnzPerVector[i]);
	printf("\n");*/

	// --- Device side dense matrix
	gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
	gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));

	
	cusparseSafeCall(cusparseSdense2csr(handle, Nrows, Ncols, descrA, LHS, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));
	// --- Host side dense matrix

	float *h_A = (float *)malloc(nnz * sizeof(*h_A));
	int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
	int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnz*sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));



	
//	std::cout << nnz << std::endl;

	/*printf("\nOriginal matrix in CSR format\n\n");
	for (int i = 0; i < 10; ++i) printf("A[%i] = %.0f ", i, h_A[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (10 + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

	for (int i = 0; i < 10; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);
	*/
	// --- Allocating and defining dense host and device data vectors

	//float *h_x = (float *)malloc(Nrows * sizeof(float));
	///*h_x[0] = 100.0;  h_x[1] = 200.0; h_x[2] = 400.0; h_x[3] = 500.0;*/
	//for (int i = 0; i < N; i++){
	//	h_x[i] = 0.00001;
	//}


	float *d_x;        gpuErrchk(cudaMalloc(&d_x, Nrows * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_x, RHS, Nrows * sizeof(float), cudaMemcpyHostToDevice));



	/******************************************/
	/* STEP 1: CREATE DESCRIPTORS FOR L AND U */
	/******************************************/




	/********************************************************************************************************/
	/* STEP 2: QUERY HOW MUCH MEMORY USED IN CHOLESKY FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/********************************************************************************************************/


	int pBufferSize_M, pBufferSize_L, pBufferSize_Lt;
	cusparseSafeCall(cusparseScsric02_bufferSize(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, &pBufferSize_M));
	cusparseSafeCall(cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, &pBufferSize_L));
	cusparseSafeCall(cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, &pBufferSize_Lt));

	int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));
	void *pBuffer = 0;  gpuErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));


	/******************************************************************************************************/
	/* STEP 3: ANALYZE THE THREE PROBLEMS: CHOLESKY FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/******************************************************************************************************/
	int structural_zero;

	cusparseSafeCall(cusparseScsric02_analysis(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));

	cusparseStatus_t status = cusparseXcsric02_zeroPivot(handle, info_A, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("A(%d,%d) is missing\n", structural_zero, structural_zero); }

	cusparseSafeCall(cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
	cusparseSafeCall(cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

	/*************************************/
	/* STEP 4: FACTORIZATION: A = L * L' */
	/*************************************/
	int numerical_zero;

	cusparseSafeCall(cusparseScsric02(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
	status = cusparseXcsric02_zeroPivot(handle, info_A, &numerical_zero);
	/*if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero); }
	*/

	gpuErrchk(cudaMemcpy(h_A, d_A, nnz * sizeof(float), cudaMemcpyDeviceToHost));
	/*printf("\nNon-zero elements in Cholesky matrix\n\n");
	for (int k = 0; k<10; k++) printf("%f\n", h_A[k]);*/


	cusparseSafeCall(cusparseScsr2dense(handle, Nrows, Ncols, descrA, d_A, d_A_RowIndices, d_A_ColIndices, LHS, Nrows));


	/*printf("\nCholesky matrix\n\n");
	for (int i = 0; i < 10; i++) {
	std::cout << "[ ";
	for (int j = 0; j < 10; j++)
	std::cout << h_A_dense[i * Ncols + j] << " ";
	std::cout << "]\n";
	}*/

	/*********************/
	/* STEP 5: L * z = x */
	/*********************/
	// --- Allocating the intermediate result vector
	float *d_z;        gpuErrchk(cudaMalloc(&d_z, N * sizeof(float)));

	const float alpha = 1.;
	cusparseSafeCall(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, d_x, d_z, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));

	/**********************/
	/* STEP 5: L' * y = z */
	/**********************/
	// --- Allocating the host and device side result vector
	float *h_y = (float *)malloc(Ncols * sizeof(float));
	float *d_y;        gpuErrchk(cudaMalloc(&d_y, Ncols * sizeof(float)));

	cusparseSafeCall(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, d_z, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

	//cudaMemcpy(h_x, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
	/*for (int k = 0; k<20; k++) printf("dx[%i] = %f\n", k, h_x[k]);
	for (int k = 0; k<20; k++) printf("xs[%i] = %f\n", k, x[k]);*/
	
	cudaMemcpy(h_y, LHS, Ncols * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			std::cout << h_y[IDX2C(j,i,Ncols)] << " ";
		}
		std::cout << std::endl;
	}
	
	std::cout << std::endl;
	
	update_geometry(d_y);


	cudaFree(d_A);
	cudaFree(d_A_ColIndices);
	cudaFree(pBuffer);

	cudaFree(d_z);
	cudaFree(d_y);
	free(h_y);
	//free(h_x);


#if 0

	free(h_A);
	free(h_A_RowIndices);
	free(h_A_ColIndices);
	free(h_x);
	free(h_y);
	cudaFree(d_x);
	cudaFree(pBuffer);
	cudaFree(d_z);
	cudaFree(d_y);


	for (int i = 0; i < numNodes; i++) {
		x[i] = x[i] + h_x[i * dim];
		y[i] = y[i] + h_x[i * dim + 1];
		if (dim == 3){
			z[i] = z[i] + h_x[i * dim + 2];
		}

	}

	
	duration_K = (std::clock() - start_K) / (double)CLOCKS_PER_SEC;
	//std::cout << " change status : " << changeNode << std::endl;

	//std::cout << "FPS time: " <<1/duration_K << std::endl;

	//std::cout << "Duration: " << duration_K << std::endl;
	return 0;
#endif // 0

}
