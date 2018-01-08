/**
 * Sum two square matrix (A = B + C)
 * Exercise 3.1 of programming massively parallel processors book
 * Solution provided with matrix view as array
 * @author Niccolò Bellaccini
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h> //rand
#include <time.h> //rand
#include <math.h> //ceil

#include <iostream> //std::cerr

#define BLOCK_WIDTH 64

void initData(float* M, int nRows){ //Remeber: matrix square

	for (int i=0; i<nRows; i++){
		for(int j=0; j<nRows; j++){
			M[nRows * i + j] = (float) (rand() & 0xFF) / 10.0f;
		}
	}
}

void displayData(float *M, int nRows){

	for (int i=0; i<nRows; i++){
		printf("\n");
		for(int j=0; j<nRows; j++){
			printf("%.1f\t", M[nRows * i + j]);
		}
	}
}

/**
 * function-like macro
 * __LINE__ = contains the line number of the currently compiled line of code
 * __FILE__ = string that contains the name of the source file being compiled
 * # operator = turns the argument it precedes into a quoted string
 * Reference: [C the complete reference]
 * check with > nvcc -E
 */
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line,
		const char *statement, cudaError_t err){
	if(err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) <<
			"(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}


//Kernel function (point b of exercise)
__global__ void matrixAddKernel(float *A, float *B, float *C, int nRows){
	int size = nRows * nRows; //Remember: square matrices
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<size)
		A[i] = B[i] + C[i];
}

//Kernel function (point c of exercise)
__global__ void matrixPerRowsAddKernel(float *A, float *B, float *C, int nRows){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<nRows){
		for (int j=0; j<nRows; j++){
			A[i * nRows + j] = B[i * nRows + j] + C[i * nRows + j];
		}
	}
}

//Kernel function (point d of exercise)
__global__ void matrixPerColumnsAddKernel(float *A, float *B, float *C, int nColumns){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i<nColumns){
		for (int j=0; j<nColumns; j++){
			A[i + j * nColumns] = B[i + j * nColumns] + C[i + j * nColumns];
		}
	}
}

/**
 * Stub function used to compute matrices sum.
 * (function used to launch the kernel and to allocate device mem, ...)
 */
void matrixAdd(float* A, float *B, float *C, int nRows){

	size_t size = nRows * nRows * sizeof(float);
	float * d_A;
	float * d_B;
	float * d_C;

	//Allocate device memory for matrices
	CUDA_CHECK_RETURN(cudaMalloc((void **) &d_B, size));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &d_C, size));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &d_A, size));

	//Copy B and C to device memory
	CUDA_CHECK_RETURN(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice));

	//Kernel launch code
	//assume block of 64 threads
	matrixAddKernel <<< ceil((double)(nRows*nRows)/BLOCK_WIDTH), BLOCK_WIDTH>>>(d_A, d_B, d_C, nRows);

	//Two other possible kernel functions

	//matrixPerRowsAddKernel<<< ceil((double)nRows/BLOCK_WIDTH) ,BLOCK_WIDTH>>>(d_A, d_B, d_C, nRows);
	//matrixPerColumnsAddKernel<<< ceil((double)nRows/BLOCK_WIDTH) ,BLOCK_WIDTH>>>(d_A, d_B, d_C, nRows);

	//Copy A from the device memory
	CUDA_CHECK_RETURN(cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost));

	//Free device matrices
	cudaFree(d_C);
	cudaFree(d_B);
	cudaFree(d_A);
}

int main(int argc, char** argv){

	//Initialize random seed
	//@see http://www.cplusplus.com/reference/cstdlib/srand/
	//@see https://stackoverflow.com/questions/20158841/my-random-number-generator-function-generates-the-same-number
    srand(time(NULL));

	int numRows;
	printf("\nInsert the number of rows (equivalently columns): ");
	scanf("%d", &numRows);

	int numColumns = numRows; //Square matrix
	int nElem = numRows * numColumns;
	float * B = (float *) malloc(nElem * sizeof(float));
	float * C = (float *) malloc(nElem * sizeof(float));

	float * A = (float *) malloc(nElem * sizeof(float));

	//Initialize B and C matrices
	initData(B, numRows);
	initData(C, numRows);

	//Display B and C matrices
	printf("\n\tMatrice B\n");
	displayData(B, numRows);
	printf("\n\n\tMatrice C\n");
	displayData(C, numRows);

	//matrices sum
	matrixAdd(A, B, C, numRows);

	//Display A matrix
	printf("\n\n\tMatrice A\n");
	displayData(A, numRows);
}
