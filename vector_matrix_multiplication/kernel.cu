#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <time.h>
#include <cuda_runtime_api.h>
/*Realization of the multiplication between a vector and a matrix, on the GPU.

It realizes multiplication between a vector of 4 element and a matrix (4x4),
so as result is expected a vector of 4 element.


We initialized the input array on the host (CPU) and trasferred to device (GPU).
Then, CPU launches the kernel which is elaborated on GPU.
Finally, the results is trasferred from GPU to CPU and printed out*/



__global__ void vector_matrix_mult(float *d_vec, float *d_mat, float *d_out, const int N, const int M)
{

int tid = threadIdx.x + blockIdx.x*blockDim.x;
float sum = 0;
if (tid < M) {
for (int i = 0; i < N; i++)
sum += d_vec[i] * d_mat[(i*M) + tid];
d_out[tid] = sum;
}
}

int main(int argc, char ** argv) {



	float elapsed = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int N = 4;
	const int M = N;
	const int ARRAY_SIZE = N;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	const int MATRIX_SIZE = ARRAY_SIZE*ARRAY_SIZE;
	const int MATRIX_BYTES = MATRIX_SIZE * sizeof(float);

	// generate the input vector and input matrix on the host
	float h_vec[ARRAY_SIZE] = { 2, 1, 1, 1 };
	float h_mat[ARRAY_SIZE][ARRAY_SIZE] = { 2, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };

	
	float h_out[ARRAY_SIZE];

	//declare GPU memory pointers
	float * d_vec;
	float * d_mat;
	float * d_out;

	//allocate GPU memory
	cudaMalloc((void**)&d_vec, ARRAY_BYTES);
	cudaMalloc((void**)&d_mat, MATRIX_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);

	//transfer the input from CPU to GPU

	cudaMemcpy(d_vec, h_vec, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat, h_mat, MATRIX_BYTES, cudaMemcpyHostToDevice);



	cudaEventRecord(start, 0);
	//launch the kernel
	vector_matrix_mult <<<1, ARRAY_SIZE>>> (d_vec, d_mat, d_out, N, M);
	

	cudaEventRecord(stop, 0);

	

	//trasfer the results from GPU to CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);

	//print out the resulting array
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	printf("The GPU time elapsed is %.6f ms \"", elapsed);

	//free GPU location memory
	cudaFree(d_vec);
	cudaFree(d_mat);
	cudaFree(d_out);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}

/*Realization though CPU*/

/*
#include <time.h>
int main(int argc, char ** argv) {

clock_t cpu_startTime, cpu_stopTime;

double cpu_elapsedTime = 0;
cpu_startTime = clock();

const int ARRAY_SIZE = 4;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

float h_vec[ARRAY_SIZE] = { 1, 1, 1, 1 };
float h_mat[ARRAY_SIZE][ARRAY_SIZE] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };
float risultato[ARRAY_SIZE];
float h_out[ARRAY_SIZE];
float tot = 0;



int i, j;

for (j = 0; j < ARRAY_SIZE; j++){
for (i = 0; i < ARRAY_SIZE; i++) {
risultato[i] = h_vec[i] * h_mat[i][j];
tot += risultato[i];

}
h_out[j] = tot;
tot = 0;

printf("%f", h_out[j]);
printf(((i % 4) != 3) ? "\t" : "\n");
}

cpu_stopTime = clock();
cpu_elapsedTime = ((cpu_startTime - cpu_stopTime) / CLOCKS_PER_SEC);
printf("The CPU elapsed time is %.6f ms \"", cpu_elapsedTime);
return 0;
}
*/