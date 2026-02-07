//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long N; // Number of elements

void init_vectors(int* v1, int* v2, int min, int max)
{
    for (int i = 0; i < N; i++) {
        v1[i] = min + rand() % (max - min + 1);
        v2[i] = min + rand() % (max - min + 1);
    }
}

void init_vectors_basic(int* v1, int* v2)
{
    for (int i = 0; i < N; i++) {
		v1[i] = 1;
		v2[i] = 2;
	}
}

// __global__
// void vector_calc(
// 	unsigned int* block,
// 	int* v1, int* v2, int* v3, int N)
// {
// 	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
// 	int branch = N/2;
// 	for (int i = 0; i < N; ++i)
// 	{
// 		if (i < branch)
// 			v3[i] = v1[i] + v2[i];
// 		else
// 			v3[i] = v1[i] - v2[i];
// 	}
// }
__global__
void vector_calc(
	unsigned int* block,
	int* v1, int* v2, int* v3, int N)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	for (int i = 0; i < N; ++i)
	{
		v3[i] = v1[i] + v2[i];
	}
}

int main(int argc, char** argv)
{
	// read command line arguments
	// int totalThreads = (1 << 20);
	// int blockSize = 256;
	// int N = 1000;
	// int min = -1000;
	// int max = 1000;

	int totalThreads = 32;
	int blockSize = 1;
	int N = 32;
	int min = -1000;
	int max = 1000;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}
	if (argc >=4) {
		N = atoi(argv[3]);
	}
	printf("Total Threads: %d, Block Size: %d\n", totalThreads, blockSize);
	printf("Computing %d number of elements\n", N);

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	int num_vecs = 3;
	long array_size = N;
	long array_size_in_bytes = sizeof(unsigned int) * array_size * num_vecs;
	unsigned int *gpu_block;
	unsigned int *gpu_thread;

	// Init vectors
	int* v1 = (int*)calloc(N , sizeof(int));
	printf("Allocated v1\n");
	int* v2 = (int*)calloc(N , sizeof(int));
	printf("Allocated v2\n");
	// init_vectors(v1, v2, min, max);
	init_vectors_basic(v1, v2);
	printf("Initialized both vectors\n");
	int* v3 = (int*)calloc(N , sizeof(int));
	unsigned int* cpu_block = (unsigned int*)malloc(num_vecs * N * sizeof(int));

	cudaMalloc((void **)&gpu_block, array_size_in_bytes);
	cudaMemcpy(cpu_block, gpu_block, array_size_in_bytes, cudaMemcpyHostToDevice);

	// Kernel
	vector_calc<<<numBlocks, blockSize>>>(gpu_block, v1, v2, v3, N);

	// Done with GPU arrays
	cudaMemcpy(cpu_block, gpu_block, array_size_in_bytes, cudaMemcpyDeviceToHost);
	cudaFree(gpu_block);

	for (unsigned int i = 0; i < array_size; ++i)
	{
		printf("Vec3 idx %d: %d\n", i, v3[i]);
	}

}
