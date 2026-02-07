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
void vector_calc_stride(int* v1, int* v2, int* v3, int N)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < N; i+=stride)
	{
		v3[idx] = v1[idx] + v2[idx];
	}
}
__global__
void vector_calc(int* v1, int* v2, int* v3, int N)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < N)
		v3[idx] = v1[idx] + v2[idx];
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	N = 10000;
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

	int numBlocks = totalThreads/blockSize;
	printf("Total Threads: %d, Block Size: %d, Num Blocks: %d\n", totalThreads, blockSize, numBlocks);
	printf("Computing %ld number of elements\n", N);

	// Seed the random number generator
	srand(time(NULL));

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	long array_size = N;
	long array_size_in_bytes = sizeof(int) * array_size;
	int* gpu_v1;
	int* gpu_v2;
	int* gpu_v3;

	// Init vectors
	int* v1 = (int*)calloc(N , sizeof(int));
	printf("Allocated v1\n");
	int* v2 = (int*)calloc(N , sizeof(int));
	printf("Allocated v2\n");
	init_vectors(v1, v2, min, max);
	// init_vectors_basic(v1, v2);
	printf("Initialized both vectors\n");
	int* v3 = (int*)calloc(N , sizeof(int));

	cudaMalloc((void **)&gpu_v1, array_size_in_bytes);
	cudaMalloc((void **)&gpu_v2, array_size_in_bytes);
	cudaMalloc((void **)&gpu_v3, array_size_in_bytes);
	cudaMemcpy(gpu_v1, v1, array_size_in_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_v2, v2, array_size_in_bytes, cudaMemcpyHostToDevice);

	// Kernel
	vector_calc<<<numBlocks, totalThreads>>>(gpu_v1, gpu_v2, gpu_v3, N);

	cudaDeviceSynchronize();

	// Done with GPU arrays
	cudaMemcpy(v3, gpu_v3, array_size_in_bytes, cudaMemcpyDeviceToHost);
	cudaFree(gpu_v1);
	cudaFree(gpu_v2);
	cudaFree(gpu_v3);

	for (unsigned int i = array_size-10; i < array_size; ++i)
	{
		printf("Vec3 idx %d: %d\n", i, v3[i]);
	}
	free(v1);
	free(v2);
	free(v3);
}
