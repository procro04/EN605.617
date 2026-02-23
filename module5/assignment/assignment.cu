#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include "CodeTimer.h"

#define CONST_SIZE 8000
__constant__ int const_v1[CONST_SIZE];
__constant__ int const_v2[CONST_SIZE];

long N; // Number of elements

void init_vectors(int* v1, int* v2)
{
    int min = -1000;
    int max = 1000;
    for (int i = 0; i < N; i++) {
        v1[i] = min + rand() % (max - min + 1);
        v2[i] = min + rand() % (max - min + 1);
    }
}

__global__
void vector_calc(
    int* v1, int* v2, int* v3, int N, int pattern)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int grid_stride = blockDim.x * gridDim.x;
    
    for (int i = thread_idx; i < N; i += grid_stride)
    {
        bool condition;
        switch(pattern) {
            case 0: condition = true; break;
            case 1: condition = (i % 2 == 0); break;
            case 2: condition = (i < N / 2); break;
            default: condition = true; break;
        }

        int result;
        if (condition) {
            result = v1[i];
            for (int j = 0; j < 100; j++) result = result * 3 + v2[i];
        } else {
            result = v2[i];
            for (int j = 0; j < 100; j++) result = result * 3 + v1[i];
        }
        v3[i] = result;
    }
}

__global__
void vector_calc_const_mem(
    int* v3, int pattern)
{
    int N = CONST_SIZE;
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int grid_stride = blockDim.x * gridDim.x;
    
    for (int i = thread_idx; i < N; i += grid_stride)
    {
        bool condition;
        switch(pattern) {
            case 0: condition = true; break;
            case 1: condition = (i % 2 == 0); break;
            case 2: condition = (i < N / 2); break;
            default: condition = true; break;
        }

        int result;
        if (condition) {
            result = const_v1[i];
            for (int j = 0; j < 100; j++) result = result * 3 + const_v2[i];
        } else {
            result = const_v2[i];
            for (int j = 0; j < 100; j++) result = result * 3 + const_v1[i];
        }
        v3[i] = result;
    }
}

__global__
void vector_calc_register_mem(
    int* v1, int* v2, int* v3, int N, int pattern)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int grid_stride = blockDim.x * gridDim.x;
    
    for (int i = thread_idx; i < N; i += grid_stride)
    {
        int reg_1 = v1[i];
        int reg_2 = v2[i];
        bool condition;
        switch(pattern) {
            case 0: condition = true; break;
            case 1: condition = (i % 2 == 0); break;
            case 2: condition = (i < N / 2); break;
            default: condition = true; break;
        }

        int result;
        if (condition) {
            result = reg_1;
            for (int j = 0; j < 100; j++) result = result * 3 + reg_2;
        } else {
            result = reg_2;
            for (int j = 0; j < 100; j++) result = result * 3 + reg_1;
        }
        v3[i] = result;
    }
}

void host_memory_test(int numBlocks, int blockSize, int N, int pattern)
{
    std::cout << "========== HOST MEMORY DEMO ==========\n";
    long array_size = N;
    long array_size_in_bytes = sizeof(int) * array_size;

    // Init vectors
    int *h_v1, *h_v2, *h_v3;
    cudaHostAlloc((void **)&h_v1, array_size_in_bytes, cudaHostAllocPortable);
    cudaHostAlloc((void **)&h_v2, array_size_in_bytes, cudaHostAllocPortable);
    cudaHostAlloc((void **)&h_v3, array_size_in_bytes, cudaHostAllocPortable);
    init_vectors(h_v1, h_v2);

    int *d_v1, *d_v2, *d_v3;
    cudaHostGetDevicePointer((void**)&d_v1, (void*)h_v1, 0);
    cudaHostGetDevicePointer((void**)&d_v2, (void*)h_v2, 0);
    cudaHostGetDevicePointer((void**)&d_v3, (void*)h_v3, 0);

    CodeTimer timer;
    timer.startTiming();

    // Execute the kernel
    vector_calc
        <<<numBlocks, blockSize>>>(d_v1, d_v2, d_v3, N, pattern);
    cudaDeviceSynchronize();

    timer.stopTiming();
    std::cout << "HOST computation took: " 
              << timer.elapsedSeconds() << " seconds\n";

    std::cout << "Sample Results (last 5)\n";
    for (unsigned int i = array_size-5; i < array_size; ++i) {
        std::cout << "Index " << i << ": " << h_v3[i] << "\n";
    }
    // Done with host arrays
    cudaFreeHost(h_v1);
    cudaFreeHost(h_v2);
    cudaFreeHost(h_v3);
}

void global_memory_test(int numBlocks, int blockSize, int N, int pattern)
{
    std::cout << "========== GLOBAL MEMORY DEMO ==========\n";
    long array_size = N;
    long array_size_in_bytes = sizeof(int) * array_size;
    int *gpu_v1, *gpu_v2, *gpu_v3;

    // Init vectors
    int* v1 = (int*)calloc(array_size , sizeof(int));
    int* v2 = (int*)calloc(array_size , sizeof(int));
    int* v3 = (int*)calloc(N , sizeof(int));
    init_vectors(v1, v2);

    cudaMalloc((void **)&gpu_v1, array_size_in_bytes);
    cudaMalloc((void **)&gpu_v2, array_size_in_bytes);
    cudaMalloc((void **)&gpu_v3, array_size_in_bytes);
    cudaMemcpy(gpu_v1, v1, array_size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v2, v2, array_size_in_bytes, cudaMemcpyHostToDevice);

    CodeTimer timer;
    timer.startTiming();

    // Execute the kernel
    vector_calc
        <<<numBlocks, blockSize>>>(gpu_v1, gpu_v2, gpu_v3, N, pattern);
    cudaDeviceSynchronize();

    timer.stopTiming();
    std::cout << "GLOBAL computation took: " 
              << timer.elapsedSeconds() << " seconds\n";

    // Done with GPU arrays
    cudaMemcpy(v3, gpu_v3, array_size_in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(gpu_v1);
    cudaFree(gpu_v2);
    cudaFree(gpu_v3);

    std::cout << "Sample Results (last 5)\n";
    for (unsigned int i = array_size-5; i < array_size; ++i) {
        std::cout << "Index " << i << ": " << v3[i] << "\n";
    }
    // Done with host arrays
    free(v1);
    free(v2);
    free(v3);
}

void constant_memory_test(int numBlocks, int blockSize, int pattern)
{
    std::cout << "========== CONSTANT MEMORY DEMO ==========\n";
    int *gpu_v3;
    int array_size = CONST_SIZE;
    int array_size_in_bytes = CONST_SIZE * sizeof(int);

    // Init const vectors
    int v1[CONST_SIZE], v2[CONST_SIZE];
    init_vectors(v1, v2);
    cudaMemcpyToSymbol(const_v1, v1, CONST_SIZE * sizeof(int));
    cudaMemcpyToSymbol(const_v2, v2, CONST_SIZE * sizeof(int));
    // std::cout << "Initialized all const memory";

    // Init dynamic mem so we can write the result
    std::vector<int> v3(array_size_in_bytes);
    cudaMalloc((void **)&gpu_v3, array_size_in_bytes);
    cudaMemcpy(gpu_v3, v3.data(), array_size_in_bytes, cudaMemcpyHostToDevice);
    // std::cout << "Initialized all dynamic memory";

    CodeTimer timer;
    timer.startTiming();

    // Execute the kernel
    vector_calc_const_mem
        <<<numBlocks, blockSize>>>(gpu_v3, pattern);
    cudaDeviceSynchronize();

    timer.stopTiming();
    std::cout << "CONSTANT computation took: " 
              << timer.elapsedSeconds() << " seconds\n";

    // Done with GPU array
    cudaMemcpy(v3.data(), gpu_v3, array_size_in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(gpu_v3);

    std::cout << "Sample Results (last 5)\n";
    for (unsigned int i = array_size-5; i < array_size; ++i) {
        std::cout << "Index " << i << ": " << v3[i] << "\n";
    }
}

void register_memory_test(int numBlocks, int blockSize, int N, int pattern)
{
    std::cout << "========== REGISTER MEMORY DEMO ==========\n";
    long array_size = N;
    long array_size_in_bytes = sizeof(int) * array_size;
    int *gpu_v1, *gpu_v2, *gpu_v3;

    // Init vectors
    int* v1 = (int*)calloc(array_size , sizeof(int));
    int* v2 = (int*)calloc(array_size , sizeof(int));
    int* v3 = (int*)calloc(N , sizeof(int));
    init_vectors(v1, v2);

    cudaMalloc((void **)&gpu_v1, array_size_in_bytes);
    cudaMalloc((void **)&gpu_v2, array_size_in_bytes);
    cudaMalloc((void **)&gpu_v3, array_size_in_bytes);
    cudaMemcpy(gpu_v1, v1, array_size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v2, v2, array_size_in_bytes, cudaMemcpyHostToDevice);

    CodeTimer timer;
    timer.startTiming();

    // Execute the kernel
    vector_calc_register_mem
        <<<numBlocks, blockSize>>>(gpu_v1, gpu_v2, gpu_v3, N, pattern);
    cudaDeviceSynchronize();

    timer.stopTiming();
    std::cout << "REGISTER computation took: " 
              << timer.elapsedSeconds() << " seconds\n";

    // Done with GPU arrays
    cudaMemcpy(v3, gpu_v3, array_size_in_bytes, cudaMemcpyDeviceToHost);
    cudaFree(gpu_v1);
    cudaFree(gpu_v2);
    cudaFree(gpu_v3);

    std::cout << "Sample Results (last 5)\n";
    for (unsigned int i = array_size-5; i < array_size; ++i) {
        std::cout << "Index " << i << ": " << v3[i] << "\n";
    }
    // Done with host arrays
    free(v1);
    free(v2);
    free(v3);
}

int main(int argc, char** argv)
{
    // read command line arguments
    int totalThreads = (1 << 20);
    int blockSize = 256;
    N = 8000;
    int pattern = 0;

    // Seed the random number generator
    srand(time(NULL));

    if (argc >= 2) totalThreads = atoi(argv[1]);
    if (argc >= 3) blockSize = atoi(argv[2]);
    if (argc >= 4) N = atoi(argv[3]);
    if (argc >= 5) pattern = atoi(argv[4]);

    int numBlocks = totalThreads/blockSize;
    std::cout << "Total Threads: " << totalThreads << "\n"
              << "Block Size: " << blockSize << "\n"
              << "Num Blocks: " << numBlocks << "\n";
    std::cout << "Computing " << N << " elements\n\n";

    host_memory_test(numBlocks, blockSize, N, pattern);
    global_memory_test(numBlocks, blockSize, N, pattern);
    constant_memory_test(numBlocks, blockSize, pattern);
    register_memory_test(numBlocks, blockSize, N, pattern);

    // validate command line arguments
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks*blockSize;
        
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }
}
