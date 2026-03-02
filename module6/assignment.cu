#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <vector>
#include "CodeTimer.h"

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

void stream_test(int numBlocks, int blockSize, int N, int pattern, int nStreams)
{
    std::cout << "========== CUDA STREAM DEMO ==========\n";
    int array_size = N;
    int chunkSize = N / nStreams;
    long chunkSizeInBytes = sizeof(int) * chunkSize;

    int *h_v1, *h_v2, *h_v3;
    cudaHostAlloc((void**)&h_v1, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_v2, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_v3, N * sizeof(int), cudaHostAllocDefault);
    init_vectors(h_v1, h_v2);

    int *d_v1, *d_v2, *d_v3;
    cudaMalloc((void**)&d_v1, N * sizeof(int));
    cudaMalloc((void**)&d_v2, N * sizeof(int));
    cudaMalloc((void**)&d_v3, N * sizeof(int));

    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&streams[i]);

    CodeTimer timer;
    timer.startTiming();

    // Depth-first: each stream gets its full pipeline slice
    for (int i = 0; i < nStreams; i++) {
        int offset = i * chunkSize;

        cudaMemcpyAsync(d_v1 + offset, h_v1 + offset,
                        chunkSizeInBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_v2 + offset, h_v2 + offset,
                        chunkSizeInBytes, cudaMemcpyHostToDevice, streams[i]);

        // Kernel sees only its chunk: pointer arithmetic shifts the base,
        // N becomes chunkSize — kernel internals don't change at all
        vector_calc<<<numBlocks, blockSize, 0, streams[i]>>>(
            &d_v1[offset], &d_v2[offset], &d_v3[offset], chunkSize, pattern);

        cudaMemcpyAsync(h_v3 + offset, d_v3 + offset,
                        chunkSizeInBytes, cudaMemcpyHostToDevice, streams[i]);
    }

    for (int i = 0; i < nStreams; i++)
        cudaStreamSynchronize(streams[i]);

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
    // Done with gpu arrays
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_v3);
}

int main(int argc, char** argv)
{
    // read command line arguments
    int totalThreads = 1024;
    int blockSize = 256;
    N = 8000;
    int nStreams = 4;
    int pattern = 0;

    // Seed the random number generator
    srand(time(NULL));

    if (argc >= 2) totalThreads = atoi(argv[1]);
    if (argc >= 3) blockSize = atoi(argv[2]);
    if (argc >= 4) N = atoi(argv[3]);
    if (argc >= 5) pattern = atoi(argv[4]);
    if (argc >= 6) nStreams = atoi(argv[5]);

    int numBlocks = totalThreads/blockSize;
    std::cout << "Total Threads: " << totalThreads << "\n"
              << "Block Size: " << blockSize << "\n"
              << "Num Blocks: " << numBlocks << "\n"
              << "Pattern: " << pattern << "\n"
              << "Num Streams: " << nStreams << "\n";
    std::cout << "Computing " << N << " elements\n\n";

    // validate command line arguments
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks*blockSize;
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }
    stream_test(numBlocks, blockSize, N, pattern, nStreams);
}
