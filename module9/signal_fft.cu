#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>

// Minimal error-checking helpers
#define CUDA_CHECK(x)                                                                        \
    do                                                                                       \
    {                                                                                        \
        cudaError_t e = (x);                                                                 \
        if (e)                                                                               \
        {                                                                                    \
            fprintf(stderr, "CUDA  %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)
#define CURAND_CHECK(x)                                                            \
    do                                                                             \
    {                                                                              \
        curandStatus_t e = (x);                                                    \
        if (e)                                                                     \
        {                                                                          \
            fprintf(stderr, "cuRAND %s:%d  err=%d\n", __FILE__, __LINE__, (int)e); \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)
#define CUFFT_CHECK(x)                                                             \
    do                                                                             \
    {                                                                              \
        cufftResult e = (x);                                                       \
        if (e)                                                                     \
        {                                                                          \
            fprintf(stderr, "cuFFT  %s:%d  err=%d\n", __FILE__, __LINE__, (int)e); \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

// Very basic main function to show the use of 2 cuda advanced library functions
int main()
{
    constexpr int N = 1024;

    // Generate N random floats with cuRAND
    float* d_signal{};
    CUDA_CHECK(cudaMalloc(&d_signal, N * sizeof(float)));

    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));
    CURAND_CHECK(curandGenerateUniform(rng, d_signal, N));   // fills with U(0,1)
    CURAND_CHECK(curandDestroyGenerator(rng));

    // Run an in-place R2C FFT with cuFFT
    // R2C output has N/2+1 complex bins
    constexpr int NUM_BINS = N / 2 + 1;
    cufftComplex* d_freq{};
    CUDA_CHECK(cudaMalloc(&d_freq, NUM_BINS * sizeof(cufftComplex)));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, /*batch=*/1));
    CUFFT_CHECK(cufftExecR2C(plan, d_signal, d_freq));
    CUFFT_CHECK(cufftDestroy(plan));

    // Copy results to host and print the first few magnitudes ────────────
    std::vector<cufftComplex> h_freq(NUM_BINS);
    CUDA_CHECK(cudaMemcpy(h_freq.data(), d_freq,
                          NUM_BINS * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    int printNumBins = 10;
    std::printf("FFT of %d-point uniform-random signal  (first 10 bins)\n", N);
    std::printf("  %-6s  %-12s  %s\n", "bin", "magnitude", "complex");
    // Skip printing bin 0 because it always is around the same value
    for (int k = 1; k < printNumBins+1; ++k) {
        float re  = h_freq[k].x;
        float im  = h_freq[k].y;
        float mag = std::sqrt(re*re + im*im);
        std::printf("  %-6d  %-12.4f  (%.4f, %.4f)\n", k, mag, re, im);
    }

    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_freq));
    return 0;
}
