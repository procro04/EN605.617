// matmul.cpp
// Matrix multiplication using OpenCL buffers and sub-buffers
// Based on the simple buffer example from the modules

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <CL/cl.h>

// Use the first platform that pops up
#define DEFAULT_PLATFORM 0

inline void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void printUsage(const char *progName)
{
    std::cerr << "Usage: " << progName
              << " --m <rows_A> --k <cols_A_rows_B> --n <cols_B>"
              << " [--platform <id>] [--verify]" << std::endl;
    std::cerr << "  Matrix A is MxK, Matrix B is KxN, Result C is MxN" << std::endl;
}

// CPU reference multiply for verification - quick sanity check
void cpuMatmul(const float *A, const float *B, float *C, int M, int K, int N)
{
    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            float sum = 0.0f;
            for (int i = 0; i < K; i++)
                sum += A[r * K + i] * B[i * N + c];
            C[r * N + c] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    int M = 0, K = 0, N = 0;
    int platform = DEFAULT_PLATFORM;
    bool verify = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc)
        {
            M = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc)
        {
            K = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc)
        {
            N = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc)
        {
            platform = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--verify") == 0)
        {
            verify = true;
        }
        else
        {
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate dimensions - exit early if the matricies are not valid to be multiplied
    if (M <= 0 || K <= 0 || N <= 0)
    {
        std::cerr << "ERROR: All matrix dimensions must be positive integers." << std::endl;
        std::cerr << "  Got M=" << M << " K=" << K << " N=" << N << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Inner dimensions must match: A is MxK, B is KxN (K is shared)
    // This is already enforced by having a single --k parameter, but
    // print the shapes so the user can confirm what they asked for.
    std::cout << "Matrix multiply: A(" << M << "x" << K << ") * B("
              << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;

    // OpenCL platform and device setup
    cl_int errNum;
    cl_uint numPlatforms;
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
             "clGetPlatformIDs");

    std::vector<cl_platform_id> platformIDs(numPlatforms);
    errNum = clGetPlatformIDs(numPlatforms, platformIDs.data(), NULL);
    checkErr(errNum, "clGetPlatformIDs");

    if (platform < 0 || platform >= (int)numPlatforms)
    {
        std::cerr << "ERROR: Invalid platform index " << platform
                  << " (available: 0-" << numPlatforms - 1 << ")" << std::endl;
        return 1;
    }

    cl_uint numDevices;
    errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL,
                            0, NULL, &numDevices);
    checkErr(errNum, "clGetDeviceIDs");

    std::vector<cl_device_id> deviceIDs(numDevices);
    errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL,
                            numDevices, deviceIDs.data(), NULL);
    checkErr(errNum, "clGetDeviceIDs");

    std::cout << "Using platform " << platform
              << " with " << numDevices << " device(s)" << std::endl;

    // Create context
    cl_context_properties contextProps[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[platform], 0};
    cl_context context = clCreateContext(contextProps, numDevices,
                                         deviceIDs.data(), NULL, NULL, &errNum);
    checkErr(errNum, "clCreateContext");

    // Load and build kernel source
    std::ifstream srcFile("matmul.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading matmul.cl");
    std::string srcProg((std::istreambuf_iterator<char>(srcFile)),
                        std::istreambuf_iterator<char>());
    const char *src = srcProg.c_str();
    size_t srcLen = srcProg.length();

    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcLen, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    errNum = clBuildProgram(program, numDevices, deviceIDs.data(), NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        std::cerr << "Build error:\n"
                  << buildLog << std::endl;
        checkErr(errNum, "clBuildProgram");
    }

    // Prepare host data
    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC, 0.0f);

    // Fill with simple test values
    for (size_t i = 0; i < sizeA; i++)
        h_A[i] = (float)(i % 10);
    for (size_t i = 0; i < sizeB; i++)
        h_B[i] = (float)(i % 7);

    // Create buffers
    // Main buffer holds both A and B contiguously, demonstrating sub-buffers.
    // Layout: [A data (MxK floats)][B data (KxN floats)]
    size_t totalElements = sizeA + sizeB;
    cl_mem mainBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       sizeof(float) * totalElements,
                                       NULL, &errNum);
    checkErr(errNum, "clCreateBuffer(mainBuffer)");

    // Sub-buffer for matrix A (offset 0)
    cl_buffer_region regionA = {0, sizeof(float) * sizeA};
    cl_mem bufA = clCreateSubBuffer(mainBuffer, CL_MEM_READ_ONLY,
                                    CL_BUFFER_CREATE_TYPE_REGION,
                                    &regionA, &errNum);
    checkErr(errNum, "clCreateSubBuffer(A)");

    // Sub-buffer for matrix B (offset after A)
    cl_buffer_region regionB = {sizeof(float) * sizeA, sizeof(float) * sizeB};
    cl_mem bufB = clCreateSubBuffer(mainBuffer, CL_MEM_READ_ONLY,
                                    CL_BUFFER_CREATE_TYPE_REGION,
                                    &regionB, &errNum);
    checkErr(errNum, "clCreateSubBuffer(B)");

    // Separate buffer for result C
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * sizeC, NULL, &errNum);
    checkErr(errNum, "clCreateBuffer(C)");

    // Create command queue on first device
    cl_command_queue queue = clCreateCommandQueue(context, deviceIDs[0], 0, &errNum);
    checkErr(errNum, "clCreateCommandQueue");

    // Write input data via the sub-buffers
    errNum = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
                                  sizeof(float) * sizeA, h_A.data(),
                                  0, NULL, NULL);
    checkErr(errNum, "clEnqueueWriteBuffer(A)");

    errNum = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
                                  sizeof(float) * sizeB, h_B.data(),
                                  0, NULL, NULL);
    checkErr(errNum, "clEnqueueWriteBuffer(B)");

    // Setup kernel
    cl_kernel kernel = clCreateKernel(program, "matmul", &errNum);
    checkErr(errNum, "clCreateKernel(matmul)");

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    errNum |= clSetKernelArg(kernel, 3, sizeof(int), &M);
    errNum |= clSetKernelArg(kernel, 4, sizeof(int), &N);
    errNum |= clSetKernelArg(kernel, 5, sizeof(int), &K);
    checkErr(errNum, "clSetKernelArg");

    // 2D global work size: one work-item per output element
    size_t globalWork[2] = {(size_t)M, (size_t)N};

    errNum = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
                                    globalWork, NULL, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");

    // Read back results
    errNum = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                 sizeof(float) * sizeC, h_C.data(),
                                 0, NULL, NULL);
    checkErr(errNum, "clEnqueueReadBuffer(C)");

    clFinish(queue);

    // Print result (only for small matrices)
    if (M <= 16 && N <= 16)
    {
        std::cout << "\nResult matrix C (" << M << "x" << N << "):" << std::endl;
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
                std::cout << h_C[r * N + c] << "\t";
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "\nMatrix too large to print. Showing top-left 4x4 corner:" << std::endl;
        int pM = (M < 4) ? M : 4;
        int pN = (N < 4) ? N : 4;
        for (int r = 0; r < pM; r++)
        {
            for (int c = 0; c < pN; c++)
                std::cout << h_C[r * N + c] << "\t";
            std::cout << std::endl;
        }
    }

    // Optional CPU verification
    if (verify)
    {
        std::cout << "\nVerifying against CPU result..." << std::endl;
        std::vector<float> h_ref(sizeC);
        cpuMatmul(h_A.data(), h_B.data(), h_ref.data(), M, K, N);

        float maxErr = 0.0f;
        for (size_t i = 0; i < sizeC; i++)
        {
            float diff = fabs(h_C[i] - h_ref[i]);
            if (diff > maxErr)
                maxErr = diff;
        }
        std::cout << "Max absolute error: " << maxErr << std::endl;
        if (maxErr < 1e-3f)
            std::cout << "PASSED" << std::endl;
        else
            std::cout << "FAILED" << std::endl;
    }

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);
    clReleaseMemObject(mainBuffer);
    clReleaseProgram(program);
    clReleaseContext(context);

    std::cout << "Program completed successfully" << std::endl;
    return 0;
}
