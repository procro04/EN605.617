// matmul.cl
// Matrix multiplication kernel using OpenCL buffers

__kernel void matmul(__global const float *A, __global const float *B,
                     __global float *C, const int M, const int N, const int K) {
  // Each work-item computes one element of C
  // C is MxN, A is MxK, B is KxN
  int row = get_global_id(0);
  int col = get_global_id(1);

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;
  for (int i = 0; i < K; i++) {
    sum += A[row * K + i] * B[i * N + col];
  }

  C[row * N + col] = sum;
}
