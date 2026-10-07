
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 8;
    const size_t size = N * sizeof(float);

    // -----------------------------
    // 1. Host (CPU) arrays
    // -----------------------------

    float h_A[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_B[N] = {10, 20, 30, 40, 50, 60, 70, 80};
    float h_C[N];

    // -----------------------------
    // 2. Device (GPU) pointers
    // -----------------------------

    float *d_A;
    float *d_B;
    float *d_C;

    // -----------------------------
    // 3. Allocate GPU memory
    // -----------------------------

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // -----------------------------
    // 4. Copy CPU → GPU
    // -----------------------------

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // -----------------------------
    // 5. Launch CUDA kernel
    // -----------------------------

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)
                        / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // -----------------------------
    // 6. Copy GPU → CPU
    // -----------------------------

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // -----------------------------
    // 7. Print result
    // -----------------------------

    std::cout << "Result:\n";

    for (int i = 0; i < N; i++)
    {
        std::cout << h_A[i]
                  << " + "
                  << h_B[i]
                  << " = "
                  << h_C[i]
                  << "\n";
    }

    // -----------------------------
    // 8. Free GPU memory
    // -----------------------------

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
