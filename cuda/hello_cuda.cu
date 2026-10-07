
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_cuda()
{
    printf("Hello from CUDA! Block=%d Thread=%d\n",
           blockIdx.x,
           threadIdx.x);
}

int main()
{
    hello_cuda<<<1, 4>>>();

    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n",
               cudaGetErrorString(error));
        return 1;
    }

    printf("CUDA execution successful!\n");

    return 0;
}
