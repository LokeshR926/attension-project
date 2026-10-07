#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                            \
        if (err != cudaSuccess) {                            \
            std::cerr << "CUDA error: "                         \
                      << cudaGetErrorString(err) << "\n";    \
            std::exit(EXIT_FAILURE);                         \
        }                                                     \
    } while (0)

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;

    for (int k = 0; k < K; ++k)
    {
        sum += A[row * K + k] *
               B[k * N + col];
    }

    C[row * N + col] = sum;
}

__global__ void transpose_kernel(
    const float* A,
    float* B,
    int rows,
    int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        B[col * rows + row] =
            A[row * cols + col];
    }
}

__global__ void scale_kernel(
    float* A,
    float scale,
    int N)
{
    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    int total = N * N;

    if (idx < total)
        A[idx] *= scale;
}

__global__ void softmax_rows_kernel(
    float* A,
    int N)
{
    int row = blockIdx.x;

    if (row >= N)
        return;

    float max_value = -INFINITY;

    for (int j = 0; j < N; ++j)
    {
        max_value =
            fmaxf(max_value,
                  A[row * N + j]);
    }

    float sum = 0.0f;

    for (int j = 0; j < N; ++j)
    {
        float value =
            expf(A[row * N + j] - max_value);

        A[row * N + j] = value;
        sum += value;
    }

    for (int j = 0; j < N; ++j)
    {
        A[row * N + j] /= sum;
    }
}

void initialize_matrix(
    std::vector<float>& matrix,
    unsigned seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(
        -1.0f,
        1.0f
    );

    for (float& value : matrix)
        value = distribution(generator);
}

float max_absolute_error(
    const std::vector<float>& A,
    const std::vector<float>& B)
{
    float error = 0.0f;

    for (std::size_t i = 0; i < A.size(); ++i)
    {
        error = std::max(
            error,
            std::fabs(A[i] - B[i])
        );
    }

    return error;
}

void cpu_matmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int K,
    int N)
{
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            float value =
                A[i * K + k];

            for (int j = 0; j < N; ++j)
            {
                C[i * N + j] +=
                    value *
                    B[k * N + j];
            }
        }
    }
}

void cpu_transpose(
    const std::vector<float>& A,
    std::vector<float>& B,
    int rows,
    int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            B[j * rows + i] =
                A[i * cols + j];
        }
    }
}

void cpu_softmax(
    std::vector<float>& A,
    int N)
{
    for (int i = 0; i < N; ++i)
    {
        float max_value = -INFINITY;

        for (int j = 0; j < N; ++j)
        {
            max_value =
                std::max(
                    max_value,
                    A[i * N + j]
                );
        }

        float sum = 0.0f;

        for (int j = 0; j < N; ++j)
        {
            float value =
                std::exp(
                    A[i * N + j] -
                    max_value
                );

            A[i * N + j] = value;
            sum += value;
        }

        for (int j = 0; j < N; ++j)
        {
            A[i * N + j] /= sum;
        }
    }
}

void cpu_attention(
    const std::vector<float>& X,
    const std::vector<float>& WQ,
    const std::vector<float>& WK,
    const std::vector<float>& WV,
    std::vector<float>& output,
    int N,
    int D,
    int dk,
    int dv)
{
    std::vector<float> Q(N * dk);
    std::vector<float> K(N * dk);
    std::vector<float> V(N * dv);

    std::vector<float> KT(dk * N);
    std::vector<float> scores(N * N);
    std::vector<float> weights(N * N);

    cpu_matmul(
        X, WQ, Q,
        N, D, dk
    );

    cpu_matmul(
        X, WK, K,
        N, D, dk
    );

    cpu_matmul(
        X, WV, V,
        N, D, dv
    );

    cpu_transpose(
        K, KT,
        N, dk
    );

    cpu_matmul(
        Q, KT, scores,
        N, dk, N
    );

    float scale =
        1.0f / std::sqrt(
            static_cast<float>(dk)
        );

    for (float& value : scores)
        value *= scale;

    weights = scores;

    cpu_softmax(
        weights,
        N
    );

    cpu_matmul(
        weights,
        V,
        output,
        N, N, dv
    );
}

void run_cuda_attention(
    const std::vector<float>& h_X,
    const std::vector<float>& h_WQ,
    const std::vector<float>& h_WK,
    const std::vector<float>& h_WV,
    std::vector<float>& h_output,
    int N,
    int D,
    int dk,
    int dv,
    float& total_ms,
    float& qkv_ms,
    float& transpose_ms,
    float& qkt_ms,
    float& scale_ms,
    float& softmax_ms,
    float& av_ms)
{
    const int block_size = 16;

    float *d_X = nullptr;
    float *d_WQ = nullptr;
    float *d_WK = nullptr;
    float *d_WV = nullptr;

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;

    float *d_KT = nullptr;
    float *d_scores = nullptr;
    float *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_X,
        h_X.size() * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_WQ,
        h_WQ.size() * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_WK,
        h_WK.size() * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_WV,
        h_WV.size() * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_Q,
        N * dk * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_K,
        N * dk * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_V,
        N * dv * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_KT,
        dk * N * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_scores,
        N * N * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        N * dv * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_X,
        h_X.data(),
        h_X.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_WQ,
        h_WQ.data(),
        h_WQ.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_WK,
        h_WK.data(),
        h_WK.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_WV,
        h_WV.data(),
        h_WV.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    dim3 block(
        block_size,
        block_size
    );

    dim3 qkv_grid(
        (dk + block_size - 1) / block_size,
        (N + block_size - 1) / block_size
    );

    dim3 v_grid(
        (dv + block_size - 1) / block_size,
        (N + block_size - 1) / block_size
    );

    dim3 score_grid(
        (N + block_size - 1) / block_size,
        (N + block_size - 1) / block_size
    );

    dim3 transpose_grid(
        (dk + block_size - 1) / block_size,
        (N + block_size - 1) / block_size
    );

    int total_scores = N * N;

    int threads_1d = 256;

    int blocks_1d =
        (total_scores + threads_1d - 1) /
        threads_1d;

    cudaEvent_t start, stop;
    cudaEvent_t qkv_start, qkv_stop;
    cudaEvent_t transpose_start, transpose_stop;
    cudaEvent_t qkt_start, qkt_stop;
    cudaEvent_t scale_start, scale_stop;
    cudaEvent_t softmax_start, softmax_stop;
    cudaEvent_t av_start, av_stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventCreate(&qkv_start));
    CUDA_CHECK(cudaEventCreate(&qkv_stop));
    CUDA_CHECK(cudaEventCreate(&transpose_start));
    CUDA_CHECK(cudaEventCreate(&transpose_stop));
    CUDA_CHECK(cudaEventCreate(&qkt_start));
    CUDA_CHECK(cudaEventCreate(&qkt_stop));
    CUDA_CHECK(cudaEventCreate(&scale_start));
    CUDA_CHECK(cudaEventCreate(&scale_stop));
    CUDA_CHECK(cudaEventCreate(&softmax_start));
    CUDA_CHECK(cudaEventCreate(&softmax_stop));
    CUDA_CHECK(cudaEventCreate(&av_start));
    CUDA_CHECK(cudaEventCreate(&av_stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaEventRecord(qkv_start));

    matmul_kernel<<<qkv_grid, block>>>(
        d_X,
        d_WQ,
        d_Q,
        N,
        D,
        dk
    );

    matmul_kernel<<<qkv_grid, block>>>(
        d_X,
        d_WK,
        d_K,
        N,
        D,
        dk
    );

    matmul_kernel<<<v_grid, block>>>(
        d_X,
        d_WV,
        d_V,
        N,
        D,
        dv
    );

    CUDA_CHECK(cudaEventRecord(qkv_stop));
    CUDA_CHECK(cudaEventRecord(transpose_start));

    transpose_kernel<<<transpose_grid, block>>>(
        d_K,
        d_KT,
        N,
        dk
    );

    CUDA_CHECK(cudaEventRecord(transpose_stop));

    CUDA_CHECK(cudaEventRecord(qkt_start));

    matmul_kernel<<<score_grid, block>>>(
        d_Q,
        d_KT,
        d_scores,
        N,
        dk,
        N
    );

    CUDA_CHECK(cudaEventRecord(qkt_stop));
    CUDA_CHECK(cudaEventRecord(scale_start));

    float scale =
        1.0f / std::sqrt(
            static_cast<float>(dk)
        );

    scale_kernel<<<blocks_1d, threads_1d>>>(
        d_scores,
        scale,
        N
    );

    CUDA_CHECK(cudaEventRecord(scale_stop));
    CUDA_CHECK(cudaEventRecord(softmax_start));

    softmax_rows_kernel<<<N, 1>>>(
        d_scores,
        N
    );

    CUDA_CHECK(cudaEventRecord(softmax_stop));
    CUDA_CHECK(cudaEventRecord(av_start));

    matmul_kernel<<<v_grid, block>>>(
        d_scores,
        d_V,
        d_output,
        N,
        N,
        dv
    );

    CUDA_CHECK(cudaEventRecord(av_stop));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(
        &qkv_ms,
        qkv_start,
        qkv_stop
    ));

    CUDA_CHECK(cudaEventElapsedTime(
        &transpose_ms,
        transpose_start,
        transpose_stop
    ));

    CUDA_CHECK(cudaEventElapsedTime(
        &qkt_ms,
        qkt_start,
        qkt_stop
    ));

    CUDA_CHECK(cudaEventElapsedTime(
        &scale_ms,
        scale_start,
        scale_stop
    ));

    CUDA_CHECK(cudaEventElapsedTime(
        &softmax_ms,
        softmax_start,
        softmax_stop
    ));

    CUDA_CHECK(cudaEventElapsedTime(
        &av_ms,
        av_start,
        av_stop
    ));

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        h_output.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaEventDestroy(qkv_start));
    CUDA_CHECK(cudaEventDestroy(qkv_stop));
    CUDA_CHECK(cudaEventDestroy(transpose_start));
    CUDA_CHECK(cudaEventDestroy(transpose_stop));
    CUDA_CHECK(cudaEventDestroy(qkt_start));
    CUDA_CHECK(cudaEventDestroy(qkt_stop));
    CUDA_CHECK(cudaEventDestroy(scale_start));
    CUDA_CHECK(cudaEventDestroy(scale_stop));
    CUDA_CHECK(cudaEventDestroy(softmax_start));
    CUDA_CHECK(cudaEventDestroy(softmax_stop));
    CUDA_CHECK(cudaEventDestroy(av_start));
    CUDA_CHECK(cudaEventDestroy(av_stop));

    cudaFree(d_X);
    cudaFree(d_WQ);
    cudaFree(d_WK);
    cudaFree(d_WV);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);

    cudaFree(d_KT);
    cudaFree(d_scores);
    cudaFree(d_output);
}

int main()
{
    int device_count = 0;

    CUDA_CHECK(cudaGetDeviceCount(
        &device_count
    ));

    if (device_count == 0)
    {
        std::cerr << "No CUDA device found.\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties;

    CUDA_CHECK(cudaGetDeviceProperties(
        &properties,
        0
    ));

    std::cout
        << "CUDA Self-Attention Benchmark\n"
        << "GPU = "
        << properties.name
        << "\n"
        << "D = 128, d_k = 128, d_v = 128\n"
        << "Repetitions = 5\n\n";

    const int D = 128;
    const int dk = 128;
    const int dv = 128;

    const int sequence_lengths[] = {
        64,
        128,
        256,
        512,
        1024
    };

    const int repetitions = 5;

    for (int N : sequence_lengths)
    {
        std::vector<float> X(N * D);
        std::vector<float> WQ(D * dk);
        std::vector<float> WK(D * dk);
        std::vector<float> WV(D * dv);

        initialize_matrix(X, 100);
        initialize_matrix(WQ, 200);
        initialize_matrix(WK, 300);
        initialize_matrix(WV, 400);

        std::vector<float> cpu_output(N * dv);
        std::vector<float> gpu_output(N * dv);

        cpu_attention(
            X,
            WQ,
            WK,
            WV,
            cpu_output,
            N,
            D,
            dk,
            dv
        );

        float best_ms = INFINITY;

        float best_qkv_ms = 0.0f;
        float best_transpose_ms = 0.0f;
        float best_qkt_ms = 0.0f;
        float best_scale_ms = 0.0f;
        float best_softmax_ms = 0.0f;
        float best_av_ms = 0.0f;

        for (int r = 0; r < repetitions; ++r)
        {
            float elapsed_ms = 0.0f;
            float qkv_ms = 0.0f;
            float transpose_ms = 0.0f;
            float qkt_ms = 0.0f;
            float scale_ms = 0.0f;
            float softmax_ms = 0.0f;
            float av_ms = 0.0f;

            run_cuda_attention(
                X,
                WQ,
                WK,
                WV,
                gpu_output,
                N,
                D,
                dk,
                dv,
                elapsed_ms,
                qkv_ms,
                transpose_ms,
                qkt_ms,
                scale_ms,
                softmax_ms,
                av_ms
            );

            if (elapsed_ms < best_ms)
            {
                best_ms = elapsed_ms;

                best_qkv_ms = qkv_ms;
                best_transpose_ms = transpose_ms;
                best_qkt_ms = qkt_ms;
                best_scale_ms = scale_ms;
                best_softmax_ms = softmax_ms;
                best_av_ms = av_ms;
            }
        }

        float error =
            max_absolute_error(
                cpu_output,
                gpu_output
            );

        double flops =
            6.0 * N * D * dk +
            2.0 * N * N * dk +
            2.0 * N * N * dv;

        double gflops =
            (flops / 1.0e9) /
            (best_ms / 1000.0);

        auto print_stage =
            [](const char* name, float ms, float total)
        {
            double percent =
                (total > 0.0f)
                    ? (100.0 * ms / total)
                    : 0.0;

            std::cout
                << std::left
                << std::setw(12)
                << name
                << std::right
                << std::setw(12)
                << std::fixed
                << std::setprecision(4)
                << ms
                << " ms"
                << std::setw(10)
                << std::setprecision(2)
                << percent
                << " %"
                << "\n";
        };

        std::cout
            << "\nN = "
            << N
            << " | GFLOPS = "
            << std::fixed
            << std::setprecision(2)
            << gflops
            << " | Max error = "
            << std::scientific
            << error
            << std::fixed
            << "\n";

        std::cout
            << "------------------------------------\n"
            << std::left
            << std::setw(12)
            << "Stage"
            << std::right
            << std::setw(12)
            << "Time"
            << std::setw(11)
            << "% Total"
            << "\n"
            << "------------------------------------\n";

        print_stage("QKV", best_qkv_ms, best_ms);
        print_stage("Transpose", best_transpose_ms, best_ms);
        print_stage("QK^T", best_qkt_ms, best_ms);
        print_stage("Scale", best_scale_ms, best_ms);
        print_stage("Softmax", best_softmax_ms, best_ms);
        print_stage("AV", best_av_ms, best_ms);

        std::cout
            << "------------------------------------\n";

        print_stage("Total", best_ms, best_ms);
    }

    std::cout
        << "\nCUDA benchmark complete.\n";

    return 0;
}
