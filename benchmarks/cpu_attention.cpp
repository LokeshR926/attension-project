#include "attention.hpp"
#include "matrix_ops.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

using Clock = std::chrono::high_resolution_clock;

void fill_random(Matrix& A, unsigned int seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        for (std::size_t j = 0; j < A.cols(); ++j)
        {
            A(i, j) = distribution(generator);
        }
    }
}

double milliseconds(
    const Clock::time_point& start,
    const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

int main()
{
    /*
     * Benchmark configuration.
     *
     * N = sequence length
     * D = model/input dimension
     * d_k = attention dimension
     * d_v = value dimension
     */

    const std::size_t D = 128;
    const std::size_t d_k = 128;
    const std::size_t d_v = 128;

    const std::size_t sequence_lengths[] =
    {
        64,
        128,
        256,
        512,
        1024
    };

    const int repetitions = 5;

    std::ofstream output(
        "results/cpu/self_attention.csv"
    );

    if (!output)
    {
        std::cerr
            << "Failed to create results file\n";

        return 1;
    }

    output
        << "N,D,d_k,d_v,"
        << "qkv_ms,"
        << "qkt_ms,"
        << "scale_ms,"
        << "softmax_ms,"
        << "av_ms,"
        << "total_ms,"
        << "gflops_equivalent\n";

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "\nCPU Self-Attention Benchmark\n"
        << "D = " << D
        << ", d_k = " << d_k
        << ", d_v = " << d_v
        << "\n"
        << "Repetitions = " << repetitions
        << "\n\n";

    for (std::size_t N : sequence_lengths)
    {
        Matrix X(N, D);

        Matrix WQ(D, d_k);
        Matrix WK(D, d_k);
        Matrix WV(D, d_v);

        fill_random(X, 100 + N);
        fill_random(WQ, 200 + N);
        fill_random(WK, 300 + N);
        fill_random(WV, 400 + N);

        double qkv_total = 0.0;
        double qkt_total = 0.0;
        double scale_total = 0.0;
        double softmax_total = 0.0;
        double av_total = 0.0;
        double total = 0.0;

        for (int repetition = 0;
             repetition < repetitions;
             ++repetition)
        {
            auto total_start = Clock::now();

            // ---------------------------------
            // QKV projection
            // ---------------------------------

            auto qkv_start = Clock::now();

            Matrix Q = matmul(X, WQ);
            Matrix K = matmul(X, WK);
            Matrix V = matmul(X, WV);

            auto qkv_end = Clock::now();

            // ---------------------------------
            // QK^T
            // ---------------------------------

            auto qkt_start = Clock::now();

            Matrix K_T = transpose(K);
            Matrix scores = matmul(Q, K_T);

            auto qkt_end = Clock::now();

            // ---------------------------------
            // Scaling
            // ---------------------------------

            auto scale_start = Clock::now();

            const float scale =
                1.0f /
                std::sqrt(
                    static_cast<float>(d_k)
                );

            for (std::size_t i = 0;
                 i < scores.rows();
                 ++i)
            {
                for (std::size_t j = 0;
                     j < scores.cols();
                     ++j)
                {
                    scores(i, j) *= scale;
                }
            }

            auto scale_end = Clock::now();

            // ---------------------------------
            // Softmax
            // ---------------------------------

            auto softmax_start = Clock::now();

            Matrix weights = softmax(scores);

            auto softmax_end = Clock::now();

            // ---------------------------------
            // Attention × V
            // ---------------------------------

            auto av_start = Clock::now();

            Matrix result = matmul(weights, V);

            auto av_end = Clock::now();

            auto total_end = Clock::now();

            qkv_total += milliseconds(
                qkv_start, qkv_end);

            qkt_total += milliseconds(
                qkt_start, qkt_end);

            scale_total += milliseconds(
                scale_start, scale_end);

            softmax_total += milliseconds(
                softmax_start, softmax_end);

            av_total += milliseconds(
                av_start, av_end);

            total += milliseconds(
                total_start, total_end);

            // Prevent compiler from treating result
            // as completely unused.
            volatile float checksum =
                result(0, 0);

            (void)checksum;
        }

        double qkv_ms =
            qkv_total / repetitions;

        double qkt_ms =
            qkt_total / repetitions;

        double scale_ms =
            scale_total / repetitions;

        double softmax_ms =
            softmax_total / repetitions;

        double av_ms =
            av_total / repetitions;

        double total_ms =
            total / repetitions;

        /*
         * Approximate floating-point operations:
         *
         * QKV:
         *   3 × 2NDd
         *
         * QK^T:
         *   2N²d
         *
         * AV:
         *   2N²d_v
         *
         * This is an equivalent FLOP estimate,
         * not a hardware-counter measurement.
         */

        double qkv_flops =
            6.0 *
            static_cast<double>(N) *
            D *
            d_k;

        double qkt_flops =
            2.0 *
            static_cast<double>(N) *
            N *
            d_k;

        double av_flops =
            2.0 *
            static_cast<double>(N) *
            N *
            d_v;

        double total_flops =
            qkv_flops +
            qkt_flops +
            av_flops;

        double gflops =
            (total_flops / total_ms) /
            1.0e6;

        std::cout
            << "N = " << std::setw(4) << N
            << " | QKV = "
            << std::setw(10) << qkv_ms
            << " ms | QK^T = "
            << std::setw(10) << qkt_ms
            << " ms | Softmax = "
            << std::setw(10) << softmax_ms
            << " ms | AV = "
            << std::setw(10) << av_ms
            << " ms | Total = "
            << std::setw(10) << total_ms
            << " ms\n";

        output
            << N << ","
            << D << ","
            << d_k << ","
            << d_v << ","
            << qkv_ms << ","
            << qkt_ms << ","
            << scale_ms << ","
            << softmax_ms << ","
            << av_ms << ","
            << total_ms << ","
            << gflops << "\n";
    }

    output.close();

    std::cout
        << "\nBenchmark complete.\n"
        << "Results saved to:\n"
        << "results/cpu/self_attention.csv\n";

    return 0;
}
