#include "attention.hpp"
#include "attention_openmp.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <omp.h>


bool approximately_equal(
    float a,
    float b,
    float tolerance = 1e-4f)
{
    return std::fabs(a - b) <= tolerance;
}


float max_absolute_error(
    const Matrix& A,
    const Matrix& B)
{
    if (A.rows() != B.rows() ||
        A.cols() != B.cols())
    {
        return INFINITY;
    }

    float max_error = 0.0f;

    for (std::size_t i = 0;
         i < A.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < A.cols();
             ++j)
        {
            float error =
                std::fabs(A(i, j) - B(i, j));

            if (error > max_error)
            {
                max_error = error;
            }
        }
    }

    return max_error;
}


void fill_random(
    Matrix& A,
    unsigned int seed)
{
    std::mt19937 generator(seed);

    std::uniform_real_distribution<float>
        distribution(-1.0f, 1.0f);

    for (std::size_t i = 0;
         i < A.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < A.cols();
             ++j)
        {
            A(i, j) =
                distribution(generator);
        }
    }
}


int main()
{
    /*
     * Test configuration.
     *
     * We intentionally use a non-trivial
     * random input instead of only identity
     * matrices.
     */

    const std::size_t N = 64;
    const std::size_t D = 32;
    const std::size_t d_k = 32;
    const std::size_t d_v = 32;

    Matrix X(N, D);
    Matrix WQ(D, d_k);
    Matrix WK(D, d_k);
    Matrix WV(D, d_v);

    fill_random(X, 100);
    fill_random(WQ, 200);
    fill_random(WK, 300);
    fill_random(WV, 400);


    /*
     * Sequential reference.
     */

    Matrix sequential =
        self_attention(
            X,
            WQ,
            WK,
            WV
        );


    /*
     * OpenMP implementation.
     */

    Matrix parallel =
        self_attention_openmp(
            X,
            WQ,
            WK,
            WV
        );


    /*
     * Compare the complete output matrices.
     */

    float error =
        max_absolute_error(
            sequential,
            parallel
        );


    std::cout
        << "OpenMP correctness test\n";

    std::cout
        << "Sequence length: "
        << N
        << "\n";

    std::cout
        << "Input dimension: "
        << D
        << "\n";

    std::cout
        << "Attention dimension: "
        << d_k
        << "\n";

    std::cout
        << "Value dimension: "
        << d_v
        << "\n";

    std::cout
        << "OpenMP threads: "
        << omp_get_max_threads()
        << "\n";

    std::cout
        << "Maximum absolute error: "
        << error
        << "\n";


    /*
     * Floating-point operations may execute in
     * a different order in parallel execution.
     *
     * A small numerical difference is therefore
     * acceptable.
     */

    const float tolerance = 1e-4f;

    if (error > tolerance)
    {
        std::cerr
            << "\nOpenMP correctness test FAILED\n";

        return 1;
    }


    /*
     * Also verify output dimensions.
     */

    if (parallel.rows() != N ||
        parallel.cols() != d_v)
    {
        std::cerr
            << "\nOutput dimension test FAILED\n";

        return 1;
    }


    std::cout
        << "\nOpenMP correctness test PASSED\n";

    return 0;
}
