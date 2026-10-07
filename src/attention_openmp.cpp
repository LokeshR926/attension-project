#include "attention_openmp.hpp"

#include "matrix_ops.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>


Matrix matmul_openmp(
    const Matrix& A,
    const Matrix& B)
{
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument(
            "Matrix dimensions incompatible for multiplication"
        );
    }

    Matrix C(A.rows(), B.cols());

    /*
     * Matrix multiplication:
     *
     * A: M x K
     * B: K x N
     * C: M x N
     *
     * Each output row is independent.
     *
     * OpenMP parallelizes the outer row loop.
     *
     * Loop order:
     *
     *     i -> k -> j
     *
     * This preserves the same cache-friendly access pattern
     * used by the sequential implementation.
     */

    #pragma omp parallel for schedule(static)
    for (long long i = 0;
         i < static_cast<long long>(A.rows());
         ++i)
    {
        for (std::size_t k = 0;
             k < A.cols();
             ++k)
        {
            float a = A(
                static_cast<std::size_t>(i),
                k
            );

            for (std::size_t j = 0;
                 j < B.cols();
                 ++j)
            {
                C(
                    static_cast<std::size_t>(i),
                    j
                ) += a * B(k, j);
            }
        }
    }

    return C;
}


Matrix softmax_openmp(const Matrix& A)
{
    if (A.rows() == 0 || A.cols() == 0)
    {
        throw std::invalid_argument(
            "Softmax requires a non-empty matrix"
        );
    }

    Matrix result(A.rows(), A.cols());

    /*
     * Every row of softmax is independent.
     *
     * Therefore each row can be processed by a
     * different OpenMP thread.
     */

    #pragma omp parallel for schedule(static)
    for (long long i = 0;
         i < static_cast<long long>(A.rows());
         ++i)
    {
        std::size_t row =
            static_cast<std::size_t>(i);

        // Find row maximum for numerical stability.
        float max_value = A(row, 0);

        for (std::size_t j = 1;
             j < A.cols();
             ++j)
        {
            max_value =
                std::max(max_value, A(row, j));
        }

        // Calculate exponentials.
        float sum = 0.0f;

        for (std::size_t j = 0;
             j < A.cols();
             ++j)
        {
            float value =
                std::exp(A(row, j) - max_value);

            result(row, j) = value;
            sum += value;
        }

        // Normalize the row.
        for (std::size_t j = 0;
             j < A.cols();
             ++j)
        {
            result(row, j) /= sum;
        }
    }

    return result;
}


Matrix scaled_dot_product_attention_openmp(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V)
{
    /*
     * Q: N x d_k
     * K: N x d_k
     * V: N x d_v
     *
     * Attention:
     *
     *     softmax(QK^T / sqrt(d_k))V
     */

    if (Q.rows() == 0 ||
        Q.cols() == 0 ||
        K.rows() == 0 ||
        K.cols() == 0 ||
        V.rows() == 0 ||
        V.cols() == 0)
    {
        throw std::invalid_argument(
            "Attention requires non-empty matrices"
        );
    }

    if (Q.cols() != K.cols())
    {
        throw std::invalid_argument(
            "Q and K must have the same feature dimension"
        );
    }

    if (K.rows() != V.rows())
    {
        throw std::invalid_argument(
            "K and V must have the same sequence length"
        );
    }

    // K^T
    Matrix K_T = transpose(K);

    // QK^T
    Matrix scores =
        matmul_openmp(Q, K_T);

    // Scale scores by 1 / sqrt(d_k).
    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(Q.cols())
        );

    #pragma omp parallel for collapse(2) schedule(static)
    for (long long i = 0;
         i < static_cast<long long>(scores.rows());
         ++i)
    {
        for (long long j = 0;
             j < static_cast<long long>(scores.cols());
             ++j)
        {
            scores(
                static_cast<std::size_t>(i),
                static_cast<std::size_t>(j)
            ) *= scale;
        }
    }

    // Row-wise softmax.
    Matrix weights =
        softmax_openmp(scores);

    // Attention weights multiplied by V.
    Matrix output =
        matmul_openmp(weights, V);

    return output;
}


Matrix self_attention_openmp(
    const Matrix& X,
    const Matrix& WQ,
    const Matrix& WK,
    const Matrix& WV)
{
    /*
     * X:  N x D
     *
     * WQ: D x d_k
     * WK: D x d_k
     * WV: D x d_v
     *
     * Q = XWQ
     * K = XWK
     * V = XWV
     */

    if (X.rows() == 0 || X.cols() == 0)
    {
        throw std::invalid_argument(
            "Input X must be non-empty"
        );
    }

    if (WQ.rows() != X.cols() ||
        WK.rows() != X.cols() ||
        WV.rows() != X.cols())
    {
        throw std::invalid_argument(
            "Projection matrices have incompatible dimensions"
        );
    }

    Matrix Q =
        matmul_openmp(X, WQ);

    Matrix K =
        matmul_openmp(X, WK);

    Matrix V =
        matmul_openmp(X, WV);

    return scaled_dot_product_attention_openmp(
        Q,
        K,
        V
    );
}
