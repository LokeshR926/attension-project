#include "attention.hpp"

#include "matrix_ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Matrix softmax(const Matrix& A)
{
    if (A.rows() == 0 || A.cols() == 0)
    {
        throw std::invalid_argument(
            "Softmax requires a non-empty matrix"
        );
    }

    Matrix result(A.rows(), A.cols());

    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        // Find the maximum value in the row.
        // Subtracting it before exp() improves numerical stability.
        float max_value = A(i, 0);

        for (std::size_t j = 1; j < A.cols(); ++j)
        {
            max_value = std::max(max_value, A(i, j));
        }

        // Compute exponentials and their sum.
        float sum = 0.0f;

        for (std::size_t j = 0; j < A.cols(); ++j)
        {
            float value = std::exp(A(i, j) - max_value);

            result(i, j) = value;
            sum += value;
        }

        // Normalize the row.
        for (std::size_t j = 0; j < A.cols(); ++j)
        {
            result(i, j) /= sum;
        }
    }

    return result;
}


Matrix scaled_dot_product_attention(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V)
{
    /*
     * Q: N x d_k
     * K: N x d_k
     * V: N x d_v
     *
     * K^T: d_k x N
     *
     * QK^T: N x N
     *
     * Attention:
     *
     * softmax(QK^T / sqrt(d_k))V
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
    Matrix scores = matmul(Q, K_T);

    // Scale by sqrt(d_k).
    const float scale =
        1.0f / std::sqrt(static_cast<float>(Q.cols()));

    for (std::size_t i = 0; i < scores.rows(); ++i)
    {
        for (std::size_t j = 0; j < scores.cols(); ++j)
        {
            scores(i, j) *= scale;
        }
    }

    // Row-wise softmax.
    Matrix weights = softmax(scores);

    // Final attention output.
    Matrix output = matmul(weights, V);

    return output;
}


Matrix self_attention(
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

    Matrix Q = matmul(X, WQ);
    Matrix K = matmul(X, WK);
    Matrix V = matmul(X, WV);

    return scaled_dot_product_attention(Q, K, V);
}
