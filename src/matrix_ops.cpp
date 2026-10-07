#include "matrix_ops.hpp"

#include <stdexcept>

Matrix matmul(const Matrix& A, const Matrix& B)
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
     * C = A * B
     *
     * A: M x K
     * B: K x N
     * C: M x N
     *
     * Loop order: i -> k -> j
     *
     * This gives contiguous access to B[k][j].
     */

    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        for (std::size_t k = 0; k < A.cols(); ++k)
        {
            float a = A(i, k);

            for (std::size_t j = 0; j < B.cols(); ++j)
            {
                C(i, j) += a * B(k, j);
            }
        }
    }

    return C;
}

Matrix transpose(const Matrix& A)
{
    Matrix T(A.cols(), A.rows());

    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        for (std::size_t j = 0; j < A.cols(); ++j)
        {
            T(j, i) = A(i, j);
        }
    }

    return T;
}
