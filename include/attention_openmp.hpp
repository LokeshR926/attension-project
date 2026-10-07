#ifndef ATTENTION_OPENMP_HPP
#define ATTENTION_OPENMP_HPP

#include "matrix.hpp"

// OpenMP-parallel matrix multiplication.
Matrix matmul_openmp(const Matrix& A, const Matrix& B);

// OpenMP-parallel row-wise softmax.
Matrix softmax_openmp(const Matrix& A);

// OpenMP scaled dot-product attention.
Matrix scaled_dot_product_attention_openmp(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V
);

// OpenMP self-attention.
Matrix self_attention_openmp(
    const Matrix& X,
    const Matrix& WQ,
    const Matrix& WK,
    const Matrix& WV
);

#endif
