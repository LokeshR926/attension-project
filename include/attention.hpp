#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "matrix.hpp"

// Apply numerically stable row-wise softmax.
Matrix softmax(const Matrix& A);

// Compute scaled dot-product attention.
//
// Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
Matrix scaled_dot_product_attention(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V
);

// Compute self-attention from input X.
//
// Q = XWQ
// K = XWK
// V = XWV
Matrix self_attention(
    const Matrix& X,
    const Matrix& WQ,
    const Matrix& WK,
    const Matrix& WV
);

#endif
