#include "attention.hpp"

#include <cmath>
#include <iostream>

bool approximately_equal(
    float a,
    float b,
    float tolerance = 1e-4f)
{
    return std::fabs(a - b) <= tolerance;
}

int main()
{
    /*
     * X: 2 x 2
     *
     * WQ, WK, WV: 2 x 2
     *
     * We use identity projection matrices so that:
     *
     * Q = X
     * K = X
     * V = X
     *
     * This makes the expected attention result easy to verify.
     */

    Matrix X(2, 2);

    X(0, 0) = 1.0f;
    X(0, 1) = 0.0f;

    X(1, 0) = 0.0f;
    X(1, 1) = 1.0f;

    Matrix WQ(2, 2);
    Matrix WK(2, 2);
    Matrix WV(2, 2);

    // Identity matrices.
    WQ(0, 0) = 1.0f;
    WQ(1, 1) = 1.0f;

    WK(0, 0) = 1.0f;
    WK(1, 1) = 1.0f;

    WV(0, 0) = 1.0f;
    WV(1, 1) = 1.0f;

    Matrix output =
        self_attention(X, WQ, WK, WV);

    std::cout << "Input X:\n";
    X.print();

    std::cout << "\nSelf-attention output:\n";
    output.print();

    // Expected result:
    //
    // Q = K = V = I
    //
    // QK^T / sqrt(2)
    //
    // = [ 1/sqrt(2)      0       ]
    //   [     0       1/sqrt(2)  ]
    //
    // Each row softmax is approximately:
    //
    // [0.669762, 0.330238]
    //
    // Therefore:
    //
    // [0.669762, 0.330238]
    // [0.330238, 0.669762]

    const float expected00 = 0.669762f;
    const float expected01 = 0.330238f;
    const float expected10 = 0.330238f;
    const float expected11 = 0.669762f;

    if (!approximately_equal(
            output(0, 0), expected00) ||
        !approximately_equal(
            output(0, 1), expected01) ||
        !approximately_equal(
            output(1, 0), expected10) ||
        !approximately_equal(
            output(1, 1), expected11))
    {
        std::cerr << "\nSelf-attention value test FAILED\n";
        return 1;
    }

    if (output.rows() != X.rows() ||
        output.cols() != WV.cols())
    {
        std::cerr << "\nSelf-attention dimension test FAILED\n";
        return 1;
    }

    std::cout << "\nSelf-attention test PASSED\n";

    return 0;
}
