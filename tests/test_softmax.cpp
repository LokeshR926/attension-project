#include "attention.hpp"

#include <cmath>
#include <iostream>

int main()
{
    Matrix A(2, 3);

    A(0, 0) = 1.0f;
    A(0, 1) = 2.0f;
    A(0, 2) = 3.0f;

    A(1, 0) = 1000.0f;
    A(1, 1) = 1001.0f;
    A(1, 2) = 1002.0f;

    Matrix S = softmax(A);

    std::cout << "Input:\n";
    A.print();

    std::cout << "\nSoftmax:\n";
    S.print();

    for (std::size_t i = 0; i < S.rows(); ++i)
    {
        float sum = 0.0f;

        for (std::size_t j = 0; j < S.cols(); ++j)
        {
            sum += S(i, j);
        }

        std::cout << "\nRow " << i
                  << " sum = " << sum << "\n";

        if (std::fabs(sum - 1.0f) > 1e-5f)
        {
            std::cerr << "Softmax test FAILED\n";
            return 1;
        }
    }

    std::cout << "\nSoftmax test PASSED\n";

    return 0;
}
