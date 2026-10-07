#include "matrix.hpp"

#include <iostream>

int main()
{
    Matrix A(3, 4);

    float value = 1.0f;

    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        for (std::size_t j = 0; j < A.cols(); ++j)
        {
            A(i, j) = value;
            value += 1.0f;
        }
    }

    std::cout << "Matrix A:\n";
    A.print();

    std::cout << "\nDimensions: "
              << A.rows()
              << " x "
              << A.cols()
              << "\n";

    std::cout << "\nA(2,1) = "
              << A(2, 1)
              << "\n";

    return 0;
}
