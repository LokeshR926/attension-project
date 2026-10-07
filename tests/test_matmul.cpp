#include "matrix_ops.hpp"

#include <iostream>

int main()
{
    Matrix A(2, 3);
    Matrix B(3, 2);

    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;

    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    B(0, 0) = 7;
    B(0, 1) = 8;

    B(1, 0) = 9;
    B(1, 1) = 10;

    B(2, 0) = 11;
    B(2, 1) = 12;

    Matrix C = matmul(A, B);
    Matrix T = transpose(A);

    std::cout << "A:\n";
    A.print();

    std::cout << "\nB:\n";
    B.print();

    std::cout << "\nC = A * B:\n";
    C.print();

    std::cout << "\nTranspose of A:\n";
    T.print();

    return 0;
}
