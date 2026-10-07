#include "matrix.hpp"

#include <iostream>
#include <stdexcept>

Matrix::Matrix()
    : rows_(0),
      cols_(0)
{
}

Matrix::Matrix(std::size_t rows,
               std::size_t cols)
    : rows_(rows),
      cols_(cols),
      data_(rows * cols, 0.0f)
{
}

std::size_t Matrix::rows() const
{
    return rows_;
}

std::size_t Matrix::cols() const
{
    return cols_;
}

float& Matrix::operator()(std::size_t row,
                          std::size_t col)
{
    if (row >= rows_ || col >= cols_)
    {
        throw std::out_of_range(
            "Matrix index out of range"
        );
    }

    return data_[row * cols_ + col];
}

const float& Matrix::operator()(std::size_t row,
                                std::size_t col) const
{
    if (row >= rows_ || col >= cols_)
    {
        throw std::out_of_range(
            "Matrix index out of range"
        );
    }

    return data_[row * cols_ + col];
}

void Matrix::fill(float value)
{
    for (float& element : data_)
    {
        element = value;
    }
}

const std::vector<float>& Matrix::data() const
{
    return data_;
}

std::vector<float>& Matrix::data()
{
    return data_;
}

void Matrix::print() const
{
    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < cols_; ++j)
        {
            std::cout << (*this)(i, j) << " ";
        }

        std::cout << "\n";
    }
}
