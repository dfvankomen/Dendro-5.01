/**
 * @file SVDAlgorithms.h
 * @brief This header file provides SVD-based data compression and decompression algorithms.
 *
 * No padding.
 *
 */

#ifndef _SVDALGORITHMS_H_
#define _SVDALGORITHMS_H_

 // Disables Eigen's memory alignment which could lead to extra memory padding.
#define EIGEN_DONT_ALIGN
#include <Eigen/Dense> 
#include <iostream>


namespace SVDAlgorithms {

    /**
    * Compress a 3D matrix using Singular Value Decomposition (SVD).
    *
    * In a 3x7x7 matrix, k can be anywhere from 1 - 21
    *
    * @param originalMatrix: A pointer to the 3D matrix to be compressed. It must be stored in a 1D array in column-major order.
    * @param x: The size of the first dimension of the original matrix.
    * @param y: The size of the second dimension of the original matrix.
    * @param z: The size of the third dimension of the original matrix.
    * @param k: The number of singular values to keep for SVD. It must be less than or equal to the smallest dimension of the 2D reshaped matrix (min(x * y, z)).
    * @param buffer_size: Output parameter that will hold the size of the resulting byte stream.
    *
    * @return: A pointer to a byte stream containing the compressed matrix and meta-information.
    */
    unsigned char* compressMatrix(double*& originalMatrix, const int x, const int y, const int z, const int k, int& size);

    double* decompressMatrix(unsigned char*& buffer, int bufferSize);

}
#endif // _SVDALGORITHMS_H_

