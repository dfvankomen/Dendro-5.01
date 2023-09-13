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





/*
  -----------------------------------------------------------------------
  Implementation of data compression algorithm based on the following paper:

  Author: Peter M. Williams
  Title: Image Compression for Neural Networks using Chebyshev Polynomials
  Journal: Artificial Neural Networks
  Publisher: North-Holland
  Year: 1992
  Pages: 1139-1142
  ISBN: 9780444894885
  DOI: https://doi.org/10.1016/B978-0-444-89488-5.50065-8
  Link: https://www.sciencedirect.com/science/article/pii/B9780444894885500658
  
  Abstract: This paper proposes an algorithm for data compression of multi-dimensional images, 
  prior to input into neural network feature detectors, using Chebyshev approximation. 
  Illustrations are given of the fidelity obtainable for geophysical data.
  -----------------------------------------------------------------------
*/

#ifndef _CHEBYSHEVALGORITHMS_H_
#define _CHEBYSHEVALGORITHMS_H_

// Disables Eigen's memory alignment which could lead to extra memory padding.
#define EIGEN_DONT_ALIGN
#include <Eigen/Dense> // Include this header for matrix and vector operations
#include <iostream>
#include <utility>
#include <unordered_map>


namespace ChebyshevAlgorithms {

	double chebyshevT(int n, double x);

	/**
	 * N, Q, and S must be less than or equal to x, y, and z respectively. 
	 *
	 * @param originalMatrix A three dimensional array represented in one dimension through row majoring order
	 * @param x The number of elements in the x dimension
	 * @param y The number of elements in the y dimension
	 * @param z The number of elements in the z dimension
	 * @param N The number of the coefficients in the x dimension
	 * @param Q The number of the coefficients in the y dimension
	 * @param S The number of the coefficients in the z dimension
	 *
	 * @return A pointer to the byte stream containing the compressed data.
	 */
	unsigned char* compressMatrix(double*& originalMatrix, int x, int y, int z, int N, int Q, int S, int& bufferSize);
	double* decompressMatrix(unsigned char*& buffer, int bufferSize);

}
#endif // _CHEBYSHEVALGORITHMS_H_



#ifndef FFT_ALGORITHMS_H
#define FFT_ALGORITHMS_H

#include <vector>
#include <complex>
#include <iostream>
#include <cmath>
#include <fftw3.h>
#include <algorithm>
#include <cstring>  

namespace FFTAlgorithms {
    typedef std::vector<std::complex<double>> ComplexVec;
    typedef std::pair<double, size_t> MagnitudeIndexPair;

    // Function to compare magnitudes for sorting
    bool compareMagnitude(const MagnitudeIndexPair& a, const MagnitudeIndexPair& b);

    unsigned char* compressMatrix(double* originalMatrix, int x, int y, int z, double compressionRatio, int& size);
    double* decompressMatrix(unsigned char* buffer, int bufferSize);
    unsigned char* compressMatrix(fftw_plan p, double* originalMatrix, int x, int y, int z, double compressionRatio, int& size);
    double* decompressMatrix(fftw_plan q, unsigned char* byteStream, int byteStreamSize);
}

#endif // FFT_ALGORITHMS_H



#ifndef _ZFPALGORITHMS_H_
#define _ZFPALGORITHMS_H_

#include <zfp.h>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>
#include <fftw3.h>
#include <numeric>
#include <cmath>

namespace ZFPAlgorithms {

    unsigned char* compressMatrix(double* originalData, int x, int y, int z, double rate, int& size);
    double* decompressMatrix(unsigned char* buffer, int bufferSize);
}

#endif // _ZFPALGORITHMS_H_ 

