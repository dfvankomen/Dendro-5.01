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
#include <utility>
#include <cmath>
#include <vector>
#include <Eigen/SVD>
#include <cmath>


namespace SVDAlgorithms {

    /**
    * Compress a 3D matrix using Singular Value Decomposition (SVD).
    *
    * In a 3x7x7 matrix, k can be anywhere from 1 - 21
    *
    * In our tests, we found setting k = 2 to be the most consistant overall. It provided
    * near lossless compression with 3:1 compression ratios. This algorithm was
    * much slower than the others. Roughly 100 times slower than ZFP and 10 times slower than FFT
    * compression algorithms.
    *
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

    std::pair<int, int> findSquareLikeDimensions(int n);

    /**
    *
    * @param cmp_str Intesity of compression to be performed. Valid values are 1 through n/2.
    *                If cmp_str is set too high, then cmp_str will be set to the highest cmp_str allowed.
    *                Recemmended to slowly test values starting at 3.
    */
	unsigned char* compressMatrix1d(double*& originalMatrix, const int n, const int cmp_str, int& size);
    double* decompressMatrix1d(unsigned char*& buffer, int bufferSize);
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
     * To compress the data,
     * choose N such that N < x,
     * choose Q such that Q < y, and/or
     * choose S such that S < z.
     *
     * All three values can be modified at once. The closer N, Q, and S are to
     * 0, the higher the compression ratio. Through testing we found this
     * chebyshev compression algorithm to be enreliable as it produced single
     * errors greater than 0.001. Best compression values were simply reducing N < x.
     *
     * @param originalMatrix A three dimensional array represented in one dimension through row majoring order
     * @param x The number of elements in the x dimension
     * @param y The number of elements in the y dimension
     * @param z The number of elements in the z dimension
     * @param N The number of the coefficients in the x dimension. Must be less than x.
     * @param Q The number of the coefficients in the y dimension. Must be less than y.
     * @param S The number of the coefficients in the z dimension. Must be less than z.
     *
     * @return A pointer to the byte stream containing the compressed data.
     */
    unsigned char* compressMatrix(double*& originalMatrix, int x, int y, int z, int N, int Q, int S, int& bufferSize);
    double* decompressMatrix(unsigned char*& buffer, int bufferSize);

}
#endif // _CHEBYSHEVALGORITHMS_H_

#if 0

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

    /**
     * To compress the data choose compressionRatio to be anywhere from 0.0 through 1.0.
     * In our testing, we found the FFT compression algorithm to be highly inconsitent
     * for all values. The best overall values were achieved when compressionRatio
     * was set to 0.10 through 0.45.
     * In a 3x7x7 matrix, compressionRatio above 0.45 offers no compression and makes the
     * compressed data larger than the original data.
     *
     * @param compressionRatio Valid values 0.0 through 1.0.
    */
    unsigned char* compressMatrix(double* originalMatrix, int x, int y, int z, double compressionRatio, int& size);
    double* decompressMatrix(unsigned char* buffer, int bufferSize);

    unsigned char* compressMatrix1D(double* originalMatrix, int n, double threshold, int& size);
    double* decompressMatrix1D(unsigned char* buffer, int bufferSize);
}

#endif // FFT_ALGORITHMS_H

#endif

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

    /**
     * We used ZFP's fixde-rate mode for this compression algoithm in 3x7x7 matrices.
     * For high level documentation of the ZFP functios used:
     * https://zfp.readthedocs.io/en/release0.5.4/high-level-api.html#
     * @param rate Valid values are positive doubles.
     *             The rate is defined as maxbits / 4^d, where d=3 for this implementation.
     *             Through testing we found values 15.0 - 35.0 to offer the most consistent compression.
     *             Values closer to 35 have minimal compression rates while values closer to 0 have much
     *             higher compression rates at the cost of precision.
     *             Values above 35.0 offer no compression and make the compressed data larger than the original data.
     *
    */
    unsigned char* compressMatrix(double* originalData, int x, int y, int z, double rate, int& size);
    double* decompressMatrix(unsigned char* buffer, int bufferSize);

    /**
     * We used ZFP's fixde-rate mode for this compression algoithm in 3x7x7 matrices.
     * For high level documentation of the ZFP functios used:
     * https://zfp.readthedocs.io/en/release0.5.4/high-level-api.html#
     * @param rate Valid values are positive doubles.
     *             The rate is defined as maxbits / 4^d, where d=1 for this algorithm.
     *             Therefore, the rate should be bounded by 4/4 -> 64/4 or 1 -> 16
     *
    */
    unsigned char* compressMatrix1D(double* originalData, int n, double rate, int& size);
    double* decompressMatrix1D(unsigned char* buffer, int bufferSize);

    // pre-allocated memory version
    void decompressMatrix1D(unsigned char* buffer, int bufferSize, double* outBuff);
}

#endif // _ZFPALGORITHMS_H_ 

#ifndef _BLOSCOMPRESSION_H_
#define _BLOSCCOMPRESSION_H_

#include <blosc.h>
#include <iostream>

/**
 * Namespace BLOSCCompression
 * 
 * Provides functionalities for compressing and decompressing data using the Blosc library.
 * For more information about blosc: 
 * https://github.com/dolphinking/blosc
 * https://github.com/Blosc/c-blosc/blob/main/blosc/blosc.h
 * https://www.blosc.org/
 * 
 * Example: 
 * 
int main() {
    int n; // number of elements in your data
    double* originalData = your data;
    int compressionLevel = 9; // Choose the compression level (1-9, where 9 is highest compression)

    int bytestreamSize;
    blosc_init();
    unsigned char* bytestream = BLOSCCompression::compressData("zstd" ,compressionLevel, n, originalMatrix, bytestreamSize);
    blosc_destroy();

    delete[] originalMatrix;

    // ...
    // Pass bytestream and bytestreamSize to a different machine 
    // ...


    blosc_init();
    double* decompressedData = BLOSCCompression::decompressData(bytestream, bytestreamSize);
    blosc_destroy();

    delete[] decompressedData;
    return 0;
}
 */
namespace BLOSCCompression {

    /**
    * Compresses a block of data using the Blosc library.
    * This function compresses a given array of doubles using the specified Blosc compressor and compression level.
    *
    * Before calling this function, ensure blosc_init() has been called to initialize the Blosc library.
    * After using the compressed data, blosc_destroy() should be called for proper cleanup.
    *
    * @param blosc_compressor The compression algorithm to be used. Must be one of "blosclz", "lz4", "lz4hc", "zlib", or "zstd".
    * @param clevel Choose the compression level (1-9, where 9 is highest compression)
    * @param n The number of elements in the original data array.
    * @param originalData Pointer to the original array of doubles to be compressed.
    * @param byteStreamSize Reference to an integer where the size of the resulting bytestream will be stored.
    * @return Pointer to the compressed data bytestream.
    * @throw std::runtime_error if compression fails.
    */
    unsigned char* compressData(const char* blosc_compressor, int clevel, int n, double* originalData, int& byteStreamSize);

    /**
    * Decompresses a bytestream using the Blosc library.
    * It decompresses a bytestream that was created by the compressData function. 
    * 
    * Before calling this function, ensure blosc_init() has been called to initialize the Blosc library.
    * After using the compressed data, blosc_destroy() should be called for proper cleanup.
    *
    * @param byteStream Pointer to the bytestream containing the compressed data.
    * @param byteStreamSize The size of the bytestream.
    * @return Pointer to the decompressed data array.
    * @throw std::runtime_error if decompression fails or if input is invalid.
    */
    double* decompressData(unsigned char* byteStream, int byteStreamSize);
}

#endif // _BLOSCCOMPRESSION_H_ 

