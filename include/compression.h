/**
 * @file SVDAlgorithms.h
 * @brief This header file provides SVD-based data compression and decompression
 * algorithms.
 *
 * No padding.
 *
 */

#pragma once

#include "lapac.h"

// Disables Eigen's memory alignment which could lead to extra memory padding.
#include <cstdint>
#include <cstdlib>
#define EIGEN_DONT_ALIGN
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

namespace SVDAlgorithms {

/**
 * Compress a 3D matrix using Singular Value Decomposition (SVD).
 *
 * In a 3x7x7 matrix, k can be anywhere from 1 - 21
 *
 * In our tests, we found setting k = 2 to be the most consistant overall. It
 * provided near lossless compression with 3:1 compression ratios. This
 * algorithm was much slower than the others. Roughly 100 times slower than ZFP
 * and 10 times slower than FFT compression algorithms.
 *
 *
 * @param originalMatrix: A pointer to the 3D matrix to be compressed. It must
 * be stored in a 1D array in column-major order.
 * @param x: The size of the first dimension of the original matrix.
 * @param y: The size of the second dimension of the original matrix.
 * @param z: The size of the third dimension of the original matrix.
 * @param k: The number of singular values to keep for SVD. It must be less than
 * or equal to the smallest dimension of the 2D reshaped matrix (min(x * y, z)).
 * @param buffer_size: Output parameter that will hold the size of the resulting
 * byte stream.
 *
 * @return: A pointer to a byte stream containing the compressed matrix and
 * meta-information.
 */
unsigned char* compressMatrix(double*& originalMatrix, const int x, const int y,
                              const int z, const int k, int& size);
double* decompressMatrix(unsigned char*& buffer, int bufferSize);

std::pair<int, int> findSquareLikeDimensions(int n);

/**
 *
 * @param cmp_str Intesity of compression to be performed. Valid values are 1
 * through n/2. If cmp_str is set too high, then cmp_str will be set to the
 * highest cmp_str allowed. Recemmended to slowly test values starting at 3.
 */
unsigned char* compressMatrix1d(double*& originalMatrix, const int n,
                                const int cmp_str, int& size);
double* decompressMatrix1d(unsigned char*& buffer, int bufferSize);
}  // namespace SVDAlgorithms

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

  Abstract: This paper proposes an algorithm for data compression of
  multi-dimensional images, prior to input into neural network feature
  detectors, using Chebyshev approximation. Illustrations are given of the
  fidelity obtainable for geophysical data.
  -----------------------------------------------------------------------
*/

// Disables Eigen's memory alignment which could lead to extra memory padding.
#define EIGEN_DONT_ALIGN
#include <Eigen/Dense>  // Include this header for matrix and vector operations
#include <iostream>
#include <unordered_map>
#include <utility>

namespace ChebyshevAlgorithms {

/**
 * Struct for using and storing data for Chebyshev computations, it's a touch
 * faster to just memcpy a struct unpack than to
 */
struct ChebyshevData {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t N;
    uint32_t Q;
    uint32_t S;
    double_t minVal;
    double_t maxVal;

    void printSelf() {
        std::cout << "ChebyshevData: (x, y, z, N, Q, S, minVal, MaxVal): (" << x
                  << ", " << y << ", " << z << ", " << N << ", " << Q << ", "
                  << S << ", " << minVal << ", " << maxVal << ")" << std::endl;
    }
};

double chebyshevT(int n, double x);

template <typename T>
uint64_t calculateChebyshevBufferSize(uint32_t x, uint32_t y, uint32_t z,
                                      uint32_t N, uint32_t Q, int S) {
    // NOTE: this returns size in BYTES only to be used with unsigned char
    // arrays!
    return (6 * sizeof(int)) + (2 * sizeof(T)) + (N * Q * S * sizeof(T));
}

/**
 * To compress the data,
 * choose N such that N < x,
 * choose Q such that Q < y, and/or
 * choose S such that S < z.
 *
 * All three values can be modified at once. The closer N, Q, and S are to
 * 0, the higher the compression ratio. Through testing we found this
 * chebyshev compression algorithm to be enreliable as it produced single
 * errors greater than 0.001. Best compression values were simply reducing N <
 * x.
 *
 * @param originalMatrix A three dimensional array represented in one dimension
 * through row majoring order
 * @param x The number of elements in the x dimension
 * @param y The number of elements in the y dimension
 * @param z The number of elements in the z dimension
 * @param N The number of the coefficients in the x dimension. Must be less than
 * x.
 * @param Q The number of the coefficients in the y dimension. Must be less than
 * y.
 * @param S The number of the coefficients in the z dimension. Must be less than
 * z.
 *
 * @return A pointer to the byte stream containing the compressed data.
 */
unsigned char* compressMatrix(const double* originalMatrix, const uint32_t x,
                              const uint32_t y, const uint32_t z,
                              const uint32_t N, const uint32_t Q,
                              const uint32_t S, int& bufferSize);

void compressMatrixBuffer(const double* originalMatrix, const uint32_t x,
                          const uint32_t y, const uint32_t z, const uint32_t N,
                          const uint32_t Q, const uint32_t S,
                          unsigned char* outputArray, int& bufferSize);

double* decompressMatrix(const unsigned char* buffer, const int bufferSize);
void decompressMatrixBuffer(const unsigned char* buffer, const int bufferSize,
                            double* outBuff);

class ChebyshevCompression {
   public:
    ChebyshevCompression() {
        // set default to ele6, out3
        set_chebyshev_mat_ele6_out3_dim1();
        set_chebyshev_mat_ele6_out3_dim2();
        set_chebyshev_mat_ele6_out3_dim3();

        recalculate_byte_sizes();
    }

    ChebyshevCompression(const size_t& eleOrder = 6,
                         const size_t& nReduced = 3) {
        if (eleOrder == 6) {
            if (nReduced == 1) {
                set_chebyshev_mat_ele6_out1_dim1();
                set_chebyshev_mat_ele6_out1_dim2();
                set_chebyshev_mat_ele6_out1_dim3();
            } else if (nReduced == 2) {
                set_chebyshev_mat_ele6_out2_dim1();
                set_chebyshev_mat_ele6_out2_dim2();
                set_chebyshev_mat_ele6_out2_dim3();
            } else if (nReduced == 3) {
                set_chebyshev_mat_ele6_out3_dim1();
                set_chebyshev_mat_ele6_out3_dim2();
                set_chebyshev_mat_ele6_out3_dim3();
            } else if (nReduced == 4) {
                set_chebyshev_mat_ele6_out4_dim1();
                set_chebyshev_mat_ele6_out4_dim2();
                set_chebyshev_mat_ele6_out4_dim3();
            }
        }

        recalculate_byte_sizes();
    }

    ~ChebyshevCompression() {
        if (A_cheb_dim1 != nullptr) {
            delete[] A_cheb_dim1;
            A_cheb_dim1 = nullptr;
        }
        if (A_cheb_dim2 != nullptr) {
            delete[] A_cheb_dim2;
            A_cheb_dim2 = nullptr;
        }
        if (A_cheb_dim3 != nullptr) {
            delete[] A_cheb_dim3;
            A_cheb_dim3 = nullptr;
        }
    }

    void set_compression_type(const size_t& eleOrder = 6,
                              const size_t& nReduced = 3);

    void print() {
        std::cout << "ChebyShev Info: mat3d Dims: " << cheb_dim3_decomp << ", "
                  << cheb_dim3_comp << " | mat2d Dims: " << cheb_dim2_decomp
                  << ", " << cheb_dim2_comp
                  << " | mat1d Dims: " << cheb_dim1_decomp << ", "
                  << cheb_dim1_comp
                  << " with compressed byte sizes (1, 2, 3): " << bytes_1d
                  << ", " << bytes_2d << ", " << bytes_3d << std::endl;
    }

    void do_array_norm(double* array, const size_t count, double& minVal,
                       double& maxVal);
    void undo_array_norm(double* array, const size_t count, const double minVal,
                         const double maxVal);

    size_t do_3d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_3d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

    size_t do_2d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_2d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

    size_t do_1d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_1d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

   private:
    double* A_cheb_dim1  = nullptr;
    double* A_cheb_dim2  = nullptr;
    double* A_cheb_dim3  = nullptr;

    // using ints to avoid casting for dgemm_
    int cheb_dim1_decomp = 0;
    int cheb_dim1_comp   = 0;
    int cheb_dim2_decomp = 0;
    int cheb_dim2_comp   = 0;
    int cheb_dim3_decomp = 0;
    int cheb_dim3_comp   = 0;
    int single_dim       = 1;
    double alpha         = 1.0;
    double beta          = 0.0;

    // then know how many total bytes we're going to use
    unsigned int doubles_1d;
    unsigned int doubles_2d;
    unsigned int doubles_3d;
    unsigned int bytes_1d;
    unsigned int bytes_2d;
    unsigned int bytes_3d;

    void recalculate_byte_sizes() {
        doubles_1d = 2 + cheb_dim1_comp;
        doubles_2d = 2 + cheb_dim2_comp;
        doubles_3d = 2 + cheb_dim3_comp;
        bytes_1d   = doubles_1d * sizeof(double);
        bytes_2d   = doubles_2d * sizeof(double);
        bytes_3d   = doubles_3d * sizeof(double);
    }

#include "generated/cheb_transform_ele6.inc.h"
};

// build up an object that we can just use
extern ChebyshevCompression cheby;

}  // namespace ChebyshevAlgorithms

#if 0

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <vector>

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

#endif

#include <zfp.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

namespace ZFPAlgorithms {

/**
 * Struct for using and storing data for Chebyshev computations, it's a touch
 * faster to just memcpy a struct unpack than to
 */
struct ZFPData3d {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    double rate;

    void printSelf() {
        std::cout << "ZFPData3d: (x, y, z, rate): (" << x << ", " << y << ", "
                  << z << ", " << rate << ")" << std::endl;
    }
};

struct ZFPData {
    uint32_t x;
    double rate;

    void printSelf() {
        std::cout << "ZFPData: (x, rate): (" << x << ", " << rate << ")"
                  << std::endl;
    }
};

template <typename T>
int32_t calculateZFPBufferSize3d(T* originalData, int x, int y, int z,
                                 double rate) {
    zfp_field* field;
    zfp_stream* zfp = zfp_stream_open(NULL);

    if constexpr (std::is_same_v<T, double>) {
        field = zfp_field_3d(originalData, zfp_type_double, x, y, z);
        zfp_stream_set_rate(zfp, rate, zfp_type_double, 3, 0);
    } else if constexpr (std::is_same_v<T, float>) {
        field = zfp_field_3d(originalData, zfp_type_float, x, y, z);
        zfp_stream_set_rate(zfp, rate, zfp_type_float, 3, 0);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        field = zfp_field_3d(originalData, zfp_type_int32, x, y, z);
        zfp_stream_set_rate(zfp, rate, zfp_type_int32, 3, 0);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        field = zfp_field_3d(originalData, zfp_type_int64, x, y, z);
        zfp_stream_set_rate(zfp, rate, zfp_type_int64, 3, 0);
    }

    // then calculate the buffer size
    int32_t bufsize = zfp_stream_maximum_size(zfp, field);

    // then add the x, y, and z values along with the double for rate
    bufsize += sizeof(ZFPData3d);

    zfp_stream_close(zfp);
    zfp_field_free(field);

    return bufsize;
}

template <typename T>
int32_t calculateZFPBufferSize(T* originalData, int n, double rate) {
    zfp_field* field;
    zfp_stream* zfp = zfp_stream_open(NULL);

    if constexpr (std::is_same_v<T, double>) {
        field = zfp_field_1d(originalData, zfp_type_double, n);
        zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);
    } else if constexpr (std::is_same_v<T, float>) {
        field = zfp_field_1d(originalData, zfp_type_float, n);
        zfp_stream_set_rate(zfp, rate, zfp_type_float, 1, 0);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        field = zfp_field_1d(originalData, zfp_type_int32, n);
        zfp_stream_set_rate(zfp, rate, zfp_type_int32, 1, 0);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        field = zfp_field_1d(originalData, zfp_type_int64, n);
        zfp_stream_set_rate(zfp, rate, zfp_type_int64, 1, 0);
    }

    int32_t bufsize = zfp_stream_maximum_size(zfp, field);

    // then add the x, y, and z values along with the double for rate
    bufsize += sizeof(ZFPData);

    zfp_stream_close(zfp);
    zfp_field_free(field);

    return bufsize;
}

/**
 * We used ZFP's fixde-rate mode for this compression algoithm in 3x7x7
 * matrices. For high level documentation of the ZFP functios used:
 * https://zfp.readthedocs.io/en/release0.5.4/high-level-api.html#
 * @param rate Valid values are positive doubles.
 *             The rate is defined as maxbits / 4^d, where d=3 for this
 * implementation. Through testing we found values 15.0 - 35.0 to offer the most
 * consistent compression. Values closer to 35 have minimal compression rates
 * while values closer to 0 have much higher compression rates at the cost of
 * precision. Values above 35.0 offer no compression and make the compressed
 * data larger than the original data.
 *
 */
unsigned char* compressMatrix(double* originalData, int x, int y, int z,
                              double rate, int& size);
double* decompressMatrix(unsigned char* buffer, int bufferSize);

void compressMatrixBuffer(const double* originalMatrix, const uint32_t x,
                          const uint32_t y, const uint32_t z, double rate,
                          unsigned char* outBuffer, int& size,
                          zfp_field* field = nullptr,
                          zfp_stream* zfp  = nullptr);

void decompressMatrixBuffer(unsigned char* buffer, int bufferSize,
                            double* outBuff);

/**
 * We used ZFP's fixde-rate mode for this compression algoithm in 3x7x7
 * matrices. For high level documentation of the ZFP functios used:
 * https://zfp.readthedocs.io/en/release0.5.4/high-level-api.html#
 * @param rate Valid values are positive doubles.
 *             The rate is defined as maxbits / 4^d, where d=1 for this
 * algorithm. Therefore, the rate should be bounded by 4/4 -> 64/4 or 1 -> 16
 *
 */
unsigned char* compressMatrix1D(double* originalData, int n, double rate,
                                int& size);
double* decompressMatrix1D(unsigned char* buffer, int bufferSize);

// pre-allocated memory version
void decompressMatrix1D(unsigned char* buffer, int bufferSize, double* outBuff);

unsigned char* compressMatrix1D_fixedPrecision(double* originalData, int n,
                                               double precision, int& size);
// pre-allocated memory version, fixed precision
void decompressMatrix1D_fixedPrecision(unsigned char* buffer, int bufferSize,
                                       double* outBuff);

class ZFPCompression {
   public:
    ZFPCompression(const size_t& eleOrder = 6, const double rate = 5.0)
        : eleOrder(eleOrder), rate(rate) {
        zfp_num_per_dim = eleOrder - 1;

        // TODO: calculate if the rate is too large

        // streams, by default are in set rate mode, this is good for knowing
        // our size
        zfp3d           = zfp_stream_open(NULL);
        zfp_stream_set_rate(zfp3d, rate, zfp_type_double, 3, 0);
        field_3d = zfp_field_3d(NULL, zfp_type_double, zfp_num_per_dim,
                                zfp_num_per_dim, zfp_num_per_dim);

        zfp2d    = zfp_stream_open(NULL);
        zfp_stream_set_rate(zfp2d, rate, zfp_type_double, 2, 0);
        field_2d = zfp_field_2d(NULL, zfp_type_double, zfp_num_per_dim,
                                zfp_num_per_dim);

        zfp1d    = zfp_stream_open(NULL);
        zfp_stream_set_rate(zfp1d, rate, zfp_type_double, 1, 0);
        field_1d = zfp_field_1d(NULL, zfp_type_double, zfp_num_per_dim);

        // TEMP: accuracy mode
        set_accuracy_mode(eleOrder, 1e-6);
    }

    ~ZFPCompression() { close_and_free_all(); }

    void close_and_free_all() {
        close_all_streams();
        free_all_fields();
    }

    void close_all_streams() {
        if (zfp3d != nullptr) zfp_stream_close(zfp3d);

        if (zfp2d != nullptr) zfp_stream_close(zfp2d);

        if (zfp1d != nullptr) zfp_stream_close(zfp1d);

        zfp3d = nullptr;
        zfp2d = nullptr;
        zfp1d = nullptr;
    }

    void free_all_fields() {
        if (field_3d != nullptr) zfp_field_free(field_3d);

        if (field_2d != nullptr) zfp_field_free(field_2d);

        if (field_1d != nullptr) zfp_field_free(field_1d);

        field_3d = nullptr;
        field_2d = nullptr;
        field_1d = nullptr;
    }

    void set_accuracy_mode(const size_t& eleOrder = 6,
                           const double tolerance = 1e-6) {
        // re-establish all fields in accuracy-only mode
        zfp_stream_set_accuracy(zfp3d, tolerance);
        assert(zfp_stream_compression_mode(zfp3d) == zfp_mode_fixed_accuracy);

        zfp_stream_set_accuracy(zfp2d, tolerance);
        assert(zfp_stream_compression_mode(zfp2d) == zfp_mode_fixed_accuracy);

        zfp_stream_set_accuracy(zfp1d, tolerance);
        assert(zfp_stream_compression_mode(zfp1d) == zfp_mode_fixed_accuracy);
    }

    size_t do_3d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_3d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

    size_t do_2d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_2d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

    size_t do_1d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_1d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

   private:
    zfp_stream* zfp3d      = nullptr;
    zfp_stream* zfp2d      = nullptr;
    zfp_stream* zfp1d      = nullptr;
    int zfp_dim1_decomp    = 0;
    int zfp_dim2_decomp    = 0;
    int zfp_dim3_decomp    = 0;
    size_t eleOrder        = 0;
    double rate            = 20.0;
    size_t zfp_num_per_dim = 0;
    zfp_field* field_3d    = nullptr;
    zfp_field* field_2d    = nullptr;
    zfp_field* field_1d    = nullptr;
};

// A ZFPCompression object to use "globally"
extern ZFPCompression zfpblockwise;

}  // namespace ZFPAlgorithms

#include <blosc.h>

#include <iostream>

/**
 * Namespace BLOSCCompression
 *
 * Provides functionalities for compressing and decompressing data using the
Blosc library.
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
    int compressionLevel = 9; // Choose the compression level (1-9, where 9 is
highest compression)

    int bytestreamSize;
    blosc_init();
    unsigned char* bytestream = BLOSCCompression::compressData("zstd"
,compressionLevel, n, originalMatrix, bytestreamSize); blosc_destroy();

    delete[] originalMatrix;

    // ...
    // Pass bytestream and bytestreamSize to a different machine
    // ...


    blosc_init();
    double* decompressedData = BLOSCCompression::decompressData(bytestream,
bytestreamSize); blosc_destroy();

    delete[] decompressedData;
    return 0;
}
 */
namespace BLOSCAlgorithms {

/**
 * Compresses a block of data using the Blosc library.
 * This function compresses a given array of doubles using the specified Blosc
 * compressor and compression level.
 *
 * Before calling this function, ensure blosc_init() has been called to
 * initialize the Blosc library. After using the compressed data,
 * blosc_destroy() should be called for proper cleanup.
 *
 * @param blosc_compressor The compression algorithm to be used. Must be one of
 * "blosclz", "lz4", "lz4hc", "zlib", or "zstd".
 * @param clevel Choose the compression level (1-9, where 9 is highest
 * compression)
 * @param n The number of elements in the original data array.
 * @param originalData Pointer to the original array of doubles to be
 * compressed.
 * @param byteStreamSize Reference to an integer where the size of the resulting
 * bytestream will be stored.
 * @return Pointer to the compressed data bytestream.
 * @throw std::runtime_error if compression fails.
 */
unsigned char* compressData(const char* blosc_compressor, int clevel, int n,
                            double* originalData, int& byteStreamSize);

/**
 * Decompresses a bytestream using the Blosc library.
 * It decompresses a bytestream that was created by the compressData function.
 *
 * Before calling this function, ensure blosc_init() has been called to
 * initialize the Blosc library. After using the compressed data,
 * blosc_destroy() should be called for proper cleanup.
 *
 * @param byteStream Pointer to the bytestream containing the compressed data.
 * @param byteStreamSize The size of the bytestream.
 * @return Pointer to the decompressed data array.
 * @throw std::runtime_error if decompression fails or if input is invalid.
 */
double* decompressData(unsigned char* byteStream, int byteStreamSize);

/**
 * Decompresses a bytestream using the Blosc library.
 * It decompresses a bytestream that was created by the compressData function.
 *
 * Before calling this function, ensure blosc_init() has been called to
 * initialize the Blosc library. After using the compressed data,
 * blosc_destroy() should be called for proper cleanup.
 *
 * @param byteStream Pointer to the bytestream containing the compressed data.
 * @param byteStreamSize The size of the bytestream.
 * @param outBuff The output buffer.
 * @throw std::runtime_error if decompression fails or if input is invalid.
 */
void decompressData(unsigned char* byteStream, int byteStreamSize,
                    double* outBuff);

class BloscCompression {
   public:
    BloscCompression(const size_t& eleOrder             = 6,
                     const std::string& bloscCompressor = "lz4",
                     const int& clevel = 4, const int& doShuffle = 1)
        : eleOrder(eleOrder),
          bloscCompressor(bloscCompressor),
          clevel(clevel),
          doShuffle(doShuffle) {
        // TODO: init shouldn't be called here, it should be in some other
        // initialization
        blosc_init();
        blosc_set_compressor(bloscCompressor.c_str());

        size_t points_1d        = eleOrder - 1;

        // calculate the number of bytes based on the element order
        blosc_original_bytes_1d = sizeof(double) * points_1d;
        blosc_original_bytes_2d = points_1d * blosc_original_bytes_1d;
        blosc_original_bytes_3d = points_1d * blosc_original_bytes_2d;

        // then with the overhead for the maximum possible amount it could take.
        // This guarantees success, but will basically never take this much.
        blosc_original_bytes_overhead_1d =
            blosc_original_bytes_1d + BLOSC_MAX_OVERHEAD;
        blosc_original_bytes_overhead_2d =
            blosc_original_bytes_2d + BLOSC_MAX_OVERHEAD;
        blosc_original_bytes_overhead_3d =
            blosc_original_bytes_3d + BLOSC_MAX_OVERHEAD;
    }

    ~BloscCompression() {
        // TODO: destroy shouldn't be called here, it should be in some other
        // destruction
        blosc_destroy();
    }

    size_t do_3d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_3d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

    size_t do_2d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_2d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

    size_t do_1d_compression(double* originalMatrix,
                             unsigned char* outputArray);
    size_t do_1d_decompression(unsigned char* compressedBuffer,
                               double* outputArray);

   private:
    size_t eleOrder;
    // blosc settings
    std::string bloscCompressor;
    int clevel;
    int doShuffle;

    // tracking original sizes
    size_t blosc_original_bytes_3d;
    size_t blosc_original_bytes_2d;
    size_t blosc_original_bytes_1d;

    size_t blosc_original_bytes_overhead_3d;
    size_t blosc_original_bytes_overhead_2d;
    size_t blosc_original_bytes_overhead_1d;

    // overhead bytes
};

extern BloscCompression bloscblockwise;

}  // namespace BLOSCAlgorithms

namespace dendro_compress {

enum CompressionType { NONE = 0, ZFP, CHEBYSHEV, BLOSC };

// then the global option
extern CompressionType COMPRESSION_OPTION;

std::size_t blockwise_compression(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<unsigned char>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);

std::size_t blockwise_decompression(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<unsigned char>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);
}  // namespace dendro_compress
