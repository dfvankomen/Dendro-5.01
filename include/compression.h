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
// #include <Eigen/Dense>
// #include <Eigen/SVD>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

namespace ChebyshevAlgorithms {

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
        } else if (eleOrder == 8) {
            if (nReduced == 1) {
                set_chebyshev_mat_ele8_out1_dim1();
                set_chebyshev_mat_ele8_out1_dim2();
                set_chebyshev_mat_ele8_out1_dim3();
            } else if (nReduced == 2) {
                set_chebyshev_mat_ele8_out2_dim1();
                set_chebyshev_mat_ele8_out2_dim2();
                set_chebyshev_mat_ele8_out2_dim3();
            } else if (nReduced == 3) {
                set_chebyshev_mat_ele8_out3_dim1();
                set_chebyshev_mat_ele8_out3_dim2();
                set_chebyshev_mat_ele8_out3_dim3();
            } else if (nReduced == 4) {
                set_chebyshev_mat_ele8_out4_dim1();
                set_chebyshev_mat_ele8_out4_dim2();
                set_chebyshev_mat_ele8_out4_dim3();
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
#include "generated/cheb_transform_ele8.inc.h"
};

// build up an object that we can just use
extern ChebyshevCompression cheby;

}  // namespace ChebyshevAlgorithms

#include <zfp.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

namespace ZFPAlgorithms {

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

        mode_set = "rate";
    }

    ~ZFPCompression() { close_and_free_all(); }

    void setEleOrder(const size_t& eleOrder_in) {
        close_and_free_all();

        // then set the new values
        eleOrder        = eleOrder_in;
        zfp_num_per_dim = eleOrder - 1;

        // finally open the new streams
        zfp3d           = zfp_stream_open(NULL);
        zfp2d           = zfp_stream_open(NULL);
        zfp1d           = zfp_stream_open(NULL);

        // then the fields
        field_3d        = zfp_field_3d(NULL, zfp_type_double, zfp_num_per_dim,
                                       zfp_num_per_dim, zfp_num_per_dim);
        field_2d        = zfp_field_2d(NULL, zfp_type_double, zfp_num_per_dim,
                                       zfp_num_per_dim);
        field_1d        = zfp_field_1d(NULL, zfp_type_double, zfp_num_per_dim);

        // std::cout << "ZFP Element Order set to: " << eleOrder << std::endl;

        // NOTE: setting rate and accuracy should always be called after this
        // function
    }

    void setRate(const double rate_in) {
        rate = rate_in;

        if (zfp3d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp2d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp1d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");

        zfp_stream_set_rate(zfp3d, rate, zfp_type_double, 3, 0);
        zfp_stream_set_rate(zfp2d, rate, zfp_type_double, 2, 0);
        zfp_stream_set_rate(zfp1d, rate, zfp_type_double, 1, 0);

        assert(zfp_stream_compression_mode(zfp3d) == zfp_mode_fixed_rate);
        assert(zfp_stream_compression_mode(zfp2d) == zfp_mode_fixed_rate);
        assert(zfp_stream_compression_mode(zfp1d) == zfp_mode_fixed_rate);

        mode_set = "rate";
        // std::cout << "ZFP Rate set to: " << rate << std::endl;
    }

    void setAccuracy(const double tolerance_in) {
        tolerance = tolerance_in;

        if (zfp3d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp2d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp1d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");

        zfp_stream_set_accuracy(zfp3d, tolerance);
        zfp_stream_set_accuracy(zfp2d, tolerance);
        zfp_stream_set_accuracy(zfp1d, tolerance);

        assert(zfp_stream_compression_mode(zfp3d) == zfp_mode_fixed_accuracy);
        assert(zfp_stream_compression_mode(zfp2d) == zfp_mode_fixed_accuracy);
        assert(zfp_stream_compression_mode(zfp1d) == zfp_mode_fixed_accuracy);

        mode_set = "accuracy";
        // std::cout << "ZFP Tolerance set to: " << rate << std::endl;
    }

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
    std::string mode_set   = "none";
    int zfp_dim1_decomp    = 0;
    int zfp_dim2_decomp    = 0;
    int zfp_dim3_decomp    = 0;
    size_t eleOrder        = 0;
    double rate            = 20.0;
    double tolerance       = 20.0;
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

        calculateSizes();
    }

    ~BloscCompression() {
        // TODO: destroy shouldn't be called here, it should be in some other
        // destruction
        blosc_destroy();
    }

    void setEleOrder(size_t eleOrder_in) {
        eleOrder = eleOrder_in;

        calculateSizes();
    }

    void setCompressor(const std::string& bloscCompressor_in) {
        bloscCompressor = bloscCompressor_in;

        blosc_set_compressor(bloscCompressor.c_str());
    }

    void calculateSizes() {
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
static const char* COMPRESSION_TYPE_NAMES[] = {"NONE", "ZFP", "CHEBYSHEV",
                                               "BLOSC"};

// then the global option
extern CompressionType COMPRESSION_OPTION;

struct CompressionOptions {
    size_t eleOrder             = 6;
    // options just for blosc
    std::string bloscCompressor = "lz4";
    int bloscClevel             = 5;
    int bloscDoShuffle          = 1;

    // options just for ZFP

    // Options: accuracy, rate
    std::string zfpMode         = "accuracy";
    double zfpRate              = 5.0;
    double zfpAccuracyTolerance = 1e-6;

    // options for chebyshev
    size_t chebyNReduced        = 3;
};

std::ostream& operator<<(std::ostream& out, const CompressionOptions opts);

std::ostream& operator<<(std::ostream& out, const CompressionType t);

void set_compression_options(CompressionType compT,
                             const CompressionOptions& compOpt);

std::size_t blockwise_compression(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<unsigned char>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);

std::size_t blockwise_decompression(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<unsigned char>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);
}  // namespace dendro_compress
