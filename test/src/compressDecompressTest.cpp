#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "compression.h"
#include "compression/compression_base.hpp"
#include "compression/compression_factory.hpp"
#include "profiler.h"

#define IDX(i, j, k) ((k) * y * z + (j) * y + (i))  // 3D to 1D indexing

#define UNIFORM_RAND_0_TO_X(X) ((double)rand() / (double)RAND_MAX * X)

namespace testcomp {

#if 1
std::string compressor_path_onnx_3d =
    "../testmodels/ENCODER_twodim_equal_in_out_3d.onnx";
std::string decompressor_path_onnx_3d =
    "../testmodels/DECODER_twodim_equal_in_out_3d.onnx";
std::string compressor_path_onnx_2d =
    "../testmodels/ENCODER_twodim_equal_in_out_2d.onnx";
std::string decompressor_path_onnx_2d =
    "../testmodels/DECODER_twodim_equal_in_out_2d.onnx";
std::string compressor_path_onnx_1d =
    "../testmodels/ENCODER_twodim_equal_in_out_1d.onnx";
std::string decompressor_path_onnx_1d =
    "../testmodels/DECODER_twodim_equal_in_out_1d.onnx";
std::string compressor_path_onnx_0d =
    "../testmodels/ENCODER_twodim_equal_in_out_0d.onnx";
std::string decompressor_path_onnx_0d =
    "../testmodels/DECODER_twodim_equal_in_out_0d.onnx";

#else
std::string compressor_path_onnx_3d =
    "../testmodels/ENCODER_singledim_equal_in_out_3d.onnx";
std::string decompressor_path_onnx_3d =
    "../testmodels/DECODER_singledim_equal_in_out_3d.onnx";
std::string compressor_path_onnx_2d =
    "../testmodels/ENCODER_singledim_equal_in_out_2d.onnx";
std::string decompressor_path_onnx_2d =
    "../testmodels/DECODER_singledim_equal_in_out_2d.onnx";
std::string compressor_path_onnx_1d =
    "../testmodels/ENCODER_singledim_equal_in_out_1d.onnx";
std::string decompressor_path_onnx_1d =
    "../testmodels/DECODER_singledim_equal_in_out_1d.onnx";
std::string compressor_path_onnx_0d =
    "../testmodels/ENCODER_singledim_equal_in_out_0d.onnx";
std::string decompressor_path_onnx_0d =
    "../testmodels/DECODER_singledim_equal_in_out_0d.onnx";
#endif

std::string compressor_path_pt_3d =
    "../testmodels/ENCODER_twodim_equal_in_out_3d.pt";
std::string decompressor_path_pt_3d =
    "../testmodels/DECODER_twodim_equal_in_out_3d.pt";
std::string compressor_path_pt_2d =
    "../testmodels/ENCODER_twodim_equal_in_out_2d.pt";
std::string decompressor_path_pt_2d =
    "../testmodels/DECODER_twodim_equal_in_out_2d.pt";
std::string compressor_path_pt_1d =
    "../testmodels/ENCODER_twodim_equal_in_out_1d.pt";
std::string decompressor_path_pt_1d =
    "../testmodels/DECODER_twodim_equal_in_out_1d.pt";
std::string compressor_path_pt_0d =
    "../testmodels/ENCODER_twodim_equal_in_out_0d.pt";
std::string decompressor_path_pt_0d =
    "../testmodels/DECODER_twodim_equal_in_out_0d.pt";

profiler_t profiler_compress_3d;
profiler_t profiler_decompress_3d;
profiler_t profiler_compress_2d;
profiler_t profiler_decompress_2d;
profiler_t profiler_compress_1d;
profiler_t profiler_decompress_1d;

}  // namespace testcomp

template <typename T>
void fillMatrixWave(T* vec, int x, int y, int z, double frequency,
                    double amplitude, double phase, double fx, double fy,
                    double fz, double dx, double dy, double dz) {
    for (int k = 0; k < z; k++) {
        double zz = k * dz;
        for (int j = 0; j < y; j++) {
            double yy = j * dy;
            for (int i = 0; i < x; i++) {
                double xx = i * dx;
                // Calculate the value of the wave at the point (i, j, k). The
                // value is determined by a sine function, which is influenced
                // by the frequency, amplitude, and phase input parameters, as
                // well as the propagation factors fx, fy, and fz for each
                // dimension. The result is a wave that can propagate
                // differently along the x, y, and z dimensions.
                double waveValue =
                    amplitude *
                    sin(frequency * (fx * xx + fy * yy + fz * zz) + phase);

                // Store the wave value in the 1D array. The 3D index (i, j, k)
                // is converted to a 1D index using the IDX macro.
                vec[IDX(i, j, k)] = waveValue;
            }
        }
    }
}

template <typename T>
T* createMatrixWave(int x, int y, int z, double frequency, double amplitude,
                    double phase, double fx, double fy, double fz, double dx,
                    double dy, double dz) {
    // Allocate a 1D array to hold the wave data. The size of the array is the
    // product of the dimensions x, y, and z.
    T* matrix = new T[x * y * z];

    fillMatrixWave(matrix, x, y, z, frequency, amplitude, phase, fx, fy, fz, dx,
                   dy, dz);

    // Return the 1D array containing the wave data.
    return matrix;
}

template <typename T>
void fillMatrixRandom(int x, int y, int z, double amplitude, T* vec) {
    for (int k = 0; k < z; k++) {
        for (int j = 0; j < y; j++) {
            for (int i = 0; i < x; i++) {
                // Calculate the value of the wave at the point (i, j, k). The
                // value is determined by a sine function, which is influenced
                // by the frequency, amplitude, and phase input parameters, as
                // well as the propagation factors fx, fy, and fz for each
                // dimension. The result is a wave that can propagate
                // differently along the x, y, and z dimensions.
                double waveValue  = amplitude * (UNIFORM_RAND_0_TO_X(2) - 1);

                // Store the wave value in the 1D array. The 3D index (i, j, k)
                // is converted to a 1D index using the IDX macro.
                vec[IDX(i, j, k)] = waveValue;
            }
        }
    }
}

template <typename T>
T* createMatrixRandom(int x, int y, int z, double amplitude) {
    // Allocate a 1D array to hold the wave data. The size of the array is the
    // product of the dimensions x, y, and z.
    T* matrix = new T[x * y * z];

    fillMatrixRandom(x, y, z, amplitude, matrix);

    // Return the 1D array containing the wave data.
    return matrix;
}

template <typename T>
void printError(std::vector<T> originalMatrix,
                std::vector<T> decompressedMatrix, std::string prefix = "") {
    // Computing and printing Mean Squared Error, Mean Absolute Error,
    // Maximum Absolute Error, Root Mean Squared Error and Peak Signal to Noise
    // Ratio.
    double mse            = 0;

    std::size_t total_pts = originalMatrix.size();

    assert(total_pts == decompressedMatrix.size());

    for (size_t i = 0; i < total_pts; ++i) {
        double error = originalMatrix[i] - decompressedMatrix[i];
        mse += error * error;
    }
    mse /= (total_pts);

    std::cout << prefix << "MSE,MAE,MaxAE,RMSE,PSNR: ";

    std::cout << mse << ",";
    // std::cout << prefix << "Mean Squared Error: " << mse << "\n";
    // Compute and print Mean Absolute Error
    double mae = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        double error = std::abs(originalMatrix[i] - decompressedMatrix[i]);
        mae += error;
    }
    mae /= (total_pts);
    std::cout << mae << ",";
    // std::cout << prefix << "Mean Absolute Error: " << mae << "\n";

    // Compute and print Maximum Absolute Error
    double maxError = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        double error = std::abs(originalMatrix[i] - decompressedMatrix[i]);
        maxError     = std::max(maxError, error);
    }
    std::cout << maxError << ",";
    // std::cout << prefix << "Max Absolute Error: " << maxError << "\n";

    // Compute and print Root Mean Squared Error
    double rmse = sqrt(mse);
    std::cout << rmse << ",";
    // std::cout << prefix << "Root Mean Squared Error: " << rmse << "\n";

    // Compute and print Peak Signal to Noise Ratio (PSNR)
    double maxOriginalValue = *std::max_element(
        originalMatrix.data(), originalMatrix.data() + (total_pts));
    double psnr = 20 * log10(maxOriginalValue / rmse);
    std::cout << psnr << std::endl;
    // std::cout << prefix << "Peak Signal to Noise Ratio (in dB): " << psnr
    //           << "\n";
}

template <typename T>
void printComparison(const T* originalMatrix, const T* decompressedMatrix,
                     int x, int y, int z) {
    double epsilon = 1e-9;
    std::cout << std::fixed
              << std::setprecision(
                     15);  // for consistent number of decimal places
    int width =
        20;  // 6 decimal digits, 1 dot, 1 sign and up to 4 whole part digits

    for (int k = 0; k < z; k++) {
        std::cout << "z = " << k << ":\n";
        for (int j = 0; j < y; j++) {
            for (int i = 0; i < x; i++) {
                T original     = originalMatrix[IDX(i, j, k)];
                T decompressed = decompressedMatrix[IDX(i, j, k)];

                std::cout << "[";
                std::cout << std::setw(width) << original << ",\n ";
                std::cout << std::setw(width) << decompressed;

                // Check if the numbers differ significantly
                if (std::abs(original - decompressed) > epsilon) {
                    std::cout << "*]\n";
                } else {
                    std::cout << "]\n";
                }
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

template <typename T>
void profileCompressor(dendrocompression::Compression<T>* compressor,
                       T* original_data, T* uncompressed_data,
                       unsigned char* compressed_buffer, unsigned int nbatches,
                       unsigned int z, unsigned int y,
                       unsigned int num_runs = 1000) {
    testcomp::profiler_compress_1d.clear();
    testcomp::profiler_compress_2d.clear();
    testcomp::profiler_compress_3d.clear();
    testcomp::profiler_decompress_1d.clear();
    testcomp::profiler_decompress_2d.clear();
    testcomp::profiler_decompress_3d.clear();
    std::cout << "\tNow profiling " << num_runs << " times..." << std::endl;
    std::size_t compressed_bytes, compressed_bytes_bak;
    for (unsigned int dim : {3, 2, 1}) {
        if (dim == 3) {
            for (unsigned int i = 0; i < num_runs; ++i) {
                testcomp::profiler_compress_3d.start();
                // run 3d compression/decompression
                compressed_bytes = compressor->do_compress_3d(
                    original_data, compressed_buffer, nbatches);
                testcomp::profiler_compress_3d.stop();

                // do decompression
                testcomp::profiler_decompress_3d.start();
                compressed_bytes_bak = compressor->do_decompress_3d(
                    compressed_buffer, uncompressed_data, nbatches);
                testcomp::profiler_decompress_3d.stop();
            }
        } else if (dim == 2) {
            for (unsigned int i = 0; i < num_runs; ++i) {
                testcomp::profiler_compress_2d.start();
                // run 3d compression/decompression
                compressed_bytes = compressor->do_compress_2d(
                    original_data, compressed_buffer, nbatches * z);
                testcomp::profiler_compress_2d.stop();

                // do decompression
                testcomp::profiler_decompress_2d.start();
                compressed_bytes_bak = compressor->do_decompress_2d(
                    compressed_buffer, uncompressed_data, nbatches * z);
                testcomp::profiler_decompress_2d.stop();
            }
        } else if (dim == 1) {
            for (unsigned int i = 0; i < num_runs; ++i) {
                testcomp::profiler_compress_1d.start();
                // run 3d compression/decompression
                compressed_bytes = compressor->do_compress_1d(
                    original_data, compressed_buffer, nbatches * z * y);
                testcomp::profiler_compress_1d.stop();

                // do decompression
                testcomp::profiler_decompress_1d.start();
                compressed_bytes_bak = compressor->do_decompress_1d(
                    compressed_buffer, uncompressed_data, nbatches * z * y);
                testcomp::profiler_decompress_1d.stop();
            }
        }
    }

    std::cout << "\tTime 3d, 2d, 1d (comp/decomp): "
              << testcomp::profiler_compress_3d.seconds / num_runs << "/"
              << testcomp::profiler_decompress_3d.seconds / num_runs << "   "
              << testcomp::profiler_compress_2d.seconds / num_runs << "/"
              << testcomp::profiler_decompress_2d.seconds / num_runs << "   "
              << testcomp::profiler_compress_1d.seconds / num_runs << "/"
              << testcomp::profiler_decompress_1d.seconds / num_runs
              << std::endl;
}

int main() {
    srand(time(0));

    typedef float COMPRESSOR_TYPE;

    // ensure the compressors are registered!
    dendrocompression::register_compressors();

    unsigned int eleorder      = 6;
    unsigned int n             = eleorder - 1;
    unsigned int nvar          = 2;

    unsigned int nbatches      = 10;

    unsigned int interp_stride = 2;

    std::cout << "N: " << n << " nvar: " << nvar << std::endl;

    int x = n, y = n, z = n;  // Dimensions of matrix. Modify as needed.
    int npts       = x * y * z;
    int total_npts = x * y * z * nvar * nbatches;

    std::cout << "x, y, z: " << x << ", " << y << ", " << z
              << " -> npts: " << npts << " total_npts: " << total_npts
              << std::endl;

    double compressed_buffer_padding = 1.1;
    std::vector<double> fullmatrix(total_npts);
    std::vector<COMPRESSOR_TYPE> fullmatrix_send_type(total_npts);
    std::vector<double> decompressed_full(total_npts, 0.0);
    std::vector<COMPRESSOR_TYPE> decompressed_send_type(total_npts, 0.0);
    unsigned char* compressed_buffer = static_cast<unsigned char*>(
        malloc(std::size_t(compressed_buffer_padding * (double)total_npts *
                           (double)sizeof(COMPRESSOR_TYPE))));

    uint32_t offset = 0;
    for (int jj = 0; jj < nbatches; jj++) {
        for (int ii = 0; ii < nvar; ii++) {
            fillMatrixWave(&fullmatrix[offset], x, y, z, 1.0,
                           (jj + 1) * (ii + 1) * 1.0, 0.0, 1.0, 1.0, 1.0, 0.01,
                           0.01, 0.01);

            fillMatrixRandom(x, y, z, 1000000000000,
                             &decompressed_full[offset]);
            fillMatrixRandom(x, y, z, 1000000000000,
                             &decompressed_send_type[offset]);

            offset += npts;
        }
    }

    // copy the data to fullmatrix_send type
    for (unsigned int i = 0; i < total_npts; ++i) {
        fullmatrix_send_type[i] = static_cast<COMPRESSOR_TYPE>(fullmatrix[i]);
    }

    std::cout << "FINISHED INITALIZING DATA!" << std::endl;

    std::size_t originalMatrixBytes = total_npts * sizeof(double);
    std::size_t originalSENDBytes   = total_npts * sizeof(COMPRESSOR_TYPE);

    double zfp_param                = 10.;
    std::string zfp_mode            = "precision";

    std::vector<std::vector<std::any>> params{
        // dummy params
        {eleorder, nvar},
        // onnx model params
        {eleorder, nvar, testcomp::compressor_path_onnx_3d,
         testcomp::decompressor_path_onnx_3d, testcomp::compressor_path_onnx_2d,
         testcomp::decompressor_path_onnx_2d, testcomp::compressor_path_onnx_1d,
         testcomp::decompressor_path_onnx_1d, testcomp::compressor_path_onnx_0d,
         testcomp::decompressor_path_onnx_0d},
        // torchscript model params
        {eleorder, nvar, testcomp::compressor_path_pt_3d,
         testcomp::decompressor_path_pt_3d, testcomp::compressor_path_pt_2d,
         testcomp::decompressor_path_pt_2d, testcomp::compressor_path_pt_1d,
         testcomp::decompressor_path_pt_1d, testcomp::compressor_path_pt_0d,
         testcomp::decompressor_path_pt_0d},
        // zfp params
        {eleorder, nvar, zfp_mode, zfp_param},
        // interp params
        {eleorder, nvar, interp_stride}};

    std::vector<dendrocompression::CompressionType> comp_types = {
        dendrocompression::CompressionType::COMP_DUMMY,
        dendrocompression::CompressionType::COMP_ONNX_MODEL,
        dendrocompression::CompressionType::COMP_TORCH_SCRIPT,
        dendrocompression::CompressionType::COMP_ZFP,
        dendrocompression::CompressionType::COMP_INTERP};

    unsigned int test_idx = 0;
    for (auto& comp_type : comp_types) {
        std::cout << "BUILDING THE COMPRESSOR" << std::endl;
        std::unique_ptr<dendrocompression::Compression<COMPRESSOR_TYPE>>
            compressor = [&]() {
                if constexpr (std::is_same_v<COMPRESSOR_TYPE, double>) {
                    return dendrocompression::doubleCompressor.create(
                        comp_type, params[test_idx++]);
                } else if constexpr (std::is_same_v<COMPRESSOR_TYPE, float>) {
                    return dendrocompression::floatCompressor.create(
                        comp_type, params[test_idx++]);
                } else {
                    // does nothing!
                }
            }();

        std::cout << " COMPRESSOR type:" << compressor->to_string()
                  << std::endl;

        for (unsigned int dim : {3, 2, 1}) {
            // do full compression/decompression based on each one
            std::size_t compressed_bytes, compressed_bytes_bak;

            std::cout << "\tNUMDIM: " << dim << std::endl;
            if (dim == 3) {
                // run 3d compression/decompression
                compressed_bytes = compressor->do_compress_3d(
                    fullmatrix_send_type.data(), compressed_buffer, nbatches);

                // do decompression
                compressed_bytes_bak = compressor->do_decompress_3d(
                    compressed_buffer, decompressed_send_type.data(), nbatches);
            } else if (dim == 2) {
                // run 3d compression/decompression
                compressed_bytes =
                    compressor->do_compress_2d(fullmatrix_send_type.data(),
                                               compressed_buffer, nbatches * z);

                // do decompression
                compressed_bytes_bak = compressor->do_decompress_2d(
                    compressed_buffer, decompressed_send_type.data(),
                    nbatches * z);
            } else if (dim == 1) {
                // run 3d compression/decompression
                compressed_bytes = compressor->do_compress_1d(
                    fullmatrix_send_type.data(), compressed_buffer,
                    nbatches * z * y);

                // do decompression
                compressed_bytes_bak = compressor->do_decompress_1d(
                    compressed_buffer, decompressed_send_type.data(),
                    nbatches * z * y);
            }
            std::cout << "\t\tOriginal,TrueSend,Compressed bytes: "
                      << originalMatrixBytes << "," << originalSENDBytes << ","
                      << compressed_bytes << " ("
                      << (double)originalMatrixBytes / (double)compressed_bytes
                      << " true ratio, "
                      << (double)originalSENDBytes / (double)compressed_bytes
                      << " updated ratio)" << std::endl;
            // Printing various types of error between original and decompressed
            // data
            // copy over the decompressed full
            // copy the data to fullmatrix_send type
            for (unsigned int i = 0; i < total_npts; ++i) {
                decompressed_full[i] =
                    static_cast<double>(decompressed_send_type[i]);
            }
            printError(fullmatrix, decompressed_full, "\t\t");
        }

        // then profile!
        profileCompressor<COMPRESSOR_TYPE>(
            compressor.get(), fullmatrix_send_type.data(),
            decompressed_send_type.data(), compressed_buffer, nbatches, z, y,
            1000);

        std::cout << std::endl << std::endl;
    }

    // Freeing the memory
    free(compressed_buffer);

    return 0;
}
