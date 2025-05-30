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

#define IDX(i, j, k) ((k) * y * z + (j) * y + (i))  // 3D to 1D indexing

#define UNIFORM_RAND_0_TO_X(X) ((double)rand() / (double)RAND_MAX * X)

void fillMatrixWave(double* vec, int x, int y, int z, double frequency,
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

double* createMatrixWave(int x, int y, int z, double frequency,
                         double amplitude, double phase, double fx, double fy,
                         double fz, double dx, double dy, double dz) {
    // Allocate a 1D array to hold the wave data. The size of the array is the
    // product of the dimensions x, y, and z.
    double* matrix = new double[x * y * z];

    fillMatrixWave(matrix, x, y, z, frequency, amplitude, phase, fx, fy, fz, dx,
                   dy, dz);

    // Return the 1D array containing the wave data.
    return matrix;
}

void fillMatrixRandom(int x, int y, int z, double amplitude, double* vec) {
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

double* createMatrixRandom(int x, int y, int z, double amplitude) {
    // Allocate a 1D array to hold the wave data. The size of the array is the
    // product of the dimensions x, y, and z.
    double* matrix = new double[x * y * z];

    fillMatrixRandom(x, y, z, amplitude, matrix);

    // Return the 1D array containing the wave data.
    return matrix;
}

void printError(std::vector<double> originalMatrix,
                std::vector<double> decompressedMatrix) {
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

    std::cout << "Mean Squared Error: " << mse << "\n";
    // Compute and print Mean Absolute Error
    double mae = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        double error = std::abs(originalMatrix[i] - decompressedMatrix[i]);
        mae += error;
    }
    mae /= (total_pts);
    std::cout << "Mean Absolute Error: " << mae << "\n";

    // Compute and print Maximum Absolute Error
    double maxError = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        double error = std::abs(originalMatrix[i] - decompressedMatrix[i]);
        maxError     = std::max(maxError, error);
    }
    std::cout << "Max Absolute Error: " << maxError << "\n";

    // Compute and print Root Mean Squared Error
    double rmse = sqrt(mse);
    std::cout << "Root Mean Squared Error: " << rmse << "\n";

    // Compute and print Peak Signal to Noise Ratio (PSNR)
    double maxOriginalValue = *std::max_element(
        originalMatrix.data(), originalMatrix.data() + (total_pts));
    double psnr = 20 * log10(maxOriginalValue / rmse);
    std::cout << "Peak Signal to Noise Ratio (in dB): " << psnr << "\n";
}

void printComparison(const double* originalMatrix,
                     const double* decompressedMatrix, int x, int y, int z) {
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
                double original     = originalMatrix[IDX(i, j, k)];
                double decompressed = decompressedMatrix[IDX(i, j, k)];

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

int main() {
    srand(time(0));

    // ensure the compressors are registered!
    dendrocompression::register_compressors();

    unsigned int eleorder      = 6;
    unsigned int n             = eleorder - 1;
    unsigned int nvar          = 10;

    unsigned int interp_stride = 2;

    std::cout << "N: " << n << " nvar: " << nvar << std::endl;

    int x = n, y = n, z = n;  // Dimensions of matrix. Modify as needed.
    int npts       = x * y * z;
    int total_npts = x * y * z * nvar;

    std::cout << "x, y, z: " << x << ", " << y << ", " << z
              << " -> npts: " << npts << " total_npts: " << total_npts
              << std::endl;

    std::vector<double> fullmatrix(total_npts);
    std::vector<double> decompressed_full(total_npts, 0.0);
    std::vector<double> decompressed_full_second(total_npts, 0.0);
    unsigned char* compressed_buffer =
        static_cast<unsigned char*>(malloc(total_npts * sizeof(double)));

    uint32_t offset = 0;
    for (int ii = 0; ii < nvar; ii++) {
        fillMatrixWave(&fullmatrix[offset], x, y, z, 1.0, (ii + 1) * 1.0, 0.0,
                       1.0, 1.0, 1.0, 0.01, 0.01, 0.01);

        fillMatrixRandom(x, y, z, 1000000000000, &decompressed_full[offset]);
        fillMatrixRandom(x, y, z, 1000000000000,
                         &decompressed_full_second[offset]);

        offset += npts;
    }
    std::cout << "FINISHED INITALIZING DATA!" << std::endl;

    std::size_t originalMatrixBytes = total_npts * sizeof(double);

    std::cout << "BUILDING THE COMPRESSOR" << std::endl;
    std::unique_ptr<dendrocompression::Compression<double>> compressor =
        dendrocompression::doubleCompressor.create(
            dendrocompression::CompressionType::COMP_INTERP,
            {eleorder, nvar, interp_stride});

    std::cout << " COMPRESSOR type:" << compressor->to_string() << std::endl;

    // compress the data
    std::size_t compressed_bytes =
        compressor->do_compress_3d(fullmatrix.data(), compressed_buffer, 1);

    // do decompression
    std::size_t compressed_bytes_2 = compressor->do_decompress_3d(
        compressed_buffer, decompressed_full.data(), 1);

    std::cout << "Original matrix size: " << originalMatrixBytes << " bytes"
              << std::endl;
    std::cout << "Compressed buffer size: " << compressed_bytes << " bytes"
              << std::endl;
    std::cout << "Original/Compressed bytes (compression ratio): "
              << (double)originalMatrixBytes / (double)compressed_bytes
              << std::endl;
    // Printing various types of error between original and decompressed data
    printError(fullmatrix, decompressed_full);

#if 0
    std::cout << "\n\n==== Sending as Chunks" << std::endl;

    // now test compressing a few chunks at a time
    int all_compressedSize      = 0;
    uint32_t decompressedOffset = 0;
    for (int ii = 0; ii < num_matrices; ii++) {
        int newCompressedSize;

        unsigned char* newCompressedMatrix = ZFPAlgorithms::compressMatrix1D(
            &fullmatrix[decompressedOffset], n, rate, newCompressedSize);

        // Decompress the matrix
        ZFPAlgorithms::decompressMatrix1D(
            newCompressedMatrix, newCompressedSize,
            &decompressed_full_second[decompressedOffset]);

        all_compressedSize += newCompressedSize;

        decompressedOffset += total_points;

        delete[] newCompressedMatrix;
    }

    std::cout << "Original matrix size: " << originalMatrixBytes << " bytes"
              << std::endl;
    std::cout << "Compressed matrix size: " << all_compressedSize << " bytes"
              << std::endl;
    std::cout << "Compressed/Original: "
              << (double)all_compressedSize / (double)originalMatrixBytes
              << std::endl;
    // Printing various types of error between original and decompressed data
    printError(fullmatrix, decompressed_full_second, x, y, z);

    // printComparison(fullmatrix, decompressed_full_second, x, y, z);

    // for (int ii = 0; ii < num_matrices; ii++) {
    //     // Define original matrix
    //     // double* originalMatrix = createMatrixWave(x, y, z, 3.1415, 10.0,
    //     // 0, 1.0,
    //     //                                           2.0, 7.0, 0.1, 0.1,
    //     0.15); double* originalMatrix = createMatrixRandom(x, y, z, 0.01 *
    //     (ii + 1)); int originalMatrixBytes = x * y * z * sizeof(double);

    //     // Compress the matrix
    //     int compressedSize;
    //     unsigned char* compressedMatrix = ZFPAlgorithms::compressMatrix1D(
    //         originalMatrix, n, rate, compressedSize);

    //     double* decompressedMatrix = new double[x * y * z];

    //     // Decompress the matrix
    //     ZFPAlgorithms::decompressMatrix1D(compressedMatrix, compressedSize,
    //                                       decompressedMatrix);

    //     // Printing comparison of original and decompressed data
    //     printComparison(originalMatrix, decompressedMatrix, x, y, z);

    //     std::cout << "Original matrix size: " << originalMatrixBytes << "
    //     bytes"
    //               << std::endl;
    //     std::cout << "Compressed matrix size: " << compressedSize << " bytes"
    //               << std::endl;
    //     // Printing various types of error between original and decompressed
    //     // data
    //     printError(originalMatrix, decompressedMatrix, x, y, z);
    // }
#endif

    // Freeing the memory
    free(compressed_buffer);

    return 0;
}
