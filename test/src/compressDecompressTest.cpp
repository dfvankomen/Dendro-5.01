#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "compression.h"

#define IDX(i, j, k) ((i)*y * z + (j)*z + (k))  // 3D to 1D indexing

#define UNIFORM_RAND_0_TO_X(X) ((double)rand() / (double)RAND_MAX * X)

double* createMatrixWave(int x, int y, int z, double frequency,
                         double amplitude, double phase, double fx, double fy,
                         double fz, double dx, double dy, double dz) {
    // Allocate a 1D array to hold the wave data. The size of the array is the
    // product of the dimensions x, y, and z.
    double* matrix = new double[x * y * z];

    // Loop over each index in the x dimension.
    for (int i = 0; i < x; i++) {
        double xx = i * dx;
        // Loop over each index in the y dimension.
        for (int j = 0; j < y; j++) {
            double yy = j * dy;
            // Loop over each index in the z dimension.
            for (int k = 0; k < z; k++) {
                double zz = k * dz;
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
                matrix[IDX(i, j, k)] = waveValue;
            }
        }
    }

    // Return the 1D array containing the wave data.
    return matrix;
}

void fillMatrixRandom(int x, int y, int z, double amplitude, double* vec) {
    // Loop over each index in the x dimension.
    for (int i = 0; i < x; i++) {
        // Loop over each index in the y dimension.
        for (int j = 0; j < y; j++) {
            // Loop over each index in the z dimension.
            for (int k = 0; k < z; k++) {
                // Calculate the value of the wave at the point (i, j, k). The
                // value is determined by a sine function, which is influenced
                // by the frequency, amplitude, and phase input parameters, as
                // well as the propagation factors fx, fy, and fz for each
                // dimension. The result is a wave that can propagate
                // differently along the x, y, and z dimensions.
                double waveValue = amplitude * (UNIFORM_RAND_0_TO_X(2) - 1);

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

void printError(const double* originalMatrix, const double* decompressedMatrix,
                int x, int y, int z) {
    // Computing and printing Mean Squared Error, Mean Absolute Error,
    // Maximum Absolute Error, Root Mean Squared Error and Peak Signal to Noise
    // Ratio.
    double mse = 0;
    for (size_t i = 0; i < x * y * z; ++i) {
        double error = originalMatrix[i] - decompressedMatrix[i];
        mse += error * error;
    }
    mse /= (x * y * z);

    std::cout << "Mean Squared Error: " << mse << "\n";
    // Compute and print Mean Absolute Error
    double mae = 0;
    for (size_t i = 0; i < x * y * z; ++i) {
        double error = std::abs(originalMatrix[i] - decompressedMatrix[i]);
        mae += error;
    }
    mae /= (x * y * z);
    std::cout << "Mean Absolute Error: " << mae << "\n";

    // Compute and print Maximum Absolute Error
    double maxError = 0;
    for (size_t i = 0; i < x * y * z; ++i) {
        double error = std::abs(originalMatrix[i] - decompressedMatrix[i]);
        maxError = std::max(maxError, error);
    }
    std::cout << "Max Absolute Error: " << maxError << "\n";

    // Compute and print Root Mean Squared Error
    double rmse = sqrt(mse);
    std::cout << "Root Mean Squared Error: " << rmse << "\n";

    // Compute and print Peak Signal to Noise Ratio (PSNR)
    double maxOriginalValue =
        *std::max_element(originalMatrix, originalMatrix + (x * y * z));
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
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                double original = originalMatrix[IDX(i, j, k)];
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
    int x = 50, y = 50, z = 20;  // Dimensions of matrix. Modify as needed.
    int n = x * y * z;
    int k = 7;
    double rate = 64.0;

    int total_points = x * y * z;

    int num_matrices = 10;

    double* fullmatrix = new double[total_points * num_matrices];
    double* decompressed_full = new double[total_points * num_matrices]();
    double* decompressed_full_second = new double[total_points * num_matrices]();

    uint32_t offset = 0;
    for (int ii = 0; ii < num_matrices; ii++) {
        fillMatrixRandom(x, y, z, 100 * (ii + 1), &fullmatrix[offset]);

        fillMatrixRandom(x, y, z, 1000000000000, &decompressed_full[offset]);
        fillMatrixRandom(x, y, z, 1000000000000, &decompressed_full_second[offset]);

        offset += total_points;
    }

    int originalMatrixBytes = x * y * z * num_matrices * sizeof(double);

    // try compressing the whole thing!
    // Compress the matrix
    int compressedSize;
    unsigned char* compressedMatrix = ZFPAlgorithms::compressMatrix1D(
        fullmatrix, total_points * num_matrices, rate, compressedSize);

    // Decompress the matrix
    ZFPAlgorithms::decompressMatrix1D(compressedMatrix, compressedSize,
                                      decompressed_full);

    std::cout << "Original matrix size: " << originalMatrixBytes << " bytes"
              << std::endl;
    std::cout << "Compressed matrix size: " << compressedSize << " bytes"
              << std::endl;
    std::cout << "Compressed/Original: " << (double)compressedSize / (double)originalMatrixBytes << std::endl;
    // Printing various types of error between original and decompressed data
    printError(fullmatrix, decompressed_full, x, y, z);

    std::cout << "\n\n==== Sending as Chunks" << std::endl;

    // now test compressing a few chunks at a time
    int all_compressedSize = 0;
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
    std::cout << "Compressed/Original: " << (double)all_compressedSize / (double)originalMatrixBytes << std::endl;
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

    // Freeing the memory
    delete[] fullmatrix;
    delete[] decompressed_full;
    delete[] decompressed_full_second;
    delete[] compressedMatrix;

    return 0;
}