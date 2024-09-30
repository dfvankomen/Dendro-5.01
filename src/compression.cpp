#include "compression.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "zfp.h"
#include "zfp/bitstream.h"
// #include "mpi.h"

namespace SVDAlgorithms {
unsigned char* compressMatrix(double*& originalMatrix, const int x, const int y,
                              const int z, int k, int& buffer_size) {
    // Convert to Eigen's VectorXd for easier manipulation
    Eigen::VectorXd eigenMatrix =
        Eigen::Map<Eigen::VectorXd>(originalMatrix, x * y * z);

    // Validate k based on the dimensions of the reshaped matrix
    int min_dimension = std::min(x * y, z);
    if (k < 1 || k > min_dimension) {
        k = 1;
    }

    // Reshape originalMatrix into a 2D matrix
    Eigen::MatrixXd dataMatrix(x * y, z);
    for (int i = 0; i < x * y; i++) {
        for (int j = 0; j < z; j++) {
            dataMatrix(i, j) = eigenMatrix[i * z + j];
        }
    }

    // Perform SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        dataMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Now we will get the first k columns of U and V, and the first k singular
    // values
    Eigen::MatrixXd U = svd.matrixU().leftCols(k);
    Eigen::MatrixXd V = svd.matrixV().leftCols(k);
    Eigen::VectorXd S = svd.singularValues().head(k);

    // Serialize the matrices and metadata into a byte stream
    int U_rows        = U.rows();
    int U_cols        = U.cols();
    int V_rows        = V.rows();
    int V_cols        = V.cols();
    int S_length      = S.size();

    int U_size        = U.size() * sizeof(double);
    int V_size        = V.size() * sizeof(double);
    int S_size        = S.size() * sizeof(double);
    int header_size =
        8 * sizeof(int);  // x, y, z, U_rows, U_cols, V_rows, V_cols, S_size
    buffer_size               = header_size + U_size + V_size + S_size;

    unsigned char* byteStream = new unsigned char[buffer_size];
    unsigned char* ptr        = byteStream;

    // Serialize the metadata into the byte stream
    memcpy(ptr, &x, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &y, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &z, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &U_rows, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &U_cols, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &V_rows, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &V_cols, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &S_length, sizeof(int));
    ptr += sizeof(int);
    // Now serialize the matrix and vector data
    // For U
    // Ret
    for (int i = 0; i < U_rows; ++i) {
        for (int j = 0; j < U_cols; ++j) {
            double value = U(i, j);
            memcpy(ptr, &value, sizeof(double));
            ptr += sizeof(double);
        }
    }
    // For V
    for (int i = 0; i < V_rows; ++i) {
        for (int j = 0; j < V_cols; ++j) {
            double value = V(i, j);
            memcpy(ptr, &value, sizeof(double));
            ptr += sizeof(double);
        }
    }

    // For S
    for (int i = 0; i < S_length; ++i) {
        double value = S(i);
        memcpy(ptr, &value, sizeof(double));
        ptr += sizeof(double);
    }
    return byteStream;
}

double* decompressMatrix(unsigned char*& byteStream, int buffer_size) {
    unsigned char* ptr = byteStream;

    // Deserialize the metadata
    int x, y, z;
    memcpy(&x, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&y, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&z, ptr, sizeof(int));
    ptr += sizeof(int);
    int U_rows, U_cols, V_rows, V_cols, S_length;
    memcpy(&U_rows, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&U_cols, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&V_rows, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&V_cols, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&S_length, ptr, sizeof(int));
    ptr += sizeof(int);

    // Deserialize the matrices
    Eigen::MatrixXd U(U_rows, U_cols);
    for (int i = 0; i < U_rows; ++i) {
        for (int j = 0; j < U_cols; ++j) {
            double value;
            memcpy(&value, ptr, sizeof(double));
            U(i, j) = value;
            ptr += sizeof(double);
        }
    }
    // For V
    Eigen::MatrixXd V(V_rows, V_cols);
    for (int i = 0; i < V_rows; ++i) {
        for (int j = 0; j < V_cols; ++j) {
            double value;
            memcpy(&value, ptr, sizeof(double));
            V(i, j) = value;
            ptr += sizeof(double);
        }
    }

    // For S
    Eigen::VectorXd S(S_length);
    for (int i = 0; i < S_length; ++i) {
        double value;
        memcpy(&value, ptr, sizeof(double));
        S(i) = value;
        ptr += sizeof(double);
    }

    // Perform matrix multiplication to decompress the data
    Eigen::MatrixXd decompressedMatrix = U * S.asDiagonal() * V.transpose();

    // Reshape and convert the decompressed matrix to a raw pointer
    double* decompressedData           = new double[x * y * z];
    for (int i = 0; i < x * y; i++) {
        for (int j = 0; j < z; j++) {
            decompressedData[i * z + j] = decompressedMatrix(i, j);
        }
    }

    return decompressedData;
}

/*
// Example how to use comperssion algorithm

int main() {
        int x = 3, y = 7, z = 7; // Dimensions of matrix. Modify as needed.
        int k = 3; // min(x*y, z)

        // Define original matrix
        double* originalMatrix = Utilities::createMatrixWave(x, y, z, 1, 3.1415,
0, 1.0, 1.0, 7); int originalMatrixBytes = x * y * z * sizeof(double);

        // Compress the matrix
        int compressedSize;
        unsigned char* compressedMatrix =
SVDAlgorithms::compressMatrix(originalMatrix, x, y, z, k, compressedSize);

        // Decompress the matrix
        double* decompressedMatrix =
SVDAlgorithms::decompressMatrix(compressedMatrix, compressedSize);

        // Printing comparison of original and decompressed data
        Utilities::printComparison(originalMatrix, decompressedMatrix, x, y, z);

        std::cout << "Original matrix size: " << originalMatrixBytes << " bytes"
<< std::endl; std::cout << "Compressed matrix size: " << compressedSize << "
bytes" << std::endl;
        // Printing various types of error between original and decompressed
data Utilities::printError(originalMatrix, decompressedMatrix, x, y, z);

        // Freeing the memory
        delete[] originalMatrix;
        delete[] compressedMatrix;
        delete[] decompressedMatrix;

        return 0;
}
*/

std::pair<int, int> findSquareLikeDimensions(int n) {
    if (n < 2) {
        return {1, n};
    }
    for (int i = static_cast<int>(sqrt(n)); i >= 1; --i) {
        if (n % i == 0) {
            return {i, n / i};
        }
    }
    return {2, (n / 2) + 1};  // if n is prime
}

unsigned char* compressMatrix1d(double*& originalMatrix, const int n,
                                int cmp_str, int& buffer_size) {
    // Find 2D dimensions
    std::pair<int, int> dimensions = findSquareLikeDimensions(n);
    int rows                       = dimensions.first;
    int cols                       = dimensions.second;

    // Handle the case when n is prime
    bool fill_zero                 = (rows * cols > n);

    // Set k based on cmp_str
    int k                          = std::min(rows, cols);
    k                              = k - cmp_str;
    if (k < 1) {
        k = 1;
    }

    // Reshape 1D to 2D matrix
    Eigen::MatrixXd dataMatrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx          = i * cols + j;
            dataMatrix(i, j) = (idx < n) ? originalMatrix[idx] : 0;
        }
    }

    // Perform SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        dataMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Now we will get the first k columns of U and V, and the first k singular
    // values
    Eigen::MatrixXd U = svd.matrixU().leftCols(k);
    Eigen::MatrixXd V = svd.matrixV().leftCols(k);
    Eigen::VectorXd S = svd.singularValues().head(k);

    // Serialize the matrices and metadata into a byte stream
    int U_rows        = U.rows();
    int U_cols        = U.cols();
    int V_rows        = V.rows();
    int V_cols        = V.cols();
    int S_length      = S.size();

    int U_size        = U.size() * sizeof(double);
    int V_size        = V.size() * sizeof(double);
    int S_size        = S.size() * sizeof(double);
    int header_size =
        7 * sizeof(int) + sizeof(bool);  // rows, cols, U_rows, U_cols, V_rows,
                                         // V_cols, S_size, fill_zero
    buffer_size               = header_size + U_size + V_size + S_size;

    unsigned char* byteStream = new unsigned char[buffer_size];
    unsigned char* ptr        = byteStream;

    // Serialize the metadata into the byte stream
    memcpy(ptr, &rows, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &cols, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &U_rows, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &U_cols, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &V_rows, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &V_cols, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &S_length, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &fill_zero, sizeof(bool));
    ptr += sizeof(bool);

    // Now serialize the matrix and vector data
    // For U
    // Ret
    for (int i = 0; i < U_rows; ++i) {
        for (int j = 0; j < U_cols; ++j) {
            double value = U(i, j);
            memcpy(ptr, &value, sizeof(double));
            ptr += sizeof(double);
        }
    }
    // For V
    for (int i = 0; i < V_rows; ++i) {
        for (int j = 0; j < V_cols; ++j) {
            double value = V(i, j);
            memcpy(ptr, &value, sizeof(double));
            ptr += sizeof(double);
        }
    }

    // For S
    for (int i = 0; i < S_length; ++i) {
        double value = S(i);
        memcpy(ptr, &value, sizeof(double));
        ptr += sizeof(double);
    }
    return byteStream;
}

double* decompressMatrix1d(unsigned char*& byteStream, int buffer_size) {
    unsigned char* ptr = byteStream;

    // Deserialize the metadata
    int rows, cols;
    memcpy(&rows, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&cols, ptr, sizeof(int));
    ptr += sizeof(int);
    int U_rows, U_cols, V_rows, V_cols, S_length;
    memcpy(&U_rows, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&U_cols, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&V_rows, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&V_cols, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&S_length, ptr, sizeof(int));
    ptr += sizeof(int);
    bool fill_zero;
    memcpy(&fill_zero, ptr, sizeof(bool));
    ptr += sizeof(bool);

    // Deserialize the matrices
    Eigen::MatrixXd U(U_rows, U_cols);
    for (int i = 0; i < U_rows; ++i) {
        for (int j = 0; j < U_cols; ++j) {
            double value;
            memcpy(&value, ptr, sizeof(double));
            U(i, j) = value;
            ptr += sizeof(double);
        }
    }
    // For V
    Eigen::MatrixXd V(V_rows, V_cols);
    for (int i = 0; i < V_rows; ++i) {
        for (int j = 0; j < V_cols; ++j) {
            double value;
            memcpy(&value, ptr, sizeof(double));
            V(i, j) = value;
            ptr += sizeof(double);
        }
    }

    // For S
    Eigen::VectorXd S(S_length);
    for (int i = 0; i < S_length; ++i) {
        double value;
        memcpy(&value, ptr, sizeof(double));
        S(i) = value;
        ptr += sizeof(double);
    }

    // Perform matrix multiplication to decompress the data
    Eigen::MatrixXd decompressedMatrix = U * S.asDiagonal() * V.transpose();

    int n                    = rows * cols - static_cast<int>(fill_zero);
    double* decompressedData = new double[n];

    // Fill the decompressedData
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            if (idx < n) {
                decompressedData[idx] = decompressedMatrix(i, j);
            }
        }
    }
    return decompressedData;
}
}  // namespace SVDAlgorithms

namespace ChebyshevAlgorithms {

// a "global" object that we can call from pretty much anywhere
ChebyshevCompression cheby{6, 3};

// Type alias for cache key, where first is the degree n and second is the value
// x
using CacheKey = std::pair<int, double>;

// Custom hash function for CacheKey
struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const {
        return std::hash<int>()(key.first) ^ std::hash<double>()(key.second);
    }
};

thread_local std::unordered_map<CacheKey, double, CacheKeyHash> cache;

// Type alias for cache key
struct SVDKey {
    int x;
    int y;
    int z;

    bool operator==(const SVDKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Custom hash function for SVDKey
struct SVDKeyHash {
    std::size_t operator()(const SVDKey& key) const {
        return std::hash<int>()(key.x) ^ std::hash<int>()(key.y) ^
               std::hash<int>()(key.z);
    }
};

// SVD struct
struct SVDs {
    Eigen::JacobiSVD<Eigen::MatrixXd> svdA;
    Eigen::JacobiSVD<Eigen::MatrixXd> svdB;
    Eigen::JacobiSVD<Eigen::MatrixXd> svdC;
};

// Thread-local cache for SVD objects
thread_local std::unordered_map<SVDKey, SVDs, SVDKeyHash> svdCache;

Eigen::MatrixXd kroneckerProduct(const Eigen::MatrixXd& A,
                                 const Eigen::MatrixXd& B) {
    Eigen::MatrixXd C(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            C.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) =
                A(i, j) * B;
        }
    }
    return C;
}

double chebyshevT(int n, double x) {
    CacheKey key = {n, x};

    // Check if value is in cache
    auto iter    = cache.find(key);
    if (iter != cache.end()) {
        return iter->second;
    }

    // Calculate Chebyshev polynomial
    double result;
    if (n == 0) {
        result = 1.0;
    } else if (n == 1) {
        result = x;
    } else {
        double T_prev = 1.0;
        double T_curr = x;
        for (int i = 2; i <= n; ++i) {
            double T_next = 2 * x * T_curr - T_prev;
            T_prev        = T_curr;
            T_curr        = T_next;
        }
        result = T_curr;
    }

    // Store result in cache and return
    cache[key] = result;
    return result;
}

Eigen::MatrixXd getFullKronProductTruncated(int x, int y, int z, int N, int Q,
                                            int S) {
    // Calculate A, B, and C, where A = x x N, B = y x Q, and C = z x S
    Eigen::MatrixXd A(x, N);
    Eigen::MatrixXd B(y, Q);
    Eigen::MatrixXd C(z, S);

    // Populate A using Chebyshev polynomials
    for (int i = 0; i < x; ++i) {
        double normalized_x = static_cast<double>(i) / (x - 1);
        for (int n = 0; n < N; ++n) {
            A(i, n) = ChebyshevAlgorithms::chebyshevT(n, normalized_x);
        }
    }

    // Populate B using Chebyshev polynomials
    for (int j = 0; j < y; ++j) {
        double normalized_y = static_cast<double>(j) / (y - 1);
        for (int q = 0; q < Q; ++q) {
            B(j, q) = ChebyshevAlgorithms::chebyshevT(q, normalized_y);
        }
    }

    // Populate C using Chebyshev polynomials
    for (int k = 0; k < z; ++k) {
        double normalized_z = static_cast<double>(k) / (z - 1);
        for (int s = 0; s < S; ++s) {
            C(k, s) = ChebyshevAlgorithms::chebyshevT(s, normalized_z);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svdA;
    Eigen::JacobiSVD<Eigen::MatrixXd> svdB;
    Eigen::JacobiSVD<Eigen::MatrixXd> svdC;

    // Define  key based on whatever criteria makes matrices A, B, C identical
    // or close enough
    SVDKey key{x, y, z};

    // Check if the SVDs for this key are already computed and stored in the
    // cache
    auto it = svdCache.find(key);
    if (it != svdCache.end()) {
        // Cache hit: SVD results are already computed
        SVDs cachedSVDs = it->second;
        svdA            = cachedSVDs.svdA;
        svdB            = cachedSVDs.svdB;
        svdC            = cachedSVDs.svdC;
    } else {
        // Cache miss: Need to perform SVD and store the results in the cache
        svdA = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeFullU);
        svdB = Eigen::JacobiSVD<Eigen::MatrixXd>(B, Eigen::ComputeFullU);
        svdC = Eigen::JacobiSVD<Eigen::MatrixXd>(C, Eigen::ComputeFullU);

        SVDs newSVDs{svdA, svdB, svdC};
        svdCache[key] = newSVDs;
    }

    // Truncate U matrices
    Eigen::MatrixXd UA_truncated = svdA.matrixU().leftCols(N);
    Eigen::MatrixXd UB_truncated = svdB.matrixU().leftCols(Q);
    Eigen::MatrixXd UC_truncated = svdC.matrixU().leftCols(S);

    // Calculate the Kronecker product with truncated U matrices
    Eigen::MatrixXd UA_UB_kron   = kroneckerProduct(UA_truncated, UB_truncated);
    Eigen::MatrixXd full_kron_product_truncated =
        kroneckerProduct(UA_UB_kron, UC_truncated);
    return full_kron_product_truncated;
}

void compressMatrixBuffer(const double* originalMatrix, const uint32_t x,
                          const uint32_t y, const uint32_t z, const uint32_t N,
                          const uint32_t Q, const uint32_t S,
                          unsigned char* outputArray, int& bufferSize) {
    // Normalize the data inside originalMatrix
    double minVal =
        *std::min_element(originalMatrix, originalMatrix + x * y * z);
    double maxVal =
        *std::max_element(originalMatrix, originalMatrix + x * y * z);
    double range             = maxVal - minVal;

    ChebyshevData dataToPack = {x, y, z, N, Q, S, minVal, maxVal};

    // Create normalizedMatrix as Eigen::VectorXd
    Eigen::VectorXd normalizedMatrix(x * y * z);
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                normalizedMatrix(i * y * z + j * z + k) =
                    ((originalMatrix[i * y * z + j * z + k] - minVal) / range) *
                        2.0 -
                    1.0;
            }
        }
    }

    Eigen::MatrixXd full_kron_product_truncated =
        getFullKronProductTruncated(x, y, z, N, Q, S);
    // std::cout << "Full Kron Product: " << full_kron_product_truncated.size()
    //           << " via " << full_kron_product_truncated.rows() << " "
    //           << full_kron_product_truncated.cols() << std::endl;

    // Calculate the coefficients using the truncated Kronecker product
    Eigen::VectorXd coefficients_truncated =
        full_kron_product_truncated.transpose() * normalizedMatrix;

    // std::cout << "Coeffs truncated..." << coefficients_truncated.size()
    //           << std::endl;

    bufferSize         = sizeof(dataToPack) + (N * Q * S * sizeof(double));

    // std::cout << "buffer size: " << bufferSize << " (" << sizeof(dataToPack)
    //           << ", " << N * Q * S * sizeof(double) << ")" << std::endl;

    unsigned char* ptr = outputArray;

    memcpy(ptr, &dataToPack, sizeof(dataToPack));
    ptr += sizeof(dataToPack);

    // then copy the coefficients_truncated
    memcpy(ptr, coefficients_truncated.data(),
           coefficients_truncated.size() * sizeof(double));

    // for (size_t i = 0; i < coefficients_truncated.size(); ++i) {
    //     memcpy(ptr, &coefficients_truncated[i], sizeof(double));
    //     ptr += sizeof(double);
    // }
}

unsigned char* compressMatrix(const double* originalMatrix, const uint32_t x,
                              const uint32_t y, const uint32_t z,
                              const uint32_t N, const uint32_t Q,
                              const uint32_t S, int& bufferSize) {
    bufferSize = sizeof(ChebyshevData) + (N * Q * S * sizeof(double));
    unsigned char* buffer = new unsigned char[bufferSize];

    // then just call the compressMatrixBfr function
    compressMatrixBuffer(originalMatrix, x, y, z, N, Q, S, buffer, bufferSize);

    return buffer;
}

double* decompressMatrix(const unsigned char* buffer, const int bufferSize) {
    unsigned char* ptr = (unsigned char*)buffer;

    ChebyshevData data{};
    memcpy(&data, ptr, sizeof(data));

    // create the array that we're going to send back
    double* reconstructedArray = new double[data.x * data.y * data.z];

    // then just call the decompress buffer function
    decompressMatrixBuffer(buffer, bufferSize, reconstructedArray);

    // return our array
    return reconstructedArray;
}

void decompressMatrixBuffer(const unsigned char* buffer, const int bufferSize,
                            double* outBuff) {
    unsigned char* ptr = (unsigned char*)buffer;

    ChebyshevData data;
    memcpy(&data, ptr, sizeof(data));
    ptr += sizeof(data);

    // Unpack the coefficients
    int numCoefficients = data.N * data.Q * data.S;
    Eigen::VectorXd coefficients_truncated(numCoefficients);

    memcpy(coefficients_truncated.data(), ptr,
           numCoefficients * sizeof(double));

    Eigen::MatrixXd full_kron_product_truncated = getFullKronProductTruncated(
        data.x, data.y, data.z, data.N, data.Q, data.S);

    // Reconstruct the matrix using the coefficients and the truncated Kronecker
    // product
    Eigen::VectorXd reconstructedMatrix =
        full_kron_product_truncated * coefficients_truncated;

    // do the unnormalize
    double range = data.maxVal - data.minVal;
    reconstructedMatrix *= range / 2.0;
    reconstructedMatrix.array() += data.minVal + (range / 2.0);

    // Convert Eigen::VectorXd back to double*
    memcpy(outBuff, reconstructedMatrix.data(),
           data.x * data.y * data.z * sizeof(double));
}

/*
int main() {
        int x = 3, y = 7, z = 7;
        double* originalMatrix = Utilities::createMatrixWave(x, y, z, 0.1, 0.1,
0.1, 1.0, 1.0, 1);
        // The first degree polynomials for each dimension
        int N = 3;
        int Q = 5;
        int S = 5;

        int bufferSize;

        unsigned char* buffer =
ChebyshevAlgorithms::compressMatrix(originalMatrix, x, y, z, N, Q, S,
bufferSize); double* decompressedMatrix =
ChebyshevAlgorithms::decompressMatrix(buffer, bufferSize);

        Utilities::printComparison(originalMatrix, decompressedMatrix, x, y, z);
        Utilities::printError(originalMatrix, decompressedMatrix, x, y, z);

        // Printing size of original and serialized compressed data
        std::cout << "Size of original matrix (bytes): " << (x * y * z) *
sizeof(double) << "\n"; std::cout << "Size of serialized compressed data
(bytes): " << bufferSize << "\n";

        delete[] decompressedMatrix;
        delete[] originalMatrix;

        return 0;

}
*/
}  // namespace ChebyshevAlgorithms

#if 0

namespace FFTAlgorithms {

	/*
	Bitmask Usage Example:

	frequency matrix: [1 + 2i, .001 + .001i, .002 + .001i, 3 + 4i]
	zero out: [1 + 2i, 0, 0, 3 + 4i]
	bitmask: 1001
	bytearray: (x), (y), (z), (bitmask),(8 bytes for 0th component),(8 bytes for 3rd component)

	decompression:
	frequency matrix:  [1 + 2i, 0, 0, 3 + 4i]
	notice there were only two components in the bytearray but the bitmask keeps track of when to fill components in.
	*/

	bool compareMagnitude(const MagnitudeIndexPair& a, const MagnitudeIndexPair& b) {
		return a.first < b.first;
	}

	/**
	 * @brief Compresses a 3D matrix using Fast Fourier Transform (FFT).
	 *
	 * This function performs FFT on a given 3D matrix and then compresses the result based on a provided compression ratio.
	 * The compressed data and a bitmask indicating which components are non-zero are stored in a single byte stream.
	 *
	 * @param originalMatrix Pointer to the original 3D matrix of doubles to be compressed.
	 * @param x The size of the matrix along the x-axis.
	 * @param y The size of the matrix along the y-axis.
	 * @param z The size of the matrix along the z-axis.
	 * @param compressionRatio Ratio of frequency components to retain (range from 0 to 1).
	 *        If given a value greater than 1, it is set to 1.
	 * @param[out] size Output parameter indicating the size of the compressed data.
	 *
	 * @return A pointer to the byte stream containing the compressed data.
	 */
	unsigned char* compressMatrix(double* originalMatrix, int x, int y, int z, double compressionRatio, int& size) {
		if (compressionRatio > 1)
		{
			compressionRatio = 1;
		}
		// Performing FFT on the original matrix
		ComplexVec complexMatrix(x * y * z);
		fftw_plan p = fftw_plan_dft_r2c_3d(x, y, z, originalMatrix,
			reinterpret_cast<fftw_complex*>(complexMatrix.data()), FFTW_ESTIMATE);
		fftw_execute(p);
		fftw_destroy_plan(p);

		// Get magnitudes and corresponding indices
		std::vector<MagnitudeIndexPair> magnitudes;
		for (int i = 0; i < x * y * z; ++i) {
			magnitudes.push_back(MagnitudeIndexPair(std::abs(complexMatrix[i]), i));
		}

		// Sort the magnitudes
		std::sort(magnitudes.begin(), magnitudes.end(), compareMagnitude);

		// Determine how many frequency components to keep
		int numToKeep = static_cast<int>(compressionRatio * x * y * z);

		// Prepare the bitmask
		int bitMaskSize = (x * y * z + 7) / 8;
		unsigned char* bitMask = new unsigned char[bitMaskSize];
		memset(bitMask, 0, bitMaskSize);

		// Compute the size of the data and prepare the buffer
		size = bitMaskSize + numToKeep * sizeof(std::complex<double>) + 3 * sizeof(int);
		unsigned char* byteStream = new unsigned char[size];
		unsigned char* dataPtr = byteStream + bitMaskSize + 3 * sizeof(int);

		// Zero out least significant frequency components
		for (int i = 0; i < numToKeep; ++i) {
			int idx = magnitudes[x * y * z - 1 - i].second;
			bitMask[idx / 8] |= (1 << (idx % 8));
		}

		// Zero out least significant frequency components
		int numToZeroOut = x * y * z - numToKeep; // Number of components to zero out
		for (int i = 0; i < numToZeroOut; ++i) {
			int idx = magnitudes[i].second; // Get the index of the least significant component
			complexMatrix[idx] = 0.0; // Set that component to zero
		}

		// Iterate over the unsorted complex matrix and copy non-zero values to byte stream
		for (int idx = 0; idx < x * y * z; ++idx) {
			if (complexMatrix[idx] != 0.0) { // Check if the component is non-zero
				std::memcpy(dataPtr, &complexMatrix[idx], sizeof(std::complex<double>));
				dataPtr += sizeof(std::complex<double>);
			}
		}

		// Copy the bitmask to the byteStream
		std::memcpy(byteStream + 3 * sizeof(int), bitMask, bitMaskSize);

		// Write x, y, z to the start of the bytestream
		std::memcpy(byteStream, &x, sizeof(int));
		std::memcpy(byteStream + sizeof(int), &y, sizeof(int));
		std::memcpy(byteStream + 2 * sizeof(int), &z, sizeof(int));

		delete[] bitMask;

		return byteStream;
	}

	double* decompressMatrix(unsigned char* byteStream, int byteStreamSize) {
		// Read x, y, z from the start of the bytestream
		int x, y, z;
		std::memcpy(&x, byteStream, sizeof(int));
		std::memcpy(&y, byteStream + sizeof(int), sizeof(int));
		std::memcpy(&z, byteStream + 2 * sizeof(int), sizeof(int));

		// Calculate the size of the bitmask
		int bitMaskSize = (x * y * z + 7) / 8;

		// Calculate the number of frequency components to keep
		int numToKeep = (byteStreamSize - bitMaskSize - 3 * sizeof(int)) / sizeof(std::complex<double>);

		// Copy the bitmask from the byteStream
		unsigned char* bitMask = new unsigned char[bitMaskSize];
		std::memcpy(bitMask, byteStream + 3 * sizeof(int), bitMaskSize);

		// Prepare to read the data
		unsigned char* dataPtr = byteStream + bitMaskSize + 3 * sizeof(int);

		// Initialize complexMatrix with zeros
		ComplexVec complexMatrix(x * y * z, 0.0);

		// Populate the complexMatrix
		for (int i = 0; i < x * y * z; ++i) {
			if (bitMask[i / 8] & (1 << (i % 8))) {
				std::memcpy(&complexMatrix[i], dataPtr, sizeof(std::complex<double>));
				dataPtr += sizeof(std::complex<double>);
			}
		}

		delete[] bitMask;

		// Perform inverse FFT
		double* decompressedMatrix = new double[x * y * z];
		fftw_plan q = fftw_plan_dft_c2r_3d(x, y, z, reinterpret_cast<fftw_complex*>(complexMatrix.data()),
			decompressedMatrix, FFTW_ESTIMATE);
		fftw_execute(q);
		fftw_destroy_plan(q);

		// FFTW's backwards transform does not normalize the result, so we have to do it manually
		for (int i = 0; i < x * y * z; ++i) {
			decompressedMatrix[i] /= (x * y * z);
		}

		return decompressedMatrix;
	}

	unsigned char* compressMatrix1D(double* originalArray, int n, double threshold, int& size) {
		// Create extended array with sign-flipped mirrored extensions
		int extended_n = 2 * n;
		double* extendedArray = new double[extended_n];
		for (int i = 0; i < n; ++i) {
			extendedArray[i] = originalArray[i];
			extendedArray[extended_n - i - 1] = -originalArray[i];
		}

		// Performing FFT on the extended array
		ComplexVec complexArray(extended_n);
		fftw_plan p = fftw_plan_dft_r2c_1d(extended_n, extendedArray, reinterpret_cast<fftw_complex*>(complexArray.data()), FFTW_ESTIMATE);
		fftw_execute(p);
		fftw_destroy_plan(p);

		delete[] extendedArray;

		// Get magnitudes and corresponding indices
		std::vector<MagnitudeIndexPair> magnitudes;
		for (int i = 0; i < extended_n; ++i) {
			magnitudes.push_back(MagnitudeIndexPair(std::abs(complexArray[i]), i));
		}

		// Prepare the bitmask
		int bitMaskSize = (extended_n + 7) / 8;
		unsigned char* bitMask = new unsigned char[bitMaskSize];
		memset(bitMask, 0, bitMaskSize);

		// Number of components to keep
		int numToKeep = 0;

		// Zero out coefficients below threshold and prepare bitmask
		for (int i = 0; i < extended_n; ++i) {
			if (magnitudes[i].first < threshold) {
				complexArray[magnitudes[i].second] = 0.0;
			}
			else {
				bitMask[i / 8] |= (1 << (i % 8));
				numToKeep++;
			}
		}

		// Compute the size of the data and prepare the buffer
		size = bitMaskSize + numToKeep * sizeof(std::complex<double>) + sizeof(int);
		unsigned char* byteStream = new unsigned char[size];
		unsigned char* dataPtr = sizeof(int) + bitMaskSize + byteStream; // extended_n + bitMaskSize  + byteStream

		// Write extended_n to the start of the bytestream
		std::memcpy(byteStream, &extended_n, sizeof(int));

		// Copy the bitmask and non-zero values to byteStream
		std::memcpy(byteStream + sizeof(int), bitMask, bitMaskSize);
		for (int i = 0; i < extended_n; ++i) {
			if (complexArray[i] != 0.0) {
				std::memcpy(dataPtr, &complexArray[i], sizeof(std::complex<double>));
				dataPtr += sizeof(std::complex<double>);
			}
		}

		delete[] bitMask;

		return byteStream;
	}


	double* decompressMatrix1D(unsigned char* byteStream, int byteStreamSize) {
		// Read extended_n from the start of the bytestream
		int extended_n;
		std::memcpy(&extended_n, byteStream, sizeof(int));

		// Calculate the size of the bitmask
		int bitMaskSize = (extended_n + 7) / 8;

		// Copy the bitmask from the byteStream
		unsigned char* bitMask = new unsigned char[bitMaskSize];
		std::memcpy(bitMask, byteStream + sizeof(int), bitMaskSize);

		// Initialize complexArray with zeros
		ComplexVec complexArray(extended_n, 0.0);

		// Prepare to read the data
		unsigned char* dataPtr = byteStream + bitMaskSize + sizeof(int);

		// Populate the complexArray based on bitmask
		for (int i = 0; i < extended_n; ++i) {
			if (bitMask[i / 8] & (1 << (i % 8))) {
				std::memcpy(&complexArray[i], dataPtr, sizeof(std::complex<double>));
				dataPtr += sizeof(std::complex<double>);
			}
		}

		delete[] bitMask;

		// Perform inverse FFT
		double* decompressedExtendedArray = new double[extended_n];
		fftw_plan q = fftw_plan_dft_c2r_1d(extended_n, reinterpret_cast<fftw_complex*>(complexArray.data()), decompressedExtendedArray, FFTW_ESTIMATE);
		fftw_execute(q);
		fftw_destroy_plan(q);

		// Normalize the result and truncate to original size
		int n = extended_n / 2;
		double* decompressedArray = new double[n];
		for (int i = 0; i < n; ++i) {
			decompressedArray[i] = decompressedExtendedArray[i] / extended_n;
		}

		// Delete the extended array
		delete[] decompressedExtendedArray;

		return decompressedArray;
	}

	/*
	int main() {
		int x = 3, y = 7, z = 7; // Dimensions of matrix. Modify as needed.
		double compressionRatio = .8; // Compression ratio. 0.9 means keeping 90% of the orignial data

		// Define original matrix
		double* originalMatrix = Utilities::createMatrixWave(x, y, z, 1, 3.1415, 0, 1.0, 1.0, 7);

		// Compressing the data
		int byteStreamSize;
		unsigned char* byteStream = FFTAlgorithms::compressMatrix(originalMatrix, x, y, z, compressionRatio, byteStreamSize);

		// Decompressing the received data
		double* decompressedMatrix = FFTAlgorithms::decompressMatrix(byteStream, byteStreamSize);

		// Printing comparison of original and decompressed data
		Utilities::printComparison(originalMatrix, decompressedMatrix, x, y, z);

		// Printing size of original and serialized compressed data
		std::cout << "Size of original matrix (bytes): " << (x * y * z) * sizeof(double) << "\n";
		std::cout << "Size of serialized compressed data (bytes): " << byteStreamSize << "\n";

		// Printing various types of error between original and decompressed data
		Utilities::printError(originalMatrix, decompressedMatrix, x, y, z);

		// Freeing the memory
		delete[] originalMatrix;
		delete[] decompressedMatrix;
		delete[] byteStream;

		return 0;
	}
	*/
}

#endif

namespace ZFPAlgorithms {

// "global" object for ZFP algorithm that can be called by dendro
ZFPCompression zfpblockwise(6, 5.0);

void compressMatrixBuffer(const double* originalData, const uint32_t x,
                          const uint32_t y, const uint32_t z, double rate,
                          unsigned char* outBuffer, int& size, zfp_field* field,
                          zfp_stream* zfp) {
    bool closeStream = false;
    bool freeField   = false;

    if (field == nullptr && zfp == nullptr) {
        // Initialize a 3D array with original data, using ZFP's special 'field'
        // structure. The field has the type of double and the dimensions are
        // given by x, y, and z.
        field = zfp_field_3d((double*)originalData, zfp_type_double, x, y, z);

        // Open a new ZFP stream. A ZFP stream is responsible for compressing
        // and decompressing data.
        zfp   = zfp_stream_open(NULL);

        closeStream = true;
        freeField   = true;
    } else if (field != nullptr && zfp == nullptr) {
        throw std::invalid_argument(
            "Received a null field and a non-null zfp stream! This doesn't "
            "work!");
    } else if (field == nullptr && zfp != nullptr) {
        // then just create the field, because the zfp object could probably be
        // open
        field = zfp_field_3d((double*)originalData, zfp_type_double, x, y, z);

        freeField = true;
    }

    // Set the compression rate for the ZFP stream. The type of the data is
    // double, the dimensionality is 3, and '0' indicates we're not using a
    // user-specified precision.
    zfp_stream_set_rate(zfp, rate, zfp_type_double, 3, 0);

    // Determine the maximum buffer size necessary for this ZFP stream given the
    // input field.
    int bufsize        = zfp_stream_maximum_size(zfp, field);

    unsigned char* ptr = outBuffer;

    // store the x, y, z, and rate data for reconstruction
    ZFPData3d outData{x, y, z, rate};
    memcpy(ptr, &outData, sizeof(ZFPData3d));
    ptr += sizeof(ZFPData3d);

    // Create a bitstream from the buffer to store compressed data.
    bitstream* stream = stream_open(ptr, bufsize);
    // Associate the bitstream with the ZFP stream, so compressed data will go
    // into our buffer.
    zfp_stream_set_bit_stream(zfp, stream);
    // Compress the data. The results will be stored in the buffer we've
    // created.
    int outsize = zfp_compress(zfp, field);

    // std::cout << outsize + sizeof(outData) << " vs " << size << std::endl;
    size        = outsize + sizeof(outData);

    // Close the bitstream. All compressed data should now reside in our buffer.
    stream_close(stream);

    // Close the ZFP stream since we're done with compression.
    if (closeStream) zfp_stream_close(zfp);
    // Release the memory allocated for the field since we're done with it.
    if (freeField) zfp_field_free(field);
}

unsigned char* compressMatrix(double* originalData, int x, int y, int z,
                              double rate, int& size) {
    // start by setting up the stream and field

    zfp_field* field = zfp_field_3d(originalData, zfp_type_double, x, y, z);
    zfp_stream* zfp  = zfp_stream_open(NULL);

    zfp_stream_set_rate(zfp, rate, zfp_type_double, 3, 0);

    int bufsize           = zfp_stream_maximum_size(zfp, field);
    size                  = bufsize + sizeof(ZFPData3d);

    // then allocate the array
    unsigned char* buffer = new unsigned char[size];

    // send it for compression
    compressMatrixBuffer(originalData, x, y, z, rate, buffer, size, field, zfp);

    // clear the field and zfp, this function is "owning"
    zfp_stream_close(zfp);
    zfp_field_free(field);

    return buffer;
}

void decompressMatrixBuffer(unsigned char* buffer, int bufferSize,
                            double* outBuff) {
    ZFPData3d data{};
    memcpy(&data, buffer, sizeof(ZFPData3d));
    buffer += sizeof(ZFPData3d);

    zfp_stream* zfp = zfp_stream_open(NULL);

    // Set the decompression rate instead of accuracy
    zfp_stream_set_rate(zfp, data.rate, zfp_type_double, 3, 0);
    bitstream* stream =
        stream_open(buffer, bufferSize - 3 * sizeof(int) - sizeof(double));
    zfp_stream_set_bit_stream(zfp, stream);

    zfp_field* dec_field =
        zfp_field_3d(outBuff, zfp_type_double, data.x, data.y, data.z);

    // do the decompression
    zfp_decompress(zfp, dec_field);

    // free things and then close the stream
    zfp_field_free(dec_field);
    stream_close(stream);
    zfp_stream_close(zfp);
}

double* decompressMatrix(unsigned char* buffer, int bufferSize) {
    // Deserialize metadata

    ZFPData3d data{};
    memcpy(&data, buffer, sizeof(ZFPData3d));

    double* decompressedData = new double[data.x * data.y * data.z];

    decompressMatrixBuffer(buffer, bufferSize, decompressedData);

    return decompressedData;
}

unsigned char* compressMatrix1D(double* originalData, int n, double rate,
                                int& size) {
    // Initialize a 3D array with original data, using ZFP's special 'field'
    // structure. The field has the type of double and the dimensions are given
    // by x, y, and z.
    zfp_field* field = zfp_field_1d(originalData, zfp_type_double, n);
    // Open a new ZFP stream. A ZFP stream is responsible for compressing and
    // decompressing data.
    zfp_stream* zfp  = zfp_stream_open(NULL);
    // Set the compression rate for the ZFP stream. The type of the data is
    // double, the dimensionality is 3, and '0' indicates we're not using a
    // user-specified precision.
    zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);

    // Determine the maximum buffer size necessary for this ZFP stream given the
    // input field.
    int bufsize = zfp_stream_maximum_size(zfp, field);
    size = bufsize + sizeof(int) + sizeof(double);  // metadata x,y,z,rate

    // Create a buffer with enough capacity to store the compressed data.
    unsigned char* buffer = new unsigned char[size];
    unsigned char* ptr    = buffer;

    // copy over the parameters
    memcpy(ptr, &n, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &rate, sizeof(double));
    ptr += sizeof(double);

    // Create a bitstream from the buffer to store compressed data.
    bitstream* stream = stream_open(ptr, bufsize);
    // Associate the bitstream with the ZFP stream, so compressed data will go
    // into our buffer.
    zfp_stream_set_bit_stream(zfp, stream);
    // Compress the data. The results will be stored in the buffer we've
    // created.
    zfp_compress(zfp, field);

    // Close the bitstream. All compressed data should now reside in our buffer.
    stream_close(stream);
    // Close the ZFP stream since we're done with compression.
    zfp_stream_close(zfp);
    // Release the memory allocated for the field since we're done with it.
    zfp_field_free(field);

    return buffer;
}

unsigned char* compressMatrix1D_fixedPrecision(double* originalData, int n,
                                               double precision, int& size) {
    // Initialize a 3D array with original data, using ZFP's special 'field'
    // structure. The field has the type of double and the dimensions are given
    // by x, y, and z.
    zfp_field* field = zfp_field_1d(originalData, zfp_type_double, n);
    // Open a new ZFP stream. A ZFP stream is responsible for compressing and
    // decompressing data.
    zfp_stream* zfp  = zfp_stream_open(NULL);
    // Set the compression rate for the ZFP stream. The type of the data is
    // double, the dimensionality is 3, and '0' indicates we're not using a
    // user-specified precision.
    zfp_stream_set_precision(zfp, precision);

    // Determine the maximum buffer size necessary for this ZFP stream given the
    // input field.
    int bufsize = zfp_stream_maximum_size(zfp, field);
    size = bufsize + sizeof(int) + sizeof(double);  // metadata x,y,z,rate

    // Create a buffer with enough capacity to store the compressed data.
    unsigned char* buffer = new unsigned char[size];
    unsigned char* ptr    = buffer;

    // copy over the parameters
    memcpy(ptr, &n, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &precision, sizeof(double));
    ptr += sizeof(double);

    // Create a bitstream from the buffer to store compressed data.
    bitstream* stream = stream_open(ptr, bufsize);
    // Associate the bitstream with the ZFP stream, so compressed data will go
    // into our buffer.
    zfp_stream_set_bit_stream(zfp, stream);
    // Compress the data. The results will be stored in the buffer we've
    // created.
    zfp_compress(zfp, field);

    // Close the bitstream. All compressed data should now reside in our buffer.
    stream_close(stream);
    // Close the ZFP stream since we're done with compression.
    zfp_stream_close(zfp);
    // Release the memory allocated for the field since we're done with it.
    zfp_field_free(field);

    return buffer;
}

double* decompressMatrix1D(unsigned char* buffer, int bufferSize) {
    // Deserialize metadata
    int n;
    double rate;

    memcpy(&n, buffer, sizeof(int));
    buffer += sizeof(int);
    memcpy(&rate, buffer, sizeof(double));
    buffer += sizeof(double);

    zfp_stream* zfp = zfp_stream_open(NULL);

    // Set the decompression rate instead of accuracy
    zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);
    bitstream* stream =
        stream_open(buffer, bufferSize - 1 * sizeof(int) - sizeof(double));

    zfp_stream_set_bit_stream(zfp, stream);
    double* decompressedData = new double[n];
    zfp_field* dec_field = zfp_field_1d(decompressedData, zfp_type_double, n);
    zfp_decompress(zfp, dec_field);
    zfp_field_free(dec_field);
    stream_close(stream);
    zfp_stream_close(zfp);
    return decompressedData;
}

void decompressMatrix1D(unsigned char* buffer, int bufferSize,
                        double* outBuff) {
    // std::cout << "in decompress" << std::endl;
    // std::cout << "buffersize: " << bufferSize << std::endl;
    // Deserialize metadata
    int n;
    double rate;

    // int rank = 0;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::memcpy(&n, buffer, sizeof(int));
    buffer += sizeof(int);
    std::memcpy(&rate, buffer, sizeof(double));
    buffer += sizeof(double);

    // std::cout << rank << ": " << "Decompressing " << bufferSize << " to: " <<
    // n * sizeof(double) << " with rate parameter of " << rate << std::endl;

    zfp_stream* zfp = zfp_stream_open(NULL);

    // Set the decompression rate instead of accuracy
    zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);
    bitstream* stream =
        stream_open(buffer, bufferSize - sizeof(int) - sizeof(double));
    zfp_stream_set_bit_stream(zfp, stream);

    zfp_field* dec_field = zfp_field_1d(outBuff, zfp_type_double, n);
    zfp_decompress(zfp, dec_field);

    zfp_field_free(dec_field);
    stream_close(stream);
    zfp_stream_close(zfp);
}

void decompressMatrix1D_fixedPrecision(unsigned char* buffer, int bufferSize,
                                       double* outBuff) {
    // std::cout << "in decompress" << std::endl;
    // std::cout << "buffersize: " << bufferSize << std::endl;
    // Deserialize metadata
    int n;
    double precision;

    // int rank = 0;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::memcpy(&n, buffer, sizeof(int));
    buffer += sizeof(int);
    std::memcpy(&precision, buffer, sizeof(double));
    buffer += sizeof(double);

    // std::cout << rank << ": " << "Decompressing " << bufferSize << " to: " <<
    // n * sizeof(double) << " with rate parameter of " << rate << std::endl;

    zfp_stream* zfp = zfp_stream_open(NULL);

    // Set the decompression rate instead of accuracy
    zfp_stream_set_precision(zfp, precision);
    bitstream* stream =
        stream_open(buffer, bufferSize - sizeof(int) - sizeof(double));
    zfp_stream_set_bit_stream(zfp, stream);

    zfp_field* dec_field = zfp_field_1d(outBuff, zfp_type_double, n);
    zfp_decompress(zfp, dec_field);

    zfp_field_free(dec_field);
    stream_close(stream);
    zfp_stream_close(zfp);
}

size_t ZFPCompression::do_3d_compression(double* originalMatrix,
                                         unsigned char* outputArray) {
    // create a field
    zfp_field_set_pointer(field_3d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize    = zfp_stream_maximum_size(zfp3d, field_3d);

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp3d, stream);

    size_t outsize = zfp_compress(zfp3d, field_3d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 3D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

size_t ZFPCompression::do_3d_decompression(unsigned char* compressedBuffer,
                                           double* outputArray) {
    // first extract out the buffer size
    size_t bufsize;

    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);

    zfp_stream_set_bit_stream(zfp3d, stream);

    zfp_field_set_pointer(field_3d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp3d, field_3d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 3D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

size_t ZFPCompression::do_2d_compression(double* originalMatrix,
                                         unsigned char* outputArray) {
    // create a field
    zfp_field_set_pointer(field_2d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize    = zfp_stream_maximum_size(zfp2d, field_2d);

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp2d, stream);

    size_t outsize = zfp_compress(zfp2d, field_2d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 2D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

size_t ZFPCompression::do_2d_decompression(unsigned char* compressedBuffer,
                                           double* outputArray) {
    // first extract out the buffer size
    size_t bufsize;

    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);

    zfp_stream_set_bit_stream(zfp2d, stream);

    zfp_field_set_pointer(field_2d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp2d, field_2d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 2D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

size_t ZFPCompression::do_1d_compression(double* originalMatrix,
                                         unsigned char* outputArray) {
    // create a field
    zfp_field_set_pointer(field_1d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize    = zfp_stream_maximum_size(zfp1d, field_1d);

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp1d, stream);

    size_t outsize = zfp_compress(zfp1d, field_1d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 1D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

size_t ZFPCompression::do_1d_decompression(unsigned char* compressedBuffer,
                                           double* outputArray) {
    // first extract out the buffer size
    size_t bufsize;

    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);

    zfp_stream_set_bit_stream(zfp1d, stream);

    zfp_field_set_pointer(field_1d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp1d, field_1d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 1D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

}  // namespace ZFPAlgorithms

namespace BLOSCAlgorithms {

BloscCompression bloscblockwise(6, "lz4", 4, 1);

size_t BloscCompression::do_3d_compression(double* originalMatrix,
                                           unsigned char* outputArray) {
    // make sure the output array includes our header
    // std::cout << "attempting to compress " << blosc_original_bytes_3d
    //           << std::endl;
    int compressedSize = blosc_compress(clevel, doShuffle, sizeof(double),
                                        blosc_original_bytes_3d, originalMatrix,
                                        outputArray + sizeof(size_t),
                                        blosc_original_bytes_overhead_3d);

    // TODO: if compressed size is 0, we have to disregard the buffer
    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 3d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}
size_t BloscCompression::do_3d_decompression(unsigned char* compressedBuffer,
                                             double* outputArray) {
    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData =
        blosc_decompress(compressedBuffer + sizeof(size_t), outputArray,
                         blosc_original_bytes_3d);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 3d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

size_t BloscCompression::do_2d_compression(double* originalMatrix,
                                           unsigned char* outputArray) {
    // TODO: need some kind of better metric or way to know if we can compress
    // or not. Current idea is if it fails, we still do a copy. We attempt to
    // copy it back out into 32 bits and see if it's garbage? idk
    if (eleOrder <= 6) {
        std::memcpy(outputArray, originalMatrix, blosc_original_bytes_2d);
        return blosc_original_bytes_2d;
    }

    // make sure the output array includes our header
    int compressedSize = blosc_compress(clevel, doShuffle, sizeof(double),
                                        blosc_original_bytes_2d, originalMatrix,
                                        outputArray + sizeof(size_t),
                                        blosc_original_bytes_overhead_2d);

    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 2d!" << std::endl;
        exit(EXIT_FAILURE);
    } else if (compressedSize == blosc_original_bytes_overhead_2d) {
        // std::cerr << "ERROR: found a block that can't be compressed in 2D!"
        //           << std::endl;
        // exit(EXIT_FAILURE);
        // need some method of handling it a bit better
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}
size_t BloscCompression::do_2d_decompression(unsigned char* compressedBuffer,
                                             double* outputArray) {
    // TODO: see 2d_compression above, this needs to be handled better
    if (eleOrder <= 6) {
        std::memcpy(outputArray, compressedBuffer, blosc_original_bytes_2d);
        return blosc_original_bytes_2d;
    }
    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData =
        blosc_decompress(compressedBuffer + sizeof(size_t), outputArray,
                         blosc_original_bytes_2d);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 2d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

size_t BloscCompression::do_1d_compression(double* originalMatrix,
                                           unsigned char* outputArray) {
    // TODO: see 1d_compression above, this needs to be handled better
    if (eleOrder <= 6) {
        std::memcpy(outputArray, originalMatrix, blosc_original_bytes_1d);
        return blosc_original_bytes_1d;
    }

    // make sure the output array includes our header
    int compressedSize = blosc_compress(
        clevel, doShuffle, sizeof(double), blosc_original_bytes_1d,
        originalMatrix, outputArray + sizeof(size_t), blosc_original_bytes_1d);

    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 1d!" << std::endl;
        exit(EXIT_FAILURE);
    } else if (compressedSize == 0) {
        // it failed to compress if we're at 0, which means garbage, so we want
        // to copy in the data
        std::cout << "FAILED in 1d Case" << std::endl;
    } else if (compressedSize == blosc_original_bytes_overhead_1d) {
        // we weren't able to get any compression!
    } else {
        // success
        std::cout << "SUCCCESS! Got a compressed 1d! Hooray!" << std::endl;
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}
size_t BloscCompression::do_1d_decompression(unsigned char* compressedBuffer,
                                             double* outputArray) {
    // TODO: see 1d_compression above, this needs to be handled better
    if (eleOrder <= 6) {
        std::memcpy(outputArray, compressedBuffer, blosc_original_bytes_1d);
        return blosc_original_bytes_1d;
    }

    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData =
        blosc_decompress(compressedBuffer + sizeof(size_t), outputArray,
                         blosc_original_bytes_1d);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 1d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

unsigned char* compressData(const char* blosc_compressor, int clevel, int n,
                            double* originalData, int& byteStreamSize) {
    blosc_set_compressor(blosc_compressor);
    int originalDataBytes         = n * sizeof(double);

    // Calculate the maximum possible size for the compressed data
    // This value is suggested to not be modified and does not affect the size
    // of the final compressed form
    int maxCompressedSize         = originalDataBytes + BLOSC_MAX_OVERHEAD;
    unsigned char* compressedData = new unsigned char[maxCompressedSize];

    // Blosc compression function: int blosc_ocmpress(int clevel, int shuffle,
    // size_t typesize, size_t nbytes, const void* src, void* dest, size_t
    // destsize); Parameters: clevel: Compression level (0-9, 0 being no
    // compression, 9 being maximum compression). shuffle: Bitshuffle filter for
    // data rearrangement.
    //          - Pass 0 for no shuffling, which can be used for data without a
    //          specific pattern or when compression speed is a priority.
    //          - Pass 1 for byte shuffle, which is effective for numerical data
    //          where each element is larger than a byte, as it aligns the least
    //          significant bits of the data types. Good for data with repeating
    //          patterns at the byte level.
    //          - Pass 2 for bit shuffle, which is more aggressive than byte
    //          shuffle and aligns the bits across data types. Useful for
    //          numerical data with repeating patterns at the bit level, often
    //          leading to better compression.
    // typesize: The size of the datatype in the array (in bytes).
    // nbytes: The number of bytes to compress from the source buffer.
    // src: Pointer to the data buffer to compress.
    // dest: Pointer to the buffer where the compressed data will be stored.
    // destsize: Maximum size of the destination buffer.
    int compressedSize =
        blosc_compress(clevel, 1, sizeof(double), originalDataBytes,
                       originalData, compressedData, maxCompressedSize);
    if (compressedSize < 0) {
        throw std::runtime_error("blosc could not compress data.");
    }

    // Allocate memory for the bytestream, including space for the size of the
    // original data
    byteStreamSize            = compressedSize + sizeof(originalDataBytes);
    unsigned char* bytestream = new unsigned char[byteStreamSize];
    // Copy compressed data to bytestream
    std::memcpy(bytestream, compressedData, compressedSize);
    // Pack originalDataBytes at the end of the bytestream
    std::memcpy(bytestream + compressedSize, &originalDataBytes,
                sizeof(originalDataBytes));
    delete[] compressedData;
    return bytestream;
}

double* decompressData(unsigned char* byteStream, int byteStreamSize) {
    // Check if byteStream is valid
    if (!byteStream || byteStreamSize <= 0) {
        return nullptr;
    }
    // Unpack originalDataBytes from the end of the byteStream
    int originalDataBytes;
    std::memcpy(&originalDataBytes,
                byteStream + (byteStreamSize - sizeof(originalDataBytes)),
                sizeof(originalDataBytes));
    double* decompressedData = new double[originalDataBytes / sizeof(double)];

    int decompressedSize =
        blosc_decompress(byteStream, decompressedData, originalDataBytes);

    // Check for decompression error
    if (decompressedSize < 0) {
        // Handle decompression error (e.g., return null or throw an exception)
        throw std::runtime_error("blosc could not decompress data.");
    }

    return decompressedData;
}

void decompressData(unsigned char* byteStream, int byteStreamSize,
                    double* outBuff) {
    // Check if byteStream is valid
    // if (!byteStream || byteStreamSize <= 0) {
    //     return nullptr;
    // }
    // Unpack originalDataBytes from the end of the byteStream

    int originalDataBytes;
    std::memcpy(&originalDataBytes,
                byteStream + (byteStreamSize - sizeof(originalDataBytes)),
                sizeof(originalDataBytes));
    // double* decompressedData = new double[originalDataBytes /
    // sizeof(double)];

    int decompressedSize =
        blosc_decompress(byteStream, outBuff, originalDataBytes);

    // Check for decompression error
    if (decompressedSize < 0) {
        // Handle decompression error (e.g., return null or throw an exception)
        throw std::runtime_error("blosc could not decompress data.");
    }
}
}  // namespace BLOSCAlgorithms

namespace dendro_compress {

CompressionType COMPRESSION_OPTION = CompressionType::ZFP;

void set_compression_options(CompressionType compT,
                             const CompressionOptions& compOpt) {
    dendro_compress::COMPRESSION_OPTION = compT;

    std::cout << "Set compression option to: "
              << dendro_compress::COMPRESSION_OPTION << std::endl;

    // then set up the options for all types

    ZFPAlgorithms::zfpblockwise.setEleOrder(compOpt.eleOrder);
    if (compOpt.zfpMode == "accuracy") {
        ZFPAlgorithms::zfpblockwise.setAccuracy(compOpt.zfpAccuracyTolerance);
    } else if (compOpt.zfpMode == "rate") {
        ZFPAlgorithms::zfpblockwise.setRate(compOpt.zfpRate);
    }

    // set up for BLOSC
    BLOSCAlgorithms::bloscblockwise.setEleOrder(compOpt.eleOrder);
    BLOSCAlgorithms::bloscblockwise.setCompressor(compOpt.bloscCompressor);

    // set up for Chebyshev
    ChebyshevAlgorithms::cheby.set_compression_type(compOpt.eleOrder,
                                                    compOpt.chebyNReduced);
}

std::size_t single_block_compress_3d(double* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_3d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_3d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_3d_compression(buffer,
                                                                     bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_decompress_3d(unsigned char* buffer,
                                       double* bufferOut) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_3d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_3d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // chebyshev decompression
            return BLOSCAlgorithms::bloscblockwise.do_3d_decompression(
                buffer, bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_compress_2d(double* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_2d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_2d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_2d_compression(buffer,
                                                                     bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_decompress_2d(unsigned char* buffer,
                                       double* bufferOut) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_2d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_2d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_2d_decompression(
                buffer, bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_compress_1d(double* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_1d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_1d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_1d_compression(buffer,
                                                                     bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_decompress_1d(unsigned char* buffer,
                                       double* bufferOut) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_1d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_1d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // chebyshev decompression
            return BLOSCAlgorithms::bloscblockwise.do_1d_decompression(
                buffer, bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t blockwise_compression(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<unsigned char>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder) {
    unsigned char config;

    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1;
    const size_t total_points_1d = points_per_dim;
    const size_t total_points_2d = total_points_1d * points_per_dim;
    const size_t total_points_3d = total_points_2d * points_per_dim;

    // TODO: set the compression type elsewhere
    // ChebyshevAlgorithms::cheby.set_compression_type(eleorder, 2);
    // ChebyshevAlgorithms::cheby.print();

    std::size_t comp_offset      = 0;
    std::size_t orig_offset      = 0;

    for (size_t ib = 0; ib < numBlocks; ib++) {
        // decode the value
        config = blockConfiguration[blockConfigOffset + ib];

        xdim   = (((config >> 6) & 7u) == 1);
        ydim   = (((config >> 3) & 7u) == 1);
        zdim   = ((config & 7u) == 1);

        // get the "dimensionality" of the block
        ndim   = xdim + ydim + zdim;

        // now based on the ndim, we will set up our compression methods
        switch (ndim) {
            case 0:
                // no compression on a single point
                std::memcpy(compressBuffer + comp_offset, &buffer[orig_offset],
                            sizeof(double));
                comp_offset += sizeof(double);
                orig_offset += total_points_0d;
                break;
            case 1:
                comp_offset += single_block_compress_1d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_1d;
                break;
            case 2:
                comp_offset += single_block_compress_2d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_2d;
                break;
            case 3:
                comp_offset += single_block_compress_3d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_3d;
                break;
            default:
                std::cerr << "Invalid number of dimensions found when doing "
                             "blockwise compression. Exiting!"
                          << std::endl;
                exit(0);
                break;
        }
    }

    return comp_offset;
}

std::size_t blockwise_decompression(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<unsigned char>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder) {
    unsigned char config;

    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;

    // these values are used to define the output side, since we're
    // decompressing back to our values. All of the decompression methods should
    // return how many bytes to advance the compression offset.
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1;
    const size_t total_points_1d = points_per_dim;
    const size_t total_points_2d = total_points_1d * points_per_dim;
    const size_t total_points_3d = total_points_2d * points_per_dim;

    std::size_t comp_offset      = 0;
    std::size_t orig_offset      = 0;

    for (std::size_t ib = 0; ib < numBlocks; ib++) {
        config = blockConfiguration[blockConfigOffset + ib];

        xdim   = (((config >> 6) & 7u) == 1);
        ydim   = (((config >> 3) & 7u) == 1);
        zdim   = ((config & 7u) == 1);

        // get the "dimensionality" of the block
        ndim   = xdim + ydim + zdim;

        // now based on the ndim, we will use our decompression methods
        switch (ndim) {
            case 0:
                // no compression on a single point
                std::memcpy(&buffer[orig_offset], compressBuffer + comp_offset,
                            sizeof(double));
                comp_offset += sizeof(double);
                orig_offset += total_points_0d;
                break;
            case 1:
                comp_offset += single_block_decompress_1d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_1d;
                break;
            case 2:
                comp_offset += single_block_decompress_2d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_2d;
                break;
            case 3:
                comp_offset += single_block_decompress_3d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_3d;
                break;
            default:
                std::cerr << "Invalid number of dimensions found when doing "
                             "blockwise decompression. Exiting!"
                          << std::endl;
                exit(0);
                break;
        }
    }
    return comp_offset;
}

std::ostream& operator<<(std::ostream& out, const CompressionOptions opts) {
    return out << "<Compression Options: eleorder " << opts.eleOrder
               << ", bloscCompressor " << opts.bloscCompressor
               << ", bloscCLevel " << opts.bloscClevel << ", bloscDoShuffle "
               << opts.bloscDoShuffle << ", zfpMode " << opts.zfpMode
               << ", zfpRate " << opts.zfpRate << ", zfpAccuracy "
               << opts.zfpAccuracyTolerance << ", chebyNReduced "
               << opts.chebyNReduced << ">";
}

std::ostream& operator<<(std::ostream& out, const CompressionType t) {
    return out << "<CompressionType: " << COMPRESSION_TYPE_NAMES[t] << ">";
}

}  // namespace dendro_compress

// TODO: move this back inside the functionality above
namespace ChebyshevAlgorithms {

void ChebyshevCompression::set_compression_type(const size_t& eleOrder,
                                                const size_t& nReduced) {
    // TODO: autogenerate this
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

void ChebyshevCompression::do_array_norm(double* array, const size_t count,
                                         double& minVal, double& maxVal) {
    minVal       = *std::min_element(array, array + count);
    maxVal       = *std::max_element(array, array + count);
    double range = maxVal - minVal;

    if (minVal < -1.0 || maxVal > 1.0) {
        // if we're outside -1 and or 1, then we need to do full normalization
        for (size_t i = 0; i < count; i++) {
            array[i] = 2.0 * ((array[i] - minVal) / range) - 1.0;
        }
    } else if (range > 1e-8) {
        // apply a shift if we're within the vals
        double shift = -(minVal + maxVal) / 2.0;
        for (size_t i = 0; i < count; i++) {
            array[i] += shift;
        }
    } else {
        // otherwise do nothing, we're close to zero or something
    }
}

void ChebyshevCompression::undo_array_norm(double* array, const size_t count,
                                           const double minVal,
                                           const double maxVal) {
    double range = maxVal - minVal;

    if (minVal < -1.0 || maxVal > 1.0) {
        // if we're outside -1 and or 1, then we need to do full
        // denormalization
        for (size_t i = 0; i < count; i++) {
            array[i] = ((array[i] + 1.0) / 2.0) * range + minVal;
        }
    } else if (range > 1e-8) {
        // apply a shift if we're within the vals
        double shift = -(minVal + maxVal) / 2.0;
        for (size_t i = 0; i < count; i++) {
            array[i] -= shift;
        }
    } else {
        // otherwise do nothing, we didn't do anything above
    }
}

size_t ChebyshevCompression::do_3d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    // the number will always be (eleorder - 1) ^3 because it's a 3d block,
    // turns out this number coencides with cheb_dim3_decomp

    double maxVal, minVal;
    do_array_norm(originalMatrix, cheb_dim3_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    // that *I* control
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    char TRANS = 'T';

    dgemv_(&TRANS, &cheb_dim3_decomp, &cheb_dim3_comp, &alpha, A_cheb_dim3,
           &cheb_dim3_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    // now we're "compressed", and we return the total number of bytes we
    // wrote:
    return bytes_3d;
}

size_t ChebyshevCompression::do_3d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim3_decomp, &cheb_dim3_comp, &alpha, A_cheb_dim3,
           &cheb_dim3_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim3_decomp, minVal, maxVal);

    // total number of doubles coming out
    return bytes_3d;
}

size_t ChebyshevCompression::do_2d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    double maxVal, minVal;
    do_array_norm(originalMatrix, cheb_dim2_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    char TRANS = 'T';
    // now do the matrix-vector multiplication
    dgemv_(&TRANS, &cheb_dim2_decomp, &cheb_dim2_comp, &alpha, A_cheb_dim2,
           &cheb_dim2_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    // total number of bytes that we wrote
    return bytes_2d;
}

size_t ChebyshevCompression::do_2d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim2_decomp, &cheb_dim2_comp, &alpha, A_cheb_dim2,
           &cheb_dim2_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim2_decomp, minVal, maxVal);

    // total number of compress values coming out, to advance. The doubles we
    // know are easy
    return bytes_2d;
}

size_t ChebyshevCompression::do_1d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    double maxVal, minVal;

    do_array_norm(originalMatrix, cheb_dim1_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    // do the matrix math
    char TRANSA         = 'T';
    char TRANSB         = 'N';

    double* temp_output = new double[3];

    // matrix multiplication because it works
    dgemv_(&TRANSA, &cheb_dim1_decomp, &cheb_dim1_comp, &alpha, A_cheb_dim1,
           &cheb_dim1_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    return bytes_1d;
}

size_t ChebyshevCompression::do_1d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    unsigned char* ptr = compressedBuffer;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    // if (minVal > maxVal) {
    //     std::cout << "Detected minval larger than maxval in 1d decompression!
    //     "
    //               << maxVal << " " << minVal << std::endl;
    // }

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim1_decomp, &cheb_dim1_comp, &alpha, A_cheb_dim1,
           &cheb_dim1_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim1_decomp, minVal, maxVal);

    // total number of doubles coming out
    return bytes_1d;
}

}  // namespace ChebyshevAlgorithms
