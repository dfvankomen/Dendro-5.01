#include "compression.h"
// #include "mpi.h"

namespace SVDAlgorithms {
	unsigned char* compressMatrix(double*& originalMatrix, const int x, const int y, const int z, int k, int& buffer_size) {
		// Convert to Eigen's VectorXd for easier manipulation
		Eigen::VectorXd eigenMatrix = Eigen::Map<Eigen::VectorXd>(originalMatrix, x * y * z);

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
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(dataMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		// Now we will get the first k columns of U and V, and the first k singular values
		Eigen::MatrixXd U = svd.matrixU().leftCols(k);
		Eigen::MatrixXd V = svd.matrixV().leftCols(k);
		Eigen::VectorXd S = svd.singularValues().head(k);


		// Serialize the matrices and metadata into a byte stream
		int U_rows = U.rows();
		int U_cols = U.cols();
		int V_rows = V.rows();
		int V_cols = V.cols();
		int S_length = S.size();

		int U_size = U.size() * sizeof(double);
		int V_size = V.size() * sizeof(double);
		int S_size = S.size() * sizeof(double);
		int header_size = 8 * sizeof(int); // x, y, z, U_rows, U_cols, V_rows, V_cols, S_size
		buffer_size = header_size + U_size + V_size + S_size;

		unsigned char* byteStream = new unsigned char[buffer_size];
		unsigned char* ptr = byteStream;

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
		for (int i = 0; i < U_rows; ++i)
		{
			for (int j = 0; j < U_cols; ++j)
			{
				double value = U(i, j);
				memcpy(ptr, &value, sizeof(double));
				ptr += sizeof(double);
			}
		}
		// For V
		for (int i = 0; i < V_rows; ++i)
		{
			for (int j = 0; j < V_cols; ++j)
			{
				double value = V(i, j);
				memcpy(ptr, &value, sizeof(double));
				ptr += sizeof(double);
			}
		}

		// For S
		for (int i = 0; i < S_length; ++i)
		{
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
		double* decompressedData = new double[x * y * z];
		for (int i = 0; i < x * y; i++) {
			for (int j = 0; j < z; j++) {
				decompressedData[i * z + j] = decompressedMatrix(i, j);
			}
		}

		return decompressedData;
	}


	/*
	// Example how to use comperssion algorithm

	#include "TimingExperiment.h"
	#include "SVDAlgorithms.h"

	int main() {
		int x = 3, y = 7, z = 7; // Dimensions of matrix. Modify as needed.
		int k = 3; // min(x*y, z)

		// Define original matrix
		double* originalMatrix = Utilities::createMatrixWave(x, y, z, 1, 3.1415, 0, 1.0, 1.0, 7);
		int originalMatrixBytes = x * y * z * sizeof(double);

		// Compress the matrix
		int compressedSize;
		unsigned char* compressedMatrix = SVDAlgorithms::compressMatrix(originalMatrix, x, y, z, k, compressedSize);

		// Decompress the matrix
		double* decompressedMatrix = SVDAlgorithms::decompressMatrix(compressedMatrix, compressedSize);

		// Printing comparison of original and decompressed data
		Utilities::printComparison(originalMatrix, decompressedMatrix, x, y, z);

		std::cout << "Original matrix size: " << originalMatrixBytes << " bytes" << std::endl;
		std::cout << "Compressed matrix size: " << compressedSize << " bytes" << std::endl;
		// Printing various types of error between original and decompressed data
		Utilities::printError(originalMatrix, decompressedMatrix, x, y, z);

		// Freeing the memory
		delete[] originalMatrix;
		delete[] compressedMatrix;
		delete[] decompressedMatrix;

		return 0;
	}
	*/

	std::pair<int, int> findSquareLikeDimensions(int n) {
		if (n < 2) {
			return { 1, n };
		}
		for (int i = static_cast<int>(sqrt(n)); i >= 1; --i) {
			if (n % i == 0) {
				return { i, n / i };
			}
		}
		return { 2, (n / 2) + 1 };  // if n is prime
	}

	unsigned char* compressMatrix1d(double*& originalMatrix, const int n, int cmp_str, int& buffer_size) {
		// Find 2D dimensions
		std::pair<int, int> dimensions = findSquareLikeDimensions(n);
		int rows = dimensions.first;
		int cols = dimensions.second;

		// Handle the case when n is prime
		bool fill_zero = (rows * cols > n);

		// Set k based on cmp_str
		int k = std::min(rows, cols);
		k = k - cmp_str;
		if (k < 1)
		{
			k = 1;
		}

		// Reshape 1D to 2D matrix
		Eigen::MatrixXd dataMatrix(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				int idx = i * cols + j;
				dataMatrix(i, j) = (idx < n) ? originalMatrix[idx] : 0;
			}
		}

		// Perform SVD
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(dataMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		// Now we will get the first k columns of U and V, and the first k singular values
		Eigen::MatrixXd U = svd.matrixU().leftCols(k);
		Eigen::MatrixXd V = svd.matrixV().leftCols(k);
		Eigen::VectorXd S = svd.singularValues().head(k);


		// Serialize the matrices and metadata into a byte stream
		int U_rows = U.rows();
		int U_cols = U.cols();
		int V_rows = V.rows();
		int V_cols = V.cols();
		int S_length = S.size();

		int U_size = U.size() * sizeof(double);
		int V_size = V.size() * sizeof(double);
		int S_size = S.size() * sizeof(double);
		int header_size = 7 * sizeof(int) + sizeof(bool); // rows, cols, U_rows, U_cols, V_rows, V_cols, S_size, fill_zero
		buffer_size = header_size + U_size + V_size + S_size;

		unsigned char* byteStream = new unsigned char[buffer_size];
		unsigned char* ptr = byteStream;

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
		for (int i = 0; i < U_rows; ++i)
		{
			for (int j = 0; j < U_cols; ++j)
			{
				double value = U(i, j);
				memcpy(ptr, &value, sizeof(double));
				ptr += sizeof(double);
			}
		}
		// For V
		for (int i = 0; i < V_rows; ++i)
		{
			for (int j = 0; j < V_cols; ++j)
			{
				double value = V(i, j);
				memcpy(ptr, &value, sizeof(double));
				ptr += sizeof(double);
			}
		}

		// For S
		for (int i = 0; i < S_length; ++i)
		{
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

		int n = rows * cols - static_cast<int>(fill_zero);
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
}



namespace ChebyshevAlgorithms {
	// Type alias for cache key, where first is the degree n and second is the value x
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
			return std::hash<int>()(key.x) ^ std::hash<int>()(key.y) ^ std::hash<int>()(key.z);
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

	Eigen::MatrixXd kroneckerProduct(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
		Eigen::MatrixXd C(A.rows() * B.rows(), A.cols() * B.cols());
		for (int i = 0; i < A.rows(); ++i) {
			for (int j = 0; j < A.cols(); ++j) {
				C.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
			}
		}
		return C;
	}

	double chebyshevT(int n, double x) {
		CacheKey key = { n, x };

		// Check if value is in cache
		auto iter = cache.find(key);
		if (iter != cache.end()) {
			return iter->second;
		}

		// Calculate Chebyshev polynomial
		double result;
		if (n == 0) {
			result = 1.0;
		}
		else if (n == 1) {
			result = x;
		}
		else {
			double T_prev = 1.0;
			double T_curr = x;
			for (int i = 2; i <= n; ++i) {
				double T_next = 2 * x * T_curr - T_prev;
				T_prev = T_curr;
				T_curr = T_next;
			}
			result = T_curr;
		}

		// Store result in cache and return
		cache[key] = result;
		return result;
	}

	Eigen::MatrixXd getFullKronProductTruncated(int x, int y, int z, int N, int Q, int S) {

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

		// Define  key based on whatever criteria makes matrices A, B, C identical or close enough
		SVDKey key{ x, y, z };

		// Check if the SVDs for this key are already computed and stored in the cache
		auto it = svdCache.find(key);
		if (it != svdCache.end()) {
			// Cache hit: SVD results are already computed
			SVDs cachedSVDs = it->second;
			svdA = cachedSVDs.svdA;
			svdB = cachedSVDs.svdB;
			svdC = cachedSVDs.svdC;
		}
		else {
			// Cache miss: Need to perform SVD and store the results in the cache
			svdA = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeFullU);
			svdB = Eigen::JacobiSVD<Eigen::MatrixXd>(B, Eigen::ComputeFullU);
			svdC = Eigen::JacobiSVD<Eigen::MatrixXd>(C, Eigen::ComputeFullU);

			SVDs newSVDs{ svdA, svdB, svdC };
			svdCache[key] = newSVDs;
		}

		// Truncate U matrices
		Eigen::MatrixXd UA_truncated = svdA.matrixU().leftCols(N);
		Eigen::MatrixXd UB_truncated = svdB.matrixU().leftCols(Q);
		Eigen::MatrixXd UC_truncated = svdC.matrixU().leftCols(S);

		// Calculate the Kronecker product with truncated U matrices
		Eigen::MatrixXd UA_UB_kron = kroneckerProduct(UA_truncated, UB_truncated);
		Eigen::MatrixXd full_kron_product_truncated = kroneckerProduct(UA_UB_kron, UC_truncated);
		return full_kron_product_truncated;
	}


	unsigned char* compressMatrix(double*& originalMatrix, int x, int y, int z, int N, int Q, int S, int& bufferSize) {
		// Normalize the data inside originalMatrix
		double minVal = *std::min_element(originalMatrix, originalMatrix + x * y * z);
		double maxVal = *std::max_element(originalMatrix, originalMatrix + x * y * z);
		double range = maxVal - minVal;

		// Create normalizedMatrix as Eigen::VectorXd
		Eigen::VectorXd normalizedMatrix(x * y * z);
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				for (int k = 0; k < z; ++k) {
					normalizedMatrix(i * y * z + j * z + k) = ((originalMatrix[i * y * z + j * z + k] - minVal) / range) * 2.0 - 1.0;
				}
			}
		}

		Eigen::MatrixXd full_kron_product_truncated = getFullKronProductTruncated(x, y, z, N, Q, S);

		// Calculate the coefficients using the truncated Kronecker product 
		Eigen::VectorXd coefficients_truncated = full_kron_product_truncated.transpose() * normalizedMatrix;

		// x + y + z + N + Q + S + minVal + maxVal + buffer
		bufferSize = (6 * sizeof(int)) + (2 * sizeof(double)) + (N * Q * S * sizeof(double));

		unsigned char* buffer = new unsigned char[bufferSize];
		unsigned char* ptr = buffer;

		// Add x, y, and z integers to the buffer
		memcpy(ptr, &x, sizeof(int));
		ptr += sizeof(int);
		memcpy(ptr, &y, sizeof(int));
		ptr += sizeof(int);
		memcpy(ptr, &z, sizeof(int));
		ptr += sizeof(int);
		// Add N, Q, and S integers to the buffer
		memcpy(ptr, &N, sizeof(int));
		ptr += sizeof(int);
		memcpy(ptr, &Q, sizeof(int));
		ptr += sizeof(int);
		memcpy(ptr, &S, sizeof(int));
		ptr += sizeof(int);
		// Add minVal and maxVal to the buffer
		memcpy(ptr, &minVal, sizeof(double));
		ptr += sizeof(double);
		memcpy(ptr, &maxVal, sizeof(double));
		ptr += sizeof(double);

		for (size_t i = 0; i < coefficients_truncated.size(); ++i) {
			memcpy(ptr, &coefficients_truncated[i], sizeof(double));
			ptr += sizeof(double);
		}

		return buffer;
	}

	double* decompressMatrix(unsigned char*& buffer, int bufferSize) {
		unsigned char* ptr = buffer;

		// Unpack x, y, and z from buffer
		int x, y, z;
		memcpy(&x, ptr, sizeof(int));
		ptr += sizeof(int);
		memcpy(&y, ptr, sizeof(int));
		ptr += sizeof(int);
		memcpy(&z, ptr, sizeof(int));
		ptr += sizeof(int);
		// Unpack N, Q, and S from buffer
		int N, Q, S;
		memcpy(&N, ptr, sizeof(int));
		ptr += sizeof(int);
		memcpy(&Q, ptr, sizeof(int));
		ptr += sizeof(int);
		memcpy(&S, ptr, sizeof(int));
		ptr += sizeof(int);
		// Unpack minVal and maxVal from buffer
		double minVal, maxVal;
		memcpy(&minVal, ptr, sizeof(double));
		ptr += sizeof(double);
		memcpy(&maxVal, ptr, sizeof(double));
		ptr += sizeof(double);

		// Unpack the coefficients
		int numCoefficients = N * Q * S;
		Eigen::VectorXd coefficients_truncated = Eigen::VectorXd::Zero(numCoefficients);
		for (int i = 0; i < numCoefficients; ++i) {
			memcpy(&coefficients_truncated[i], ptr, sizeof(double));
			ptr += sizeof(double);
		}

		Eigen::MatrixXd full_kron_product_truncated = getFullKronProductTruncated(x, y, z, N, Q, S);

		// Reconstruct the matrix using the coefficients and the truncated Kronecker product
		Eigen::VectorXd reconstructedMatrix = full_kron_product_truncated * coefficients_truncated;

		// Convert Eigen::VectorXd back to double*
		double* reconstructedArray = new double[x * y * z];
		for (int i = 0; i < x * y * z; ++i) {
			reconstructedArray[i] = reconstructedMatrix(i);
		}

		// Unnormalize the data
		double range = maxVal - minVal;
		for (int i = 0; i < x * y * z; ++i) {
			reconstructedArray[i] = ((reconstructedArray[i] + 1.0) / 2.0) * range + minVal;
		}


		return reconstructedArray;
	}

	/*
	int main() {
		int x = 3, y = 7, z = 7;
		double* originalMatrix = Utilities::createMatrixWave(x, y, z, 0.1, 0.1, 0.1, 1.0, 1.0, 1);
		// The first degree polynomials for each dimension
		int N = 3;
		int Q = 5;
		int S = 5;

		int bufferSize;

		unsigned char* buffer = ChebyshevAlgorithms::compressMatrix(originalMatrix, x, y, z, N, Q, S, bufferSize);
		double* decompressedMatrix = ChebyshevAlgorithms::decompressMatrix(buffer, bufferSize);

		Utilities::printComparison(originalMatrix, decompressedMatrix, x, y, z);
		Utilities::printError(originalMatrix, decompressedMatrix, x, y, z);

		// Printing size of original and serialized compressed data
		std::cout << "Size of original matrix (bytes): " << (x * y * z) * sizeof(double) << "\n";
		std::cout << "Size of serialized compressed data (bytes): " << bufferSize << "\n";

		delete[] decompressedMatrix;
		delete[] originalMatrix;

		return 0;

	}
	*/
}

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

	unsigned char* compressMatrix(double* originalData, int x, int y, int z, double rate, int& size) {
		// Initialize a 3D array with original data, using ZFP's special 'field' structure. 
		// The field has the type of double and the dimensions are given by x, y, and z.
		zfp_field* field = zfp_field_3d(originalData, zfp_type_double, x, y, z);
		// Open a new ZFP stream. A ZFP stream is responsible for compressing and decompressing data.
		zfp_stream* zfp = zfp_stream_open(NULL);
		// Set the compression rate for the ZFP stream. The type of the data is double, the dimensionality is 3, and '0' indicates we're not using a user-specified precision.
		zfp_stream_set_rate(zfp, rate, zfp_type_double, 3, 0);

		// Determine the maximum buffer size necessary for this ZFP stream given the input field.
		int bufsize = zfp_stream_maximum_size(zfp, field);
		size = bufsize + 3 * sizeof(int) + sizeof(double); // metadata x,y,z,rate
		// Create a buffer with enough capacity to store the compressed data.
		unsigned char* buffer = new unsigned char[size];
		unsigned char* ptr = buffer;
		memcpy(ptr, &x, sizeof(int));
		ptr += sizeof(int);
		memcpy(ptr, &y, sizeof(int));
		ptr += sizeof(int);
		memcpy(ptr, &z, sizeof(int));
		ptr += sizeof(int);
		memcpy(ptr, &rate, sizeof(double));
		ptr += sizeof(double);
		// Create a bitstream from the buffer to store compressed data.
		bitstream* stream = stream_open(ptr, bufsize);
		// Associate the bitstream with the ZFP stream, so compressed data will go into our buffer.
		zfp_stream_set_bit_stream(zfp, stream);
		// Compress the data. The results will be stored in the buffer we've created.
		zfp_compress(zfp, field);

		// Close the bitstream. All compressed data should now reside in our buffer.
		stream_close(stream);
		// Close the ZFP stream since we're done with compression.
		zfp_stream_close(zfp);
		// Release the memory allocated for the field since we're done with it.
		zfp_field_free(field);

		return buffer;
	}

	unsigned char* compressMatrix1D(double* originalData, int n, double rate, int& size) {
		// Initialize a 3D array with original data, using ZFP's special 'field' structure. 
		// The field has the type of double and the dimensions are given by x, y, and z.
		zfp_field* field = zfp_field_1d(originalData, zfp_type_double, n);
		// Open a new ZFP stream. A ZFP stream is responsible for compressing and decompressing data.
		zfp_stream* zfp = zfp_stream_open(NULL);
		// Set the compression rate for the ZFP stream. The type of the data is double, the dimensionality is 3, and '0' indicates we're not using a user-specified precision.
		zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);

		// Determine the maximum buffer size necessary for this ZFP stream given the input field.
		int bufsize = zfp_stream_maximum_size(zfp, field);
		size = bufsize + 1 * sizeof(int) + sizeof(double); // metadata x,y,z,rate
		// Create a buffer with enough capacity to store the compressed data.
		unsigned char* buffer = new unsigned char[size];
		unsigned char* ptr = buffer;
		memcpy(ptr, &n, sizeof(int));
		ptr += sizeof(int);
		memcpy(ptr, &rate, sizeof(double));
		ptr += sizeof(double);
		// Create a bitstream from the buffer to store compressed data.
		bitstream* stream = stream_open(ptr, bufsize);
		// Associate the bitstream with the ZFP stream, so compressed data will go into our buffer.
		zfp_stream_set_bit_stream(zfp, stream);
		// Compress the data. The results will be stored in the buffer we've created.
		zfp_compress(zfp, field);

		// Close the bitstream. All compressed data should now reside in our buffer.
		stream_close(stream);
		// Close the ZFP stream since we're done with compression.
		zfp_stream_close(zfp);
		// Release the memory allocated for the field since we're done with it.
		zfp_field_free(field);

		return buffer;
	}


	double* decompressMatrix(unsigned char* buffer, int bufferSize) {
		// Deserialize metadata
		int x, y, z;
		memcpy(&x, buffer, sizeof(int));
		buffer += sizeof(int);
		memcpy(&y, buffer, sizeof(int));
		buffer += sizeof(int);
		memcpy(&z, buffer, sizeof(int));
		buffer += sizeof(int);
		double rate;
		memcpy(&rate, buffer, sizeof(double));
		buffer += sizeof(double);

		zfp_stream* zfp = zfp_stream_open(NULL);

		// Set the decompression rate instead of accuracy
		zfp_stream_set_rate(zfp, rate, zfp_type_double, 3, 0);
		bitstream* stream = stream_open(buffer, bufferSize - 3 * sizeof(int) - sizeof(double));
		zfp_stream_set_bit_stream(zfp, stream);
		double* decompressedData = new double[x * y * z];
		zfp_field* dec_field = zfp_field_3d(decompressedData, zfp_type_double, x, y, z);
		zfp_decompress(zfp, dec_field);
		zfp_field_free(dec_field);
		stream_close(stream);
		zfp_stream_close(zfp);
		return decompressedData;
	}

	double* decompressMatrix1D(unsigned char* buffer, int bufferSize) {
		// Deserialize metadata
		int n;
		memcpy(&n, buffer, sizeof(int));
		buffer += sizeof(int);
		double rate;
		memcpy(&rate, buffer, sizeof(double));
		buffer += sizeof(double);

		zfp_stream* zfp = zfp_stream_open(NULL);

		// Set the decompression rate instead of accuracy
		zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);
		bitstream* stream = stream_open(buffer, bufferSize - 1 * sizeof(int) - sizeof(double));
		zfp_stream_set_bit_stream(zfp, stream);
		double* decompressedData = new double[n];
		zfp_field* dec_field = zfp_field_1d(decompressedData, zfp_type_double, n);
		zfp_decompress(zfp, dec_field);
		zfp_field_free(dec_field);
		stream_close(stream);
		zfp_stream_close(zfp);
		return decompressedData;
	}

    void decompressMatrix1D(unsigned char* buffer, int bufferSize, double* outBuff) {
		// std::cout << "in decompress" << std::endl;
		// std::cout << "buffersize: " << bufferSize << std::endl;
		// Deserialize metadata
		int n;
		memcpy(&n, buffer, sizeof(int));

		// int rank = 0;
		// MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		buffer += sizeof(int);
		double rate;
		memcpy(&rate, buffer, sizeof(double));

		// std::cout << rank << ": " << "Decompressing " << bufferSize << " to: " << n * sizeof(double) << " with rate parameter of " << rate << std::endl;
		buffer += sizeof(double);

		zfp_stream* zfp = zfp_stream_open(NULL);

		// Set the decompression rate instead of accuracy
		zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);
		bitstream* stream = stream_open(buffer, bufferSize - 1 * sizeof(int) - sizeof(double));
		zfp_stream_set_bit_stream(zfp, stream);

		zfp_field* dec_field = zfp_field_1d(outBuff, zfp_type_double, n);
		zfp_decompress(zfp, dec_field);
		zfp_field_free(dec_field);
		stream_close(stream);
		zfp_stream_close(zfp);

	}
}



namespace BLOSCCompression {

    unsigned char* compressData(const char* blosc_compressor, int clevel, int n, double* originalData, int& byteStreamSize) {
        blosc_set_compressor(blosc_compressor);
        int originalDataBytes = n * sizeof(double); 
        // Calculate the maximum possible size for the compressed data
        // This value is suggested to not be modified and does not affect the size of the final compressed form
        int maxCompressedSize = originalDataBytes + BLOSC_MAX_OVERHEAD; 
        unsigned char* compressedData = new unsigned char[maxCompressedSize]; 

        // Blosc compression function: int blosc_ocmpress(int clevel, int shuffle, size_t typesize, size_t nbytes, const void* src, void* dest, size_t destsize);
        // Parameters:
        // clevel: Compression level (0-9, 0 being no compression, 9 being maximum compression).
        // shuffle: Bitshuffle filter for data rearrangement. 
        //          - Pass 0 for no shuffling, which can be used for data without a specific pattern or when compression speed is a priority.
        //          - Pass 1 for byte shuffle, which is effective for numerical data where each element is larger than a byte, as it aligns the least significant bits of the data types. Good for data with repeating patterns at the byte level.
        //          - Pass 2 for bit shuffle, which is more aggressive than byte shuffle and aligns the bits across data types. Useful for numerical data with repeating patterns at the bit level, often leading to better compression.
        // typesize: The size of the datatype in the array (in bytes).
        // nbytes: The number of bytes to compress from the source buffer.
        // src: Pointer to the data buffer to compress.
        // dest: Pointer to the buffer where the compressed data will be stored.
        // destsize: Maximum size of the destination buffer.
        int compressedSize = blosc_compress(clevel, 1, sizeof(double), originalDataBytes, originalData, compressedData, maxCompressedSize); 
        if (compressedSize < 0) {
            throw std::runtime_error("blosc could not compress data.");
        }

        // Allocate memory for the bytestream, including space for the size of the original data
        byteStreamSize = compressedSize + sizeof(originalDataBytes);
        unsigned char* bytestream = new unsigned char[byteStreamSize];
        // Copy compressed data to bytestream
        std::memcpy(bytestream, compressedData, compressedSize);
        // Pack originalDataBytes at the end of the bytestream
        std::memcpy(bytestream + compressedSize, &originalDataBytes, sizeof(originalDataBytes));
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
        std::memcpy(&originalDataBytes, byteStream + (byteStreamSize - sizeof(originalDataBytes)), sizeof(originalDataBytes));
        double* decompressedData = new double[originalDataBytes / sizeof(double)];

        int decompressedSize = blosc_decompress(byteStream, decompressedData, originalDataBytes);

        // Check for decompression error
        if (decompressedSize < 0) {
            // Handle decompression error (e.g., return null or throw an exception)
            throw std::runtime_error("blosc could not decompress data.");
        }

        return decompressedData;
    }
}