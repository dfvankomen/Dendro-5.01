#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "lapac.h"
#include "libxsmm.h"

bool approx_equal(double a, double b, double epsilon = 1e-6) {
    return std::fabs(a - b) <= epsilon * std::max(std::fabs(a), std::fabs(b));
}

int main() {
    void libxsmm_init(void);
    const int M = 19, N = 13, K = 19;
    const int batchsize = 100000;

    std::vector<double> a(batchsize * M * K, 1.0);
    std::vector<double> b(batchsize * K * N, 2.0);
    std::vector<double> c_blas(batchsize * M * N, 0.0);
    std::vector<double> c_xsmm(batchsize * M * N, 0.0);

    // LIBXSMM kernel
    libxsmm_mmfunction<double> kernel(LIBXSMM_GEMM_FLAG_NONE, M, N, K, 1.0,
                                      1.0);

    // libxsmm_dgemm_batch_function kernel_batch =
    //     libxsmm_dgemm_batch_function(LIBXSMM_GEMM_FLAG_NONE, M, N,
    //     K, 1.0, 1.0);

    char transa     = 'N';
    char transb     = 'N';
    double alpha    = 1.0;
    double beta     = 1.0;
    int lda         = M;
    int ldb         = K;
    int ldc         = M;

    // Timing BLAS
    auto start_blas = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < batchsize; ++i) {
        lapack::dgemm_cpp_safe(&transa, &transb, &M, &N, &K, &alpha,
                               &a[i * M * K], &lda, &b[i * K * N], &ldb, &beta,
                               &c_blas[i * M * N], &ldc);
    }

    auto end_blas   = std::chrono::high_resolution_clock::now();

    // Timing LIBXSMM
    auto start_xsmm = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < batchsize; ++i) {
        kernel(&a[i * M * K], &b[i * K * N], &c_xsmm[i * M * N]);
    }
    auto end_xsmm = std::chrono::high_resolution_clock::now();

#if 0
    auto start_xsmm_batched      = std::chrono::high_resolution_clock::now();
    libxsmm_blasint index_base   = 0;
    libxsmm_blasint index_stride = 1;
    libxsmm_blasint stride_a     = M * K;
    libxsmm_blasint stride_b     = K * N;
    libxsmm_blasint stride_c     = M * N;
    libxsmm_gemm_batch(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, &transa,
                       &transb, M, N, K, &alpha, a.data(), &lda, b.data(), &ldb,
                       &beta, c_xsmm.data(), &ldc, index_base, index_stride,
                       &stride_a, &stride_b, &stride_c, batchsize);
    auto end_xsmm_batched = std::chrono::high_resolution_clock::now();
#endif

    // Calculate and print execution times
    auto blas_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_blas - start_blas)
                         .count();
    auto xsmm_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_xsmm - start_xsmm)
                         .count();

    std::cout << "BLAS execution time: \t\t" << blas_time << " microseconds\n";
    std::cout << "LIBXSMM execution time: \t" << xsmm_time << " microseconds\n";
    std::cout << "LIBXSMM has speedup of: \t"
              << (double)blas_time / (double)xsmm_time << std::endl;

    // verify correctness
    bool results_match = true;
    for (int i = 0; i < batchsize * M * N; ++i) {
        if (!approx_equal(c_blas[i], c_xsmm[i])) {
            results_match = false;
            std::cout << "Mismatch at index " << i << ": BLAS = " << c_blas[i]
                      << ", LIBXSMM = " << c_xsmm[i] << std::endl;
            break;
        }
    }

    if (results_match) {
        std::cout
            << "Results match: BLAS and LIBXSMM produce identical outputs.\n";
    } else {
        std::cout << "Results do not match: BLAS and LIBXSMM produce different "
                     "outputs.\n";
    }

    void libxsmm_finalize(void);

    return 0;
}
