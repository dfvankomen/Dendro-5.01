//
// Created by milinda on 1/19/17.
//

/**
 *
 * @author Milinda Fernando
 * School of Computing University of Utah.
 * @brief Constains lapack routines such as linear system solve, eigen solve to
 * build the interpolation matrices.
 *
 *
 * */

#ifndef SFCSORTBENCH_LAPAC_H
#define SFCSORTBENCH_LAPAC_H

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// #include "lapacke.h"

extern "C" void sgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv,
                       double* b, int* ldb, int* info);
extern "C" void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv,
                       double* b, int* ldb, int* info);
extern "C" void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
                       double* w, double* work, int* lwork, int* info);

// double banded matrix vector multiplication.
extern "C" void dgbmv_(char* trans, int* m, int* n, int* kl, int* ku,
                       double* alpha, double* A, int* lda, double* X, int* incx,
                       double* beta, double* y, int* incy);

// LU decomoposition of a general matrix
extern "C" void dgetrf_(int* M, int* N, double* A, int* lda, int* IPIV,
                        int* INFO);

// generate inverse of a matrix given its LU decomposition
extern "C" void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK,
                        int* lwork, int* INFO);

// generic mat-mat multiplications
extern "C" void dgemm_(char* transa, char* transb, int* m, int* n, int* k,
                       double* alpha, double* A, int* lda, double* B, int* ldb,
                       double* beta, double* C, int* ldc);

// generic matrix vector multiplication.
extern "C" void dgemv_(char* trans, int* m, int* n, double* alpha, double* A,
                       int* lda, double* x, int* incx, double* beta, double* y,
                       int* incy);

extern "C" void dgbsvx_(char* fact, char* trans, int* n, int* kl, int* ku,
                        int* nrhs, double* ab, int* ldab, double* afb,
                        int* ldafb, int* ipiv, char* equed, double* r,
                        double* c, double* b, int* ldb, double* x, int* ldx,
                        double* rcond, double* ferr, double* berr, double* work,
                        int* iwork, int* info);

namespace lapack {

/**
 *  @brief: Wrapper for LAPACK DGESV solver for AX=B. Parameters are given
 * below.
 *  @param[in] n : number of rows or columns of linear system
 *  @param[in] nrhs: number of right hand sides.
 *  @param[in] A: matrix A (row major)
 *  @param[in] lda: leading dimention of the array A
 *  @param[in] B: matrix B (row major)
 *  @param[out] X:  matrix X (solution)
 *  @param[in]  ldb:  leading dimention of B
 *  @param[out] info:  returns the status of the solve.
 */
inline void lapack_DGESV(int n, int nrhs, const double* A, int lda, double* B,
                         double* X, int ldb, int info) {
    int* ipiv = new int[n];
    // memcpy(X,B,sizeof(double)*n*nrhs);

    for (unsigned int i = 0; i < nrhs; i++)
        for (unsigned int j = 0; j < n; j++) X[i * n + j] = B[j * nrhs + i];

    double* L = new double[n * n];

    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < n; j++) L[i * n + j] = A[j * n + i];

    dgesv_(&n, &nrhs, L, &lda, ipiv, X, &lda, &info);

    if (info > 0) {
        printf("The diagonal element of the triangular factor of A,\n");
        printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
        printf("the solution could not be computed.\n");
    }

    memcpy(L, X, sizeof(double) * n * nrhs);

    for (unsigned int i = 0; i < nrhs; i++)
        for (unsigned int j = 0; j < n; j++) X[i * nrhs + j] = L[j * n + i];

    /*lapack_int * ipiv = (lapack_int *)malloc(n*sizeof(lapack_int)) ;
    memcpy(X,B,sizeof(double)*n*nrhs);

    double * L=new double[n*n];
    memcpy(L,A,sizeof(double)*n*n);

    info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, nrhs, L, lda, ipiv, X,ldb);*/

    if (info < 0) {
        printf(" lapack linear solve failed. \n");
    }
    delete[] L;
    delete[] ipiv;

    return;
}

// both are transposed
inline void lapack_DGESV_T(int n, int nrhs, double* A, int lda, double* B,
                           double* X, int ldb, int info) {
    int* ipiv      = new int[n];

    double* A_copy = new double[n * n];

    // copy A into a matrix to retain original values
    std::memcpy(A_copy, A, sizeof(double) * n * n);
    // copy B into X
    std::memcpy(X, B, sizeof(double) * n * n);

    // Call LAPACK's dgesv function directly on the input arrays
    dgesv_(&n, &nrhs, A_copy, &lda, ipiv, X, &ldb, &info);

    if (info > 0) {
        printf("The diagonal element of the triangular factor of A,\n");
        printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
        printf("the solution could not be computed.\n");
    } else if (info < 0) {
        printf("LAPACK linear solve failed. Error at argument %d.\n", -info);
    }

    delete[] ipiv;
    delete[] A_copy;
}

/**
 *  @brief: Wrapper for LAPACK DGESV compute eigen values of a square matrix of
 * A. Parameters are given below.
 *  @param[in] n : number of rows or columns of linear system
 *  @param[in] A: matrix A (row major)
 *  @param[in] lda: leading dimention of the array A
 *  @param[out] wr: real part of eigen values
 *  @param[out] vs eigen vectors (row major)
 *  @param[out] info:  returns the status of the solve.
 */

inline void lapack_DSYEV(int n, const double* A, int lda, double* wr,
                         double* vs, int info) {
    double* L = new double[n * n];

    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < n; j++) L[i * n + j] = A[j * n + i];

    double wkopt;
    double* work;
    int lwork = -1;
    dsyev_((char*)"Vectors", (char*)"Upper", (int*)&n, L, (int*)&lda, wr,
           &wkopt, &lwork, (int*)&info);
    lwork = (int)wkopt;
    work  = new double[lwork];
    /* Solve eigenproblem */
    dsyev_((char*)"Vectors", (char*)"Upper", (int*)&n, L, (int*)&lda, wr, work,
           &lwork, (int*)&info);

    for (unsigned int i = 0; i < n; i++)
        for (unsigned int j = 0; j < n; j++) vs[i * n + j] = L[j * n + i];

    /* std::cout<<"lapack: "<<std::endl;

     for(unsigned int i=0;i<n;i++)
         std::cout<<" i: "<<i<<" eig : "<<wr[i]<<std::endl;

     for(unsigned int i=0;i<n;i++)
     {
         for(unsigned int j=0;j<n;j++)
         {
             std::cout<<vs[i*(n)+j]<<" ";
         }

         std::cout<<std::endl;
     }

     memcpy(vs,A,sizeof(double)*n*n);
     info=LAPACKE_dsyev(LAPACK_ROW_MAJOR,'V','U',n,vs,lda,wr);
     std::cout<<"lapacke: "<<std::endl;
     for(unsigned int i=0;i<n;i++)
         std::cout<<" i: "<<i<<" eig : "<<wr[i]<<std::endl;

     for(unsigned int i=0;i<n;i++)
     {
         for(unsigned int j=0;j<n;j++)
         {
             std::cout<<vs[i*(n)+j]<<" ";
         }

         std::cout<<std::endl;
     }*/

    delete[] work;
    delete[] L;
    if (info != 0) std::cout << "lapack eigen solve failed. " << std::endl;
    return;
}

// vector implementation of solving a linear system
template <typename T>
inline std::vector<T> solveLinearSystem(const std::vector<T>& A,
                                        std::vector<T>& B, int n, int nrhs,
                                        bool A_COL_ORDER = true,
                                        bool B_COL_ORDER = true) {
    // copy A
    std::vector<T> L = A;
    // copy B
    std::vector<T> X = B;

    std::vector<int> ipiv(n);

    if (!A_COL_ORDER) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                L[i * n + j] = A[j * n + i];
            }
        }
    }

    if (!B_COL_ORDER) {
        for (int i = 0; i < nrhs; i++) {
            for (int j = 0; j < n; j++) {
                X[i * n + j] = B[j * nrhs + i];
            }
        }
    }

    // then call sgesv or dgesv
    int info;

    if constexpr (std::is_same_v<T, float>) {
        sgesv_(&n, &nrhs, L.data(), &n, ipiv.data(), X.data(), &n, &info);
    } else if constexpr (std::is_same_v<T, double>) {
        dgesv_(&n, &nrhs, L.data(), &n, ipiv.data(), X.data(), &n, &info);
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Type must be either float or double!");
    }

    if (info > 0) {
        std::cerr << "The diagonal element of the triangular factor of A,\n";
        std::cerr << "U(" << info << "," << info
                  << ") is zero; A is singular;\n";
        std::cerr << "the solution could not be computed.\n";
        return X;
    } else if (info < 0) {
        std::cerr << "LAPACK DGESV failed with error code: " << info << "\n";
        return X;
    }

    return X;
}

/**
 * @brief computes the inverse of a matrix in place
 * @param[in] A : input matrix. (should be invertible)
 * @param[in] N : size of the matrix.
 *
 */
inline void inverse(double* A, int N) {
    int* IPIV    = new int[N + 1];
    int LWORK    = N * N;
    double* WORK = new double[LWORK];
    int INFO;

    dgetrf_(&N, &N, A, &N, IPIV, &INFO);
    dgetri_(&N, A, &N, IPIV, WORK, &LWORK, &INFO);

    delete[] IPIV;
    delete[] WORK;
}

inline void dgemm_cpp_safe(const char* TRANSA, const char* TRANSB, const int* m,
                           const int* n, const int* k, const double* alpha,
                           const double* a, const int* lda, const double* b,
                           const int* ldb, const double* beta, const double* c,
                           const int* ldc) {
    dgemm_(const_cast<char*>(TRANSA), const_cast<char*>(TRANSB),
           const_cast<int*>(m), const_cast<int*>(n), const_cast<int*>(k),
           const_cast<double*>(alpha), const_cast<double*>(a),
           const_cast<int*>(lda), const_cast<double*>(b), const_cast<int*>(ldb),
           const_cast<double*>(beta), const_cast<double*>(c),
           const_cast<int*>(ldc));
}

/**
 * @brief A type-safe C++ wrapper for the LAPACK dgbsvx_ function.
 *
 * This function solves the system of linear equations A*X = B for X, where A is
 * an N-by-N band matrix and X and B are N-by-NRHS matrices.
 *
 * @param fact [in] Factored form of A supplied on entry ('F') or ('N' or 'E').
 * @param trans [in] Form of the system of equations:
 *                   'N' for A*X=B, 'T' for A^T*X=B, or 'C' for A^H*X=B.
 * @param n [in] Number of linear equations.
 * @param kl [in] Number of subdiagonals within the band of A.
 * @param ku [in] Number of superdiagonals within the band of A.
 * @param nrhs [in] Number of RHS, the number of columns of B and X.
 * @param ab [in] Band matrix A, stored in band storage mode.
 * @param ldab [in] Leading dimension of the array AB.
 * @param afb [in,out] If FACT='F', AFB contains the factored form of A.
 *                     If FACT='N' or 'E', calculate factored form
 * @param ldafb [in] Leading dimension of the array AFB.
 * @param ipiv [in,out] Pivot indices that define the permutation matrix P.
 * @param equed [in,out] The form of equilibration that was done.
 * @param r [in,out] Row scale factors for A.
 * @param c [in,out] Column scale factors for A.
 * @param b [in] Right-hand side matrix B.
 * @param ldb [in] Leading dimension of the array B.
 * @param x [out] Solution matrix X.
 * @param ldx [in] Leading dimension of the array X.
 * @param rcond [out] Estimate of reciprocal condition number of matrix A after
 * equilibration (if done).
 * @param ferr [out] Estimated forward error bound for each solution vector
 * X(j).
 * @param berr [out] Componentwise relative backward error of each solution
 * vector X(j).
 * @param work [out] Array of dimension (3*N). Will contain info about the
 * equilibration.
 * @param iwork [out] Array of dimension (N). Will contain the pivot indices.
 * @param info [out] Error information, see function for more details
 */
inline void dgbsvx_cpp_safe(const char* fact, const char* trans, const int n,
                            const int* kl, const int* ku, const int nrhs,
                            const double* ab, const int* ldab, double* afb,
                            const int* ldafb, int* ipiv, char* equed, double* r,
                            double* c, double* b, const int ldb, double* x,
                            const int ldx, double* rcond, double* ferr,
                            double* berr, double* work, int* iwork, int* info) {
    dgbsvx_(const_cast<char*>(fact), const_cast<char*>(trans),
            const_cast<int*>(&n), const_cast<int*>(kl), const_cast<int*>(ku),
            const_cast<int*>(&nrhs), const_cast<double*>(ab),
            const_cast<int*>(ldab), afb, const_cast<int*>(ldafb), ipiv, equed,
            r, c, b, const_cast<int*>(&ldb), x, const_cast<int*>(&ldx), rcond,
            ferr, berr, work, iwork, info);

    // if not successful
    if (*info != 0) {
        std::cerr << "BandedMatrixSolve (dgbsvx) solve failed. info = " << *info
                  << std::endl;
        if (*info < 0) {
            std::cerr << "Illegal value in element " << std::abs(*info)
                      << std::endl;
        } else if (*info > 0 && *info <= n) {
            std::cerr << "U(i,i) is exactly zero for i = " << *info
                      << std::endl;
        } else {
            std::cerr
                << "U(i,i) is nonsingular, but rcond is less than machine zero."
                << std::endl;
        }
    }
}

inline void square_matrix_multiplication(double* A, double* B, double* X, int n,
                                         double alpha = 1.0,
                                         double beta  = 0.0) {
    // this assumes that everything is set up properly, i.e. square matrices and
    // both are col major order

    static const char TRANSA = 'N';
    static const char TRANSB = 'N';

    dgemm_cpp_safe(&TRANSA, &TRANSB, &n, &n, &n, &alpha, A, &n, B, &n, &beta, X,
                   &n);
}

inline double* iterative_inverse(double* A, int n, double tol = 1e-15,
                                 int max_iters = 5) {
    int LWORK    = n * n;
    double* X    = new double[LWORK];
    double* WORK = new double[LWORK];
    double* TEMP = new double[LWORK];
    double* I    = new double[LWORK]{};

    for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;

    // copy the data from A into X, which is always our "best" solution
    std::memcpy(X, A, sizeof(double) * LWORK);

    // calculate the inverse via LRU as an intitial "guess"
    inverse(X, n);

    // static information for various BLAS functions
    static const char TRANSA = 'N';
    static const char TRANSB = 'N';
    const int M              = n;
    const int N              = n;
    const int K              = n;
    const double alpha       = 1.0;
    const double beta        = 0.0;

    for (int iter = 0; iter < max_iters; ++iter) {
        // work needs to be AX
        dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, A, &M, X, &M,
                       &beta, WORK, &M);

        // then then make WORK = 2I - AX
        for (int i = 0; i < LWORK; ++i) WORK[i] = 2.0 * I[i] - WORK[i];

        // check the convergence by calculating l2 norm, we should be "zero"
        double error = 0.0;
        for (unsigned int k = 0; k < LWORK; k++) {
            double diff = WORK[k] - I[k];
            error += (diff * diff);
        }
        error = std::sqrt(error);

        // if we're less than our tolerence, we can just exit
        if (error < tol) break;

        // next X is then X + X * (2I - AX)
        dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, X, &M, WORK, &M,
                       &beta, TEMP, &M);

        // copy results from TEMP back into X
        std::memcpy(X, TEMP, sizeof(double) * LWORK);
    }

    delete[] WORK;
    delete[] TEMP;
    delete[] I;
    return X;
}

inline void iterative_inverse(double* A, double* X, int n, double tol = 1e-15,
                              int max_iters = 5) {
    int LWORK    = n * n;
    double* WORK = new double[LWORK];
    double* TEMP = new double[LWORK];
    double* I    = new double[LWORK]{};

    for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;

    // copy the data from A into X, which is always our "best" solution
    std::memcpy(X, A, sizeof(double) * LWORK);

    // calculate the inverse via LRU as an intitial "guess"
    inverse(X, n);

    // static information for various BLAS functions
    static const char TRANSA = 'N';
    static const char TRANSB = 'N';
    const int M              = n;
    const int N              = n;
    const int K              = n;
    const double alpha       = 1.0;
    const double beta        = 0.0;

    for (int iter = 0; iter < max_iters; ++iter) {
        // work needs to be AX
        dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, A, &M, X, &M,
                       &beta, WORK, &M);

        // then then make WORK = 2I - AX
        for (int i = 0; i < LWORK; ++i) WORK[i] = 2.0 * I[i] - WORK[i];

        // check the convergence by calculating l2 norm, we should be "zero"
        double error = 0.0;
        for (unsigned int k = 0; k < LWORK; k++) {
            double diff = WORK[k] - I[k];
            error += (diff * diff);
        }
        error = std::sqrt(error);

        // if we're less than our tolerence, we can just exit
        if (error < tol) break;

        // next X is then X + X * (2I - AX)
        dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, X, &M, WORK, &M,
                       &beta, TEMP, &M);

        // copy results from TEMP back into X
        std::memcpy(X, TEMP, sizeof(double) * LWORK);
    }

    delete[] WORK;
    delete[] TEMP;
    delete[] I;
}

}  // namespace lapack

#endif  // SFCSORTBENCH_LAPAC_H
