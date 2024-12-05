#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "dendro.h"
#include "lapac.h"
#include "refel.h"

// NOTE: lapac.h stores all BLAS/LAPACK routine references and generic versions
// and wrappers

#define INDEX_3D(i, j, k) ((i) + nx * ((j) + ny * (k)))

#define INDEX_N2D(i, j, n) ((i) + (n) * (j))

/**
 * Declarations for external FORTRAN linear algebra routines
 */
#if 0
extern "C" {
// LU decomposition of a general matrix
void dgetrf_(int *n, int *m, double *P, int *lda, int *IPIV, int *INFO);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
             int *INFO);

// multiplies two matrices C = alpha*A*B + beta*C
void dgemm_(char *TA, char *TB, int *M, int *N, int *K, double *ALPHA,
            double *A, int *LDA, double *B, int *LDB, double *BETA, double *C,
            int *LDC);

// generic matrix vector multiplication.
void dgemv_(char *trans, int *m, int *n, double *alpha, double *A, int *lda,
            double *x, int *incx, double *beta, double *y, int *incy);

// Multiply banded matrix (A) by a vector (y), y = A * x
void dgbmv_(char *TRANS, int *M, int *N, int *KL, int *KU, double *alpha,
            double *A, int *LDA, double *x, int *INCX, double *beta, double *y,
            int *INCY);

// banded linear system solver
void dgbsvx_(char *fact, char *trans, int *n, int *kl, int *ku, int *nrhs,
             double *ab, int *ldab, double *afb, int *ldafb, int *ipiv,
             char *equed, double *r, double *c, double *b, int *ldb, double *x,
             int *ldx, double *rcond, double *ferr, double *berr, double *work,
             int *iwork, int *info);
}
#endif

namespace dendroderivs {

/**
 * Matrix storage struct for our different boundary types.
 *
 */
struct DerivMatrixStorage {
    std::vector<double> D_original;  ///< Storage for matrix with no boundary
    std::vector<double> D_left;      ///< Storage for matrix with left boundary
    std::vector<double> D_right;     ///< Storage for matrix with right boundary
    std::vector<double> D_leftright;  ///< Storage for matrix left and right
    uint32_t dim_size = 13;

    // Destructor to self-clean
    ~DerivMatrixStorage() {}

    void print() {
        std::cout << "::::::::::::::::::::::" << std::endl;
        std::cout << "::DerivMatrixStorage::" << std::endl;
        std::cout << "Original: " << std::endl;
        printArray_2D_transpose(D_original.data(), dim_size, dim_size);
        std::cout << "Left: " << std::endl;
        printArray_2D_transpose(D_left.data(), dim_size, dim_size);
        std::cout << "Right: " << std::endl;
        printArray_2D_transpose(D_right.data(), dim_size, dim_size);
        std::cout << "LeftRight: " << std::endl;
        printArray_2D_transpose(D_leftright.data(), dim_size, dim_size);
        std::cout << "::::::::::::::::::::::" << std::endl;
    }
};

enum BoundaryType {
    NO_BOUNDARY = 0,
    LEFT_BOUNDARY,
    RIGHT_BOUNDARY,
    LEFTRIGHT_BOUNDARY
};

inline std::vector<double> *const get_deriv_mat_by_boundary(
    DerivMatrixStorage *dmat, const BoundaryType &b) {
    switch (b) {
        case BoundaryType::NO_BOUNDARY:
            return &dmat->D_original;
            break;
        case BoundaryType::LEFT_BOUNDARY:
            return &dmat->D_left;
            break;
        case BoundaryType::RIGHT_BOUNDARY:
            return &dmat->D_right;
            break;
        case BoundaryType::LEFTRIGHT_BOUNDARY:
            return &dmat->D_leftright;
            break;
        default:
            throw std::runtime_error(
                "Somehow we're trying to build the matrix, but this should "
                "never be hit!");
            break;
    }
}

inline std::vector<double> *const get_deriv_mat_by_bflag_x(
    DerivMatrixStorage *dmat, const unsigned int &bflag) {
    if (!(bflag & (1u << OCT_DIR_LEFT)) && !(bflag & (1u << OCT_DIR_RIGHT))) {
        return &dmat->D_original;
    } else if ((bflag & (1u << OCT_DIR_LEFT)) &&
               !(bflag & (1u << OCT_DIR_RIGHT))) {
        return &dmat->D_left;
    } else if (!(bflag & (1u << OCT_DIR_LEFT)) &&
               (bflag & (1u << OCT_DIR_RIGHT))) {
        return &dmat->D_right;
    } else {
        return &dmat->D_leftright;
    }
}

inline std::vector<double> *const get_deriv_mat_by_bflag_y(
    DerivMatrixStorage *dmat, const unsigned int &bflag) {
    if (!(bflag & (1u << OCT_DIR_DOWN)) && !(bflag & (1u << OCT_DIR_UP))) {
        return &dmat->D_original;
    } else if ((bflag & (1u << OCT_DIR_DOWN)) &&
               !(bflag & (1u << OCT_DIR_UP))) {
        return &dmat->D_left;
    } else if (!(bflag & (1u << OCT_DIR_DOWN)) &&
               (bflag & (1u << OCT_DIR_UP))) {
        return &dmat->D_right;
    } else {
        return &dmat->D_leftright;
    }
}

inline std::vector<double> *const get_deriv_mat_by_bflag_z(
    DerivMatrixStorage *dmat, const unsigned int &bflag) {
    if (!(bflag & (1u << OCT_DIR_BACK)) && !(bflag & (1u << OCT_DIR_FRONT))) {
        return &dmat->D_original;
    } else if ((bflag & (1u << OCT_DIR_BACK)) &&
               !(bflag & (1u << OCT_DIR_FRONT))) {
        return &dmat->D_left;
    } else if (!(bflag & (1u << OCT_DIR_BACK)) &&
               (bflag & (1u << OCT_DIR_FRONT))) {
        return &dmat->D_right;
    } else {
        return &dmat->D_leftright;
    }
}

/**
 * Here are defined all the variables needed for calling the
 *  LAPACK routine dgbsvx.
 * Please define these separately for 1st and 2nd derivatives, as
 *  they contain crucial outputs for analyzing the individual derivatives taken
 */
struct BandedMatrixSolveVars {
    // characters
    char *FACT    = nullptr;
    char *TRANS   = nullptr;
    char *EQUED   = nullptr;

    // numbers
    int *N        = nullptr;
    int *NRHS     = nullptr;
    int *LDAB     = nullptr;
    int *LDAFB    = nullptr;
    int *LDB      = nullptr;
    int *LDX      = nullptr;
    int *KL       = nullptr;
    int *KU       = nullptr;
    int *INFO     = nullptr;

    // arrays
    double *AB    = nullptr;
    double *AFB   = nullptr;
    int *IPIV     = nullptr;
    double *R     = nullptr;
    double *C     = nullptr;
    double *B     = nullptr;
    double *X     = nullptr;
    double *RCOND = nullptr;
    double *FERR  = nullptr;
    double *BERR  = nullptr;
    double *WORK  = nullptr;
    int *IWORK    = nullptr;

    BandedMatrixSolveVars(char FACT, char TRANS, int N, int NRHS, int KL,
                          int KU, double *AB);
    BandedMatrixSolveVars(const BandedMatrixSolveVars &obj);
    ~BandedMatrixSolveVars();
};

/**
 * struct containing KL and KU for each matrix
 */
struct BandedMatrixDiagonalWidths {
    int pkl;
    int pku;
    int qkl;
    int qku;
};

/**
 * @brief   Multiplies two matrices using LAPACK/BLAS dgemm, C = A B.  Assumes A
 * is square.
 *
 * NOTE: Previously, TA and TB were character arrays of size 4. sgemm_ does not
 * need more than one character (this has been tested), so TA and TB are now
 * single characters.
 *
 * NOTE: in C++, ints are passed by value on function calls. Even if sgemm_ were
 * to modify inputs depending on these, the original memory locations would not
 * be modified. For a (marginal) speed increase (when mulMM is called many
 * times), the extra declarations M thru LDC can be removed.
 *
 * @param C   Matrix of size (na, nb)
 * @param A   Square matrix of size (na, na)
 * @param B   Matrix of size (na, nb)
 * @param na  Rows of A and B, columns of A
 * @param nb  Columns of B
 */
void mulMM(double *C, double *A, double *B, int na, int nb);

/**
 * @brief     Calulates \f$D = P^{-1} Q\f$ using LAPACK with LU decomposition.
 *
 * @param D   Square matrix (n,n)
 * @param P   Square matrix (n,n)
 * @param Q   Square matrix (n,n)
 * @param n   size of matrices
 */
void calculateDerivMatrix(double *D, double *P, double *Q, const int n);

/**
 * @brief Take a matrix A (n, n) and store it in banded storage
 *  as a matrix AB (kl + ku + 1, n), according to
 *  https://netlib.org/lapack/lug/node124.html
 * @warning I ( Colin :) ) wrote my own implementation, but I later realized
 *  I think they (LAPACK/BLAS) provide an algorithm for this. It's probably
 *  faster than mine, but this is only to be run at the beginning of each DNS.
 *  For optimizing BL operators, though, we may be able to get a speed increase
 *  by improving this algorithm.
 *
 * @param AB  (kl + ku + 1): A stored in banded storage
 * @param A   The banded matrix (n, n) to be stored in banded storage
 * @param kl  number of sub-diagonals
 * @param ku  number of super-diagonals
 * @param n   rank of A
 */
void bandedMatrixStore(double *AB, double *A, const int kl, const int ku,
                       const unsigned int n);

void bandedMatrixVectorMult(double *y, double *A, double *x, int kl, int ku,
                            double alpha, int n);

/**
 * @brief Solve a system of linear equations of the form A * X = B,
 *  where A is a banded matrix (in banded storage), X is a column
 *  vector of unknowns, and B is a column vector.
 *
 * @deprecated
 *
 * NOTE: to understand this, see documentation at
 *
 https://netlib.org/lapack/explore-html/d1/da6/group__gbsvx_ga38273d98ae4d598529fc9647ca847ce2.html
 *
 * NOTE: in certain cases, AB and B will be modified on exit.
 *  Please account for this.
 * @todo the above note (I have not accounted for this!)
 *
 * @param FACT how the alg handles factorization: 'F', 'N', or 'E'
 * @param TRANS if the matrix is transposed: 'N', 'T', or 'C'
 * @param n number of linear equations
 * @param kl number of lower diagonals
 * @param ku number of upper diagonals
 * @param AB the matrix A in banded storage (kl + ku + 1, n)
 * @param AFB a matrix (2kl + ku + 1, n); input if FACT='F' (see docs),
 output otherwise
 * @param IPIV int array (n); input if FACT='F' (see docs), output otherwise
 * @param EQUED how equilibration was done; input if FACT='F' (see docs),
 output otherwise
 * @param R double array (n); input if FACT='F' (see docs), output otherwise
 * @param C double array (n); input if FACT='F' (see docs), output otherwise
 * @param B double array (n); input: RHS. output: may be overwritten (see
 docs)
 * @param X double array (n); the solution (output)
 * @param RCOND output concerning "reciprocal condition number of the matrix"
 (see docs)
 * @param FERR output (see docs); double array, (1)
 * @param BERR output (see docs); double array, (1)
 * @param WORK output (see docs); double array, (3*n)
 * @param IWORK output; int array, (n)
 */
int bandedMatrixSolve(char FACT, char TRANS, double *X, double *AB, double *B,
                      double *AFB, int *IPIV, char EQUED, double *R, double *C,
                      double RCOND, double *FERR, double *BERR, double *WORK,
                      int *IWORK, int KL, int KU, unsigned int n);

/**
 * TODO: add documentation
 *
 */
void bandedMatrixSolve(BandedMatrixSolveVars *vars);

// C++ safe versions of dgemm

#if 0
inline void dgemm_cpp_safe(const char *TRANSA, const char *TRANSB, const int *m,
                           const int *n, const int *k, const double *alpha,
                           const double *a, const int *lda, const double *b,
                           const int *ldb, const double *beta, const double *c,
                           const int *ldc) {
    dgemm_(const_cast<char *>(TRANSA), const_cast<char *>(TRANSB),
           const_cast<int *>(m), const_cast<int *>(n), const_cast<int *>(k),
           const_cast<double *>(alpha), const_cast<double *>(a),
           const_cast<int *>(lda), const_cast<double *>(b),
           const_cast<int *>(ldb), const_cast<double *>(beta),
           const_cast<double *>(c), const_cast<int *>(ldc));
}
#endif

// inline void domatcopy_cpp_safe(const char *ordering, const char *trans,
//                                const int *rows, const int *cols,
//                                const double *alpha, const double *A,
//                                const int *lda, const double *b,
//                                const int *ldb) {
//     domatcopy_(const_cast<char *>(ordering), const_cast<char *>(trans),
//                const_cast<int *>(rows), const_cast<int *>(cols),
//                const_cast<double *>(alpha), const_cast<double *>(A),
//                const_cast<int *>(lda), const_cast<double *>(b),
//                const_cast<int *>(ldb));
// }

void matmul_x_dim(const double *const R, double *const Dxu,
                  const double *const u, const double alpha,
                  const unsigned int *sz, const unsigned int bflag);

void matmul_y_dim(const double *const R, double *const Dyu,
                  const double *const u, const double alpha,
                  const unsigned int *sz, double *const workspace,
                  const unsigned int bflag);

void matmul_z_dim(const double *const R, double *const Dzu,
                  const double *const u, const double alpha,
                  const unsigned int *sz, double *const workspace,
                  const unsigned int bflag);

}  // namespace dendroderivs
