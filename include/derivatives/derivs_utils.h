#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "dendro.h"
#include "lapac.h"
#include "libxsmm.h"
#include "libxsmm_typedefs.h"
#include "refel.h"

// when enabled, the matmul functions use raw libxsmm JIT function pointers
// instead of the C++ wrapper, reducing per-call dispatch overhead. disable
// this if you suspect the tighter dispatch is causing issues.
#ifndef DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
#define DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH 1
#endif

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

// NOTE: pw is the ghost-zone padding width (derived from ele_order/2 by the
// caller). it used to live in a mutable global DENDRO_DERIVS_PW; passing it
// explicitly lets two derivs instances at different ele_orders coexist
// safely in the same process
//
// OUTPUT CONTRACT (active region). A compact operator along axis a is dense
// along a, so each active output reads the FULL extent of its input along a
// (padding included) but only the ACTIVE extent [pw, n-pw) along the other
// two axes. Consequently a derivative output only ever needs to be defined
//   - full along every axis that a downstream operator will differentiate,
//   - active along every other axis.
// The x-padding rows [0,pw) and [nx-pw,nx) of ANY derivative output are never
// read by anyone (RHS loops, boundary conditions and chained y/z operators all
// stay inside the active x range), so no path computes them. The GEMMs are
// restricted to M = active x-rows (padded up to a multiple of 4 for unmasked
// SIMD, see active_m_padded) and the corresponding output rows only.
//
// is_last_op = true additionally tells the kernel that no downstream operator
// will read the output at all, so the y/z padding is skipped too.
// Default false (safe): writes all y columns and all z slices (at active x
// rows), suitable for use as an intermediate step in mixed 2nd-order
// derivatives (e.g. v = grad_x(u) followed by w = grad_y(v) — y reads v
// across the full y range including y-padding, so x must write those cells).
// matmul_z_dim does NOT take this flag because by the project convention
// "z is always called last in mixed chains" it unconditionally skips.
void matmul_x_dim(const double *__restrict__ R, double *__restrict__ Dxu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, const unsigned int bflag,
                  const unsigned int pw, bool is_last_op = false);

void matmul_y_dim(const double *__restrict__ R, double *__restrict__ Dyu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, double *__restrict__ workspace,
                  const unsigned int bflag, const unsigned int pw,
                  bool is_last_op = false);

void matmul_z_dim(const double *__restrict__ R, double *__restrict__ Dzu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, double *__restrict__ workspace,
                  const unsigned int bflag, const unsigned int pw);

void matmul_x_dim_old(const double *const R, double *const Dxu,
                      const double *const u, const double alpha,
                      const unsigned int *sz, const unsigned int bflag,
                      const unsigned int pw);

void matmul_y_dim_old(const double *const R, double *const Dyu,
                      const double *const u, const double alpha,
                      const unsigned int *sz, double *const workspace,
                      const unsigned int bflag, const unsigned int pw);

void matmul_z_dim_old(const double *const R, double *const Dzu,
                      const double *const u, const double alpha,
                      const unsigned int *sz, double *const workspace,
                      const unsigned int bflag, const unsigned int pw);

std::vector<std::vector<double>> inline generate_identity_bdys(size_t nbdry) {
    std::vector<std::vector<double>> bdry_coeffs;

    for (size_t i = 0; i < nbdry; ++i) {
        std::vector<double> temp(i + 1, 0);
        temp[i] = 1.0;

        bdry_coeffs.push_back(temp);
    }

    return bdry_coeffs;
}

// number of active x-rows to hand the GEMM as M: the active count n - 2*pw,
// rounded up to a multiple of DENDRO_DERIVS_M_PAD when the extra rows still
// fit inside the block (they land in x-padding, which nobody reads). an
// unmasked M is measurably faster for libxsmm (n=13: 7 -> 8 is ~1.3x)
#ifndef DENDRO_DERIVS_M_PAD
#define DENDRO_DERIVS_M_PAD 4u
#endif
inline unsigned int active_m_padded(unsigned int n, unsigned int pw) {
    const unsigned int na = n - 2u * pw;
    const unsigned int mp =
        ((na + DENDRO_DERIVS_M_PAD - 1u) / DENDRO_DERIVS_M_PAD) *
        DENDRO_DERIVS_M_PAD;
    return (pw + mp <= n) ? mp : na;
}

using KernelType = libxsmm_mmfunction<double>;

// general kernel cache keyed on the full GEMM description (flags + shape +
// leading dimensions). the active-region paths need LDA/LDB/LDC that differ
// from M/K/M (they address a sub-block of the operator and of the field), so
// the three shape-only caches above can't describe them
struct KernelKey {
    int flags, M, N, K, lda, ldb, ldc;
    bool operator==(const KernelKey &o) const {
        return flags == o.flags && M == o.M && N == o.N && K == o.K &&
               lda == o.lda && ldb == o.ldb && ldc == o.ldc;
    }
};

struct KernelKeyHash {
    size_t operator()(const KernelKey &k) const {
        size_t h = std::hash<int>{}(k.flags);
        h = h * 1315423911u ^ std::hash<int>{}(k.M);
        h = h * 1315423911u ^ std::hash<int>{}(k.N);
        h = h * 1315423911u ^ std::hash<int>{}(k.K);
        h = h * 1315423911u ^ std::hash<int>{}(k.lda);
        h = h * 1315423911u ^ std::hash<int>{}(k.ldb);
        h = h * 1315423911u ^ std::hash<int>{}(k.ldc);
        return h;
    }
};

extern std::unordered_map<KernelKey, KernelType, KernelKeyHash> kernel_cache_ld;
extern std::shared_mutex kernel_cache_ld_mutex;

inline KernelType get_or_create_kernel_ld(int flags, int M, int N, int K,
                                          int lda, int ldb, int ldc) {
    KernelKey key{flags, M, N, K, lda, ldb, ldc};
    {
        std::shared_lock<std::shared_mutex> lk(kernel_cache_ld_mutex);
        auto it = kernel_cache_ld.find(key);
        if (it != kernel_cache_ld.end()) return it->second;
    }
    std::unique_lock<std::shared_mutex> lk(kernel_cache_ld_mutex);
    auto it = kernel_cache_ld.find(key);
    if (it != kernel_cache_ld.end()) return it->second;

    KernelType new_kernel(flags, M, N, K, lda, ldb, ldc, 1.0, 0.0);
    if (!new_kernel) {
        kernel_cache_ld[key] = KernelType();
        return KernelType();
    }
    kernel_cache_ld[key] = new_kernel;
    return new_kernel;
}

// ----------------------------------------------------------------------
// per-shape kernel plan + apply routines (the matrix-path hot loop)
// ----------------------------------------------------------------------
// the five kernels one block shape needs. built from the shared cache once
// per shape and then memoized per Derivs instance (one instance per thread
// under the clone model), so the timestep loop never takes the cache's
// shared_mutex or hashes a key. see MatrixCompactDerivs::plan_for
struct MatmulPlan {
    unsigned int nx = 0, ny = 0, nz = 0, pw = 0, ma = 0;
    KernelType kx_last, kx_int, ky_last, ky_int, kz;
    // fused mixed derivatives (1st-order engines): step 1 differentiates a
    // slice/slab along the first axis into a small L1 intermediate, step 2
    // applies the second axis from it straight into the active output
    KernelType kxy1, kxy2, kxz1, kxz2, kyz1, kyz2;
    bool valid = false;

    bool matches(const unsigned int *sz, unsigned int pw_) const {
        return valid && nx == sz[0] && ny == sz[1] && nz == sz[2] && pw == pw_;
    }
};

MatmulPlan build_matmul_plan(const unsigned int *sz, unsigned int pw);

// apply routines: take a kernel from the plan and an operator that already
// has the 1/h^p spacing folded in (Ds = alpha * D). return false when the
// kernel is empty (JIT failure) so the caller can fall back to the BLAS path
// with the same Ds and alpha = 1.0. the addressing here IS the active-region
// contract documented above matmul_x_dim; the matmul_*_dim wrappers and the
// Derivs instances both route through these so there is one implementation
inline bool matmul_x_apply(const KernelType &kernel,
                           const double *__restrict__ Ds,
                           double *__restrict__ Dxu,
                           const double *__restrict__ u, const unsigned int *sz,
                           const unsigned int pw, bool is_last_op) {
    if (!kernel) return false;
    const unsigned int nx       = sz[0];
    const unsigned int ny       = sz[1];
    const unsigned int nz       = sz[2];
    const unsigned int slice_sz = nx * ny;
    const double *A             = Ds + pw;  // rows [pw, pw+ma), LDA = nx

    if (!is_last_op) {
        // chain intermediate: every y column and z slice, active x rows
        kernel(A, u, Dxu + pw);
        return true;
    }
    // terminal: active y columns of the active z slices only
    const unsigned int y_off = pw * nx;
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
    libxsmm_gemmfunction raw_fn = kernel.kernel();
    if (raw_fn) {
        libxsmm_gemm_param args;
        args.a.primary = (void *)A;
        for (unsigned int k = pw; k < nz - pw; k++) {
            args.b.primary = (void *)(u + k * slice_sz + y_off);
            args.c.primary = (void *)(Dxu + k * slice_sz + y_off + pw);
            raw_fn(&args);
        }
        return true;
    }
#endif
    for (unsigned int k = pw; k < nz - pw; k++) {
        kernel(A, u + k * slice_sz + y_off, Dxu + k * slice_sz + y_off + pw);
    }
    return true;
}

inline bool matmul_y_apply(const KernelType &kernel,
                           const double *__restrict__ Ds,
                           double *__restrict__ Dyu,
                           const double *__restrict__ u, const unsigned int *sz,
                           const unsigned int pw, bool is_last_op) {
    if (!kernel) return false;
    const unsigned int nx         = sz[0];
    const unsigned int ny         = sz[1];
    const unsigned int nz         = sz[2];
    const unsigned int slice_size = nx * ny;
    // per z-slice C(ma, N) = U(ma, ny) * D^T[rows y_start..]: A = slice + pw
    // (active x rows, LDA = nx), B = Ds + y_start with TRANS_B selecting the
    // output y columns, C at the same rows/columns. last op: active y and z
    // only; intermediate: all y columns and all z slices
    const unsigned int z_start = is_last_op ? pw : 0u;
    const unsigned int z_end   = is_last_op ? nz - pw : nz;
    const unsigned int y_start = is_last_op ? pw : 0u;
    const double *B            = Ds + y_start;
    const unsigned int c_off   = y_start * nx + pw;
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
    libxsmm_gemmfunction raw_fn = kernel.kernel();
    if (raw_fn) {
        libxsmm_gemm_param args;
        args.b.primary = (void *)B;
        for (unsigned int k = z_start; k < z_end; k++) {
            args.a.primary = (void *)(u + k * slice_size + pw);
            args.c.primary = (void *)(Dyu + k * slice_size + c_off);
            raw_fn(&args);
        }
        return true;
    }
#endif
    for (unsigned int k = z_start; k < z_end; k++) {
        kernel(u + k * slice_size + pw, B, Dyu + k * slice_size + c_off);
    }
    return true;
}

inline bool matmul_z_apply(const KernelType &kernel,
                           const double *__restrict__ Ds,
                           double *__restrict__ Dzu,
                           const double *__restrict__ u, const unsigned int *sz,
                           const unsigned int pw) {
    if (!kernel) return false;
    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int ld_3d = nx * ny;
    // z is always last: per active y-row, C(ma, nz_active) = U(ma, nz) *
    // D^T[active rows] with LDA = LDC = nx*ny (z-stride in the 3D array)
    const double *B          = Ds + pw;
    const unsigned int c_off = pw + pw * ld_3d;
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
    libxsmm_gemmfunction raw_fn = kernel.kernel();
    if (raw_fn) {
        libxsmm_gemm_param args;
        args.b.primary = (void *)B;
        for (unsigned int j = pw; j < ny - pw; j++) {
            args.a.primary = (void *)(u + j * nx + pw);
            args.c.primary = (void *)(Dzu + j * nx + c_off);
            raw_fn(&args);
        }
        return true;
    }
#endif
    for (unsigned int j = pw; j < ny - pw; j++) {
        kernel(u + j * nx + pw, B, Dzu + j * nx + c_off);
    }
    return true;
}

// ----------------------------------------------------------------------
// thread-safety helper: prewarm_kernel_cache
// ----------------------------------------------------------------------
// the kernel cache is shared-mutex-protected, so concurrent lazy creation is
// safe — but the first touch of each new shape still JITs under the write
// lock. call this once at mesh setup (before any threading begins) to build
// every kernel the matrix path dispatches for the given block shapes (the
// full MatmulPlan: x/y/z, last and intermediate, and the fused mixed pairs),
// so a thread's first plan_for() is a cache hit under the shared read lock.
struct BlockShape {
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
};

void prewarm_kernel_cache(const std::vector<BlockShape> &shapes,
                          unsigned int pw);

}  // namespace dendroderivs
