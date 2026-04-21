#include "derivatives/derivs_utils.h"

#include <algorithm>
#include <bitset>

#include "derivatives.h"
#include "libxsmm.h"
#include "refel.h"

// #define DEBUG_COMPACT_DERIVS

namespace dendroderivs {

using KernelType = libxsmm_mmfunction<double>;

std::unordered_map<KernelDimensions, KernelType, KernelDimensionsHash>
    kernel_cache_x;
std::unordered_map<KernelDimensions, KernelType, KernelDimensionsHash>
    kernel_cache_yz;
std::unordered_map<KernelDimensions, KernelType, KernelDimensionsHash>
    kernel_cache_y_direct;
std::unordered_map<ZDirectKernelKey, KernelType, ZDirectKernelKeyHash>
    kernel_cache_z_direct;

void mulMM(double *C, double *A, double *B, int na, int nb) {
    /*  M = number of rows of A and C
        N = number of columns of B and C
        K = number of columns of A and rows of B
    */

    char TA      = 'N';  // don't transpose A
    char TB      = 'N';  // don't transpose B
    double ALPHA = 1.0;  // don't scale result
    double BETA  = 0.0;  // don't add and scale C

    // explicit sizes
    int M        = na;
    int N        = nb;
    int K        = na;
    int LDA      = na;
    int LDB      = na;
    int LDC      = na;

    // make the call
    dgemm_(&TA, &TB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void calculateDerivMatrix(double *D, double *P, double *Q, const int n) {
    int *ipiv = new int[n];

    int info;
    int nx = n;

    dgetrf_(&nx, &nx, P, &nx, ipiv, &info);

    if (info != 0) {
        delete[] ipiv;
        throw std::runtime_error("LU factorization failed: info=" +
                                 std::to_string(info));
    }

    double *Pinv = new double[n * n];
    std::memcpy(Pinv, P, n * n * sizeof(double));

    int lwork    = n * n;
    double *work = new double[lwork];

    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        throw std::runtime_error("Matrix inversion failed: info=" +
                                 std::to_string(info));
    }

    mulMM(D, Pinv, Q, n, n);

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
}

void bandedMatrixStore(double *AB, double *A, const int kl, const int ku,
                       const unsigned int n) {
    int Aindex     = -1;
    int ABindex    = -1;
    double tempVal = -1;

    // at each diagonal term in A, reach up ku times and down kl times
    // omit terms outside of matrix
    for (int j = 0; j < n; j++) {
        for (int i = j - ku; i <= j + kl; i++) {
            // check we're reaching inside the matrix
            if (i >= 0 && i < n && j >= 0 && j < n) {
                Aindex  = i + (n * j);
                tempVal = A[Aindex];
            } else {
                // element does not exist, set to zero
                tempVal = 0;
            }
            // find where to store in the banded array (see documentation)
            int ABi     = (i - (j - ku));
            int ABj     = j;
            int ABn     = kl + ku + 1;
            ABindex     = ABi + (ABn * ABj);
            AB[ABindex] = tempVal;
        }
    }
}

void bandedMatrixVectorMult(double *y, double *A, double *x, int kl, int ku,
                            double alpha, int n) {
    /**
     * These are static so as to not waste time reallocating each and every time
     *  we want to call this method.
     */
    static char TRANS  = 'N';  // don't transpose A
    // is there a way to make alpha static?
    static double BETA = 0.0;  // don't scale and add previous y value
    static int INCX    = 1;    // no spacing between values in x
    static int INCY    = 1;    // no spacing between values in y

    // int M = n;
    // int N = n;
    // int KL = kl;
    // int KU = ku;
    int LDA            = kl + ku + 1;

    dgbmv_(&TRANS, &n /*previously &M*/, &n /*previously &M*/,
           &kl /*previously &KL*/, &ku /*previously &KU*/, &alpha, A, &LDA, x,
           &INCX, &BETA, y, &INCY);
}

int bandedMatrixSolve(char FACT, char TRANS, double *X, double *AB, double *B,
                      double *AFB, int *IPIV, char EQUED, double *R, double *C,
                      double RCOND, double *FERR, double *BERR, double *WORK,
                      int *IWORK, int KL, int KU, unsigned int n) {
    int N     = n;  // need signed
    int NRHS  = 1;  // number of columns of B
    int INFO  = 0;  // for output
    // leading dimensions of matrices
    int LDAB  = KL + KU + 1;
    int LDAFB = 2 * KL + KU + 1;
    int LDB   = N;
    int LDX   = N;

    // make the call to external lapack/blas (idk which) method
    // note that
    dgbsvx_(&FACT, &TRANS, &N, &KL, &KU, &NRHS, AB, &LDAB, AFB, &LDAFB, IPIV,
            &EQUED, R, C, B, &LDB, X, &LDX, &RCOND, FERR, BERR, WORK, IWORK,
            &INFO);

    // if not successful
    if (INFO != 0) {
        std::cerr << "BandedMatrixSolve solve failed. info = " << INFO
                  << std::endl;
        if (INFO < 0) {
            std::cerr << "Illegal value in element " << std::abs(INFO)
                      << std::endl;
        } else if (INFO > 0 && INFO <= n) {
            std::cerr << "U(i,i) is exactly zero for i = " << INFO << std::endl;
        } else {
            std::cerr
                << "U(i,i) is nonsingular, but rcond is less than machine zero."
                << std::endl;
        }
    }

    return INFO;
}

void bandedMatrixSolve(BandedMatrixSolveVars *vars) {
    // NOTE: original implementation of this in 1D repository wondered if this
    // would have a performance impact. Here's a bit of an answer: yes and no.
    // This one might have an advantage for not needing to initialize so many
    // variables, but we're also accessing struct members, which could be
    // potentially costly. Though with the fact that Dendro uses mostly small
    // blocks and this should only be handled upon initialization, I think the
    // "more elegant" solution is fine.

    // make the call
    dgbsvx_(vars->FACT, vars->TRANS, vars->N, vars->KL, vars->KU, vars->NRHS,
            vars->AB, vars->LDAB, vars->AFB, vars->LDAFB, vars->IPIV,
            vars->EQUED, vars->R, vars->C, vars->B, vars->LDB, vars->X,
            vars->LDX, vars->RCOND, vars->FERR, vars->BERR, vars->WORK,
            vars->IWORK, vars->INFO);

    // if not successful, throw so the caller can decide what to do. a
    // library should not terminate the user's program
    int info = *(vars->INFO);
    if (info != 0) {
        std::string detail;
        if (info < 0) {
            detail = "illegal value in element " + std::to_string(std::abs(info));
        } else if (info > 0 && info <= *(vars->N)) {
            detail = "U(i,i) is exactly zero for i = " + std::to_string(info);
        } else {
            detail =
                "U(i,i) is nonsingular, but rcond is less than machine zero";
        }
        throw std::runtime_error("bandedMatrixSolve failed (info=" +
                                 std::to_string(info) + "): " + detail);
    }
}

void print_delta_coeffs(std::string prefix1, std::vector<double> &delta1,
                        std::string prefix2, std::vector<double> &delta2) {
    std::cout << "  . " << prefix1 << " :  ";
    for (auto &ee : delta1) {
        std::cout << ee << " ";
    }
    std::cout << std::endl << "  . " << prefix2 << " :  ";
    for (auto &ee : delta2) {
        std::cout << ee << " ";
    }
    std::cout << std::endl;
}

// Routines that actually compute the derivatives given the right matrices, not
// "dependent" on particular values

// fallback path: used only when libxsmm JIT fails (rare). kept as a pure
// BLAS implementation so the library degrades gracefully rather than crashing
void matmul_x_dim_old(const double *const R, double *const Dxu,
                      const double *const u, const double alpha,
                      const unsigned int *sz, const unsigned int bflag,
                      const unsigned int pw) {
    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    const char TRANSA        = 'N';
    const char TRANSB        = 'N';
    const int M              = nx;
    const int N              = ny;
    const int K              = nx;
    const double beta        = 0.0;

    // skip ghost zones in z at a boundary
    const int z_start = (bflag & (1u << OCT_DIR_BACK)) ? pw : 0;
    const int z_end =
        (bflag & (1u << OCT_DIR_FRONT)) ? nz - pw : (int)nz;

    for (unsigned int k = z_start; k < z_end; k++) {
        const double *u_slice  = u + k * nx * ny;
        const double *du_slice = Dxu + k * nx * ny;
        lapack::dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R, &M,
                               u_slice, &K, &beta, du_slice, &M);
    }
}

// fallback path: used only when libxsmm JIT fails (rare). does the GEMM
// into workspace then transposes back to the output array
void matmul_y_dim_old(const double *const R, double *const Dyu,
                      const double *const u, const double alpha,
                      const unsigned int *sz, double *const workspace,
                      const unsigned int bflag, const unsigned int pw) {
    const unsigned int nx         = sz[0];
    const unsigned int ny         = sz[1];
    const unsigned int nz         = sz[2];

    const char TRANSA             = 'N';
    const char TRANSB             = 'T';
    const int M                   = ny;
    const int N                   = nx;
    const int K                   = ny;
    const double beta             = 0.0;

    const unsigned int slice_size = nx * ny;

    const int z_start = (bflag & (1u << OCT_DIR_BACK)) ? pw : 0;
    const int z_end =
        (bflag & (1u << OCT_DIR_FRONT)) ? nz - pw : (int)nz;

    for (unsigned int k = z_start; k < z_end; k++) {
        const double *u_slice = u + k * slice_size;
        double *du_slice      = Dyu + k * slice_size;

        lapack::dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R, &M,
                               u_slice, &N, &beta, workspace, &M);

        // transpose workspace back to the output layout
#pragma omp simd collapse(2)
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                du_slice[INDEX_N2D(i, j, nx)] = workspace[j + i * ny];
            }
        }
    }
}

// fallback path: used only when libxsmm JIT fails (rare). gathers each
// y-slice into contiguous workspace, runs the GEMM, scatters back out
void matmul_z_dim_old(const double *const R, double *const Dzu,
                      const double *const u, const double alpha,
                      const unsigned int *sz, double *const workspace,
                      const unsigned int bflag, const unsigned int pw) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    const char TRANSA     = 'N';
    const char TRANSB     = 'T';
    const int M           = nz;
    const int N           = nx;
    const int K           = nz;
    const double beta     = 0.0;

    double const *workspace_offset = workspace + nx * nz;

    // z is always called last on 2nd-order mixed derivatives, so the y
    // padding regions are never needed here — skip them
    const unsigned int y_start = pw;
    const unsigned int y_end   = ny - pw;

    for (unsigned int j = y_start; j < y_end; j++) {
#pragma omp simd collapse(2)
        for (unsigned int k = 0; k < nz; k++) {
            for (unsigned int i = 0; i < nx; i++) {
                workspace[k * nx + i] = u[INDEX_3D(i, j, k)];
            }
        }

        lapack::dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R, &M,
                               workspace, &N, &beta, workspace_offset, &M);

#pragma omp simd collapse(2)
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = workspace_offset[k + i * nz];
            }
        }
    }
}

void matmul_x_dim(const double *__restrict__ R, double *__restrict__ Dxu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, const unsigned int bflag,
                  const unsigned int pw) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // skip ghost zones in z if at a boundary
    const int z_start = (bflag & (1u << OCT_DIR_BACK)) ? pw : 0;
    const int z_end =
        (bflag & (1u << OCT_DIR_FRONT)) ? nz - pw : (int)nz;

    const unsigned int n_active_cols = ny * (z_end - z_start);

    // pre-scale D by alpha so the GEMM writes the final answer directly
    double R_scaled[nx * nx];
    for (unsigned int ii = 0; ii < nx * nx; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    // batch all active z-slices into one big GEMM:
    // the 3D array is contiguous in memory, so treating the valid z-range
    // as a 2D matrix (nx, ny*nz_active) lets us do one kernel call
    // instead of nz separate ones
    auto kernel = get_or_create_kernel_x(nx, n_active_cols, nx);

    if (!kernel) {
        std::cout << "FALLING BACK TO MATMUL X DIM" << std::endl;
        return matmul_x_dim_old(R, Dxu, u, alpha, sz, bflag, pw);
    }

    const double *u_start = u + z_start * nx * ny;
    double *du_start      = Dxu + z_start * nx * ny;

    // one GEMM with pre-scaled D: no post-scaling needed
    kernel(R_scaled, u_start, du_start);
}

void matmul_y_dim(const double *__restrict__ R, double *__restrict__ Dyu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, double *__restrict__ workspace,
                  const unsigned int bflag, const unsigned int pw) {
    const unsigned int nx               = sz[0];
    const unsigned int ny               = sz[1];
    const unsigned int nz               = sz[2];

    static const char TRANSA            = 'N';
    static const char TRANSB            = 'T';
    const int M                         = ny;
    const int N                         = nx;
    const int K                         = ny;
    // NOTE: LDA = M, LDB = N, and LDC = M
    // LDB is N because in memory, Y is transposed!

    static const double beta            = 0.0;

    static const double alpha_domatcopy = 1.0;

    auto kernel                         = get_or_create_kernel_yz(M, N, K);

    if (!kernel) {
        std::cout << "FALLING BACK TO MATMUL Y DIM" << std::endl;
        return matmul_y_dim_old(R, Dyu, u, alpha, sz, workspace, bflag, pw);
    }

    const unsigned int slice_size = nx * ny;

    // skip ghost zones in z if we're at a z boundary
    const int z_start = (bflag & (1u << OCT_DIR_BACK)) ? pw : 0;
    const int z_end =
        (bflag & (1u << OCT_DIR_FRONT)) ? nz - pw : (int)nz;

    // pre-scale the derivative matrix by alpha so the GEMM writes the
    // final result directly. R is ny*ny which is small (e.g. 81 doubles),
    // so the copy+scale cost is negligible vs scaling nx*ny per z-slice
    double R_scaled[ny * ny];
    for (unsigned int ii = 0; ii < ny * ny; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    auto kernel_direct = get_or_create_kernel_y_direct(nx, ny);

    if (kernel_direct) {
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        // use the raw JIT function pointer for tighter per-call dispatch
        libxsmm_gemmfunction raw_fn = kernel_direct.kernel();

        if (raw_fn) {
            libxsmm_gemm_param args;
            args.b.primary = (void *)R_scaled;

            for (unsigned int k = z_start; k < z_end; k++) {
                args.a.primary = (void *)(u + k * slice_size);
                args.c.primary = (void *)(Dyu + k * slice_size);
                raw_fn(&args);
            }
            return;
        }
#endif
        // standard C++ wrapper dispatch
        for (unsigned int k = z_start; k < z_end; k++) {
            kernel_direct(u + k * slice_size, R_scaled, Dyu + k * slice_size);
        }
        return;
    }

    // fallback
    std::cout << "FALLING BACK TO MATMUL Y DIM (old)" << std::endl;
    return matmul_y_dim_old(R, Dyu, u, alpha, sz, workspace, bflag, pw);
}

void matmul_z_dim(const double *__restrict__ R, double *__restrict__ Dzu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, double *__restrict__ workspace,
                  const unsigned int bflag, const unsigned int pw) {
    const int nx             = sz[0];
    const int ny             = sz[1];
    const int nz             = sz[2];

    static const char TRANSA = 'N';
    static const char TRANSB = 'T';
    const int M              = nz;
    const int N              = nx;
    const int K              = nz;
    static const double beta = 0.0;

    auto kernel              = get_or_create_kernel_yz(M, N, K);

    if (!kernel) {
        return matmul_z_dim_old(R, Dzu, u, alpha, sz, workspace, bflag, pw);
    }

    // NOTE: due to how derivatives are called, and thanks to the padding width,
    // we can actually skip both padding regions of j, they'll never be needed
    // on the z derivative, because z is always called last on 2nd order mixed
    // derivatives

    const unsigned int y_start    = pw;
    const unsigned int y_end      = ny - pw;

    const unsigned int slice_size = nx * nz;

    // pre-scale D by alpha
    double R_scaled[nz * nz];
    for (int ii = 0; ii < nz * nz; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    // zero-copy approach: strided LDA/LDC = nx*ny so the GEMM reads/writes
    // directly from/to the 3D array with no gather/scatter
    auto kernel_direct = get_or_create_kernel_z_direct(nx, ny, nz);

    if (kernel_direct) {
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        libxsmm_gemmfunction raw_fn = kernel_direct.kernel();

        if (raw_fn) {
            libxsmm_gemm_param args;
            args.b.primary = (void *)R_scaled;

            for (unsigned int j = y_start; j < y_end; j++) {
                args.a.primary = (void *)(u + j * nx);
                args.c.primary = (void *)(Dzu + j * nx);
                raw_fn(&args);
            }
            return;
        }
#endif
        // standard C++ wrapper dispatch
        for (unsigned int j = y_start; j < y_end; j++) {
            kernel_direct(u + j * nx, R_scaled, Dzu + j * nx);
        }
        return;
    }

    // fallback: gather/scatter approach if kernel creation fails
    const double *workspace_out = workspace + nx * ny * nz;

    for (unsigned int j = y_start; j < y_end; j++) {
        double *u_slice  = workspace;
        double *du_slice = (double *)workspace_out;

        for (unsigned int k = 0; k < nz; k++) {
            for (unsigned int i = 0; i < nx; i++) {
                u_slice[k * nx + i] = u[INDEX_3D(i, j, k)];
            }
        }

        kernel(R, u_slice, du_slice);

        for (unsigned int k = 0; k < nz; k++) {
            for (unsigned int i = 0; i < nx; i++) {
                Dzu[INDEX_3D(i, j, k)] = du_slice[k + i * nz] * alpha;
            }
        }
    }
}

}  // namespace dendroderivs
