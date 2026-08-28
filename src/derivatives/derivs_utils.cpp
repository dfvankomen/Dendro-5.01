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
std::shared_mutex kernel_cache_x_mutex;
std::unordered_map<KernelDimensions, KernelType, KernelDimensionsHash>
    kernel_cache_yz;
std::shared_mutex kernel_cache_yz_mutex;
std::unordered_map<KernelDimensions, KernelType, KernelDimensionsHash>
    kernel_cache_y_direct;
std::shared_mutex kernel_cache_y_direct_mutex;
std::unordered_map<ZDirectKernelKey, KernelType, ZDirectKernelKeyHash>
    kernel_cache_z_direct;
std::shared_mutex kernel_cache_z_direct_mutex;
std::unordered_map<KernelKey, KernelType, KernelKeyHash> kernel_cache_ld;
std::shared_mutex kernel_cache_ld_mutex;

void prewarm_kernel_cache(const std::vector<BlockShape> &shapes,
                          unsigned int pw) {
    for (const auto &s : shapes) {
        // X-dim: interior (bflag=0) case uses the full z range
        get_or_create_kernel_x(s.nx, s.ny * s.nz, s.nx);
        // X-dim: boundary variants shrink the active z range by pw on one
        // or both sides. skip if pw=0 or if the variant would be degenerate
        if (pw > 0 && 2u * pw < s.nz) {
            get_or_create_kernel_x(s.nx, s.ny * (s.nz - pw), s.nx);
            get_or_create_kernel_x(s.nx, s.ny * (s.nz - 2u * pw), s.nx);
        }
        // y-direct and z-direct are bflag-independent
        get_or_create_kernel_y_direct(s.nx, s.ny);
        get_or_create_kernel_z_direct(s.nx, s.ny, s.nz);
    }
}

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

    // Full z range: this fallback also serves chain intermediates read by a
    // downstream grad_z (trimming would leave padding uninitialized -> NaN).
    const int z_start        = 0;
    const int z_end          = (int)nz;

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

    // Full z range: this fallback also serves chain intermediates read by a
    // downstream grad_z (trimming would leave padding uninitialized -> NaN).
    const int z_start             = 0;
    const int z_end               = (int)nz;

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
                  const unsigned int pw, bool is_last_op) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // pre-scale D by alpha so the GEMM writes the final answer directly
    double R_scaled[nx * nx];
    for (unsigned int ii = 0; ii < nx * nx; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    // only the active x-rows of the output are ever read (see the contract in
    // derivs_utils.h), so the GEMM is D[pw:pw+ma, :] * u with LDA = nx and the
    // output pointer offset by pw. ma is the active count padded for SIMD
    const unsigned int ma       = active_m_padded(nx, pw);
    const unsigned int slice_sz = nx * ny;
    const double *A             = R_scaled + pw;

    if (is_last_op) {
        // skip the y and z padding of the output as well: one GEMM per active
        // z-slice over the active y columns only. at eleorder=6 that is
        // (8, 7, 13) x 7 slices instead of (13, 169, 13)
        const unsigned int y_start   = pw;
        const unsigned int z_start   = pw;
        const unsigned int z_end     = nz - pw;
        const unsigned int ny_active = ny - 2u * pw;
        const unsigned int y_off     = y_start * nx;

        auto kernel = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma,
                                              ny_active, nx, nx, nx, nx);
        if (!kernel) {
            std::cout << "FALLING BACK TO MATMUL X DIM (last-op path)\n";
            return matmul_x_dim_old(R, Dxu, u, alpha, sz, bflag, pw);
        }

#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        libxsmm_gemmfunction raw_fn = kernel.kernel();
        if (raw_fn) {
            libxsmm_gemm_param args;
            args.a.primary = (void *)A;
            for (unsigned int k = z_start; k < z_end; k++) {
                args.b.primary = (void *)(u + k * slice_sz + y_off);
                args.c.primary = (void *)(Dxu + k * slice_sz + y_off + pw);
                raw_fn(&args);
            }
            return;
        }
#endif
        for (unsigned int k = z_start; k < z_end; k++) {
            kernel(A, u + k * slice_sz + y_off, Dxu + k * slice_sz + y_off + pw);
        }
        return;
    }

    // Chain-intermediate path: a downstream grad_y / grad_z reads this across
    // the full y and z range (at active x rows), so write every column
    auto kernel = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma, ny * nz,
                                          nx, nx, nx, nx);
    if (!kernel) {
        std::cout << "FALLING BACK TO MATMUL X DIM" << std::endl;
        return matmul_x_dim_old(R, Dxu, u, alpha, sz, bflag, pw);
    }
    kernel(A, u, Dxu + pw);
}

void matmul_y_dim(const double *__restrict__ R, double *__restrict__ Dyu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, double *__restrict__ workspace,
                  const unsigned int bflag, const unsigned int pw,
                  bool is_last_op) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // pre-scale the derivative matrix by alpha so the GEMM writes the
    // final result directly. R is ny*ny which is small (e.g. 81 doubles),
    // so the copy+scale cost is negligible vs scaling nx*ny per z-slice
    double R_scaled[ny * ny];
    for (unsigned int ii = 0; ii < ny * ny; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    // per z-slice: C(ma, N) = U(ma, ny) * D^T restricted to the output
    // columns. A = u_slice + pw (active x rows, LDA = nx), TRANS_B with the
    // operator rows selecting which y outputs are produced, C offset to the
    // same rows/columns. last op: only the active y columns and z slices.
    // intermediate: all y columns and all z slices (a downstream y/z op reads
    // them), still only the active x rows
    const unsigned int ma         = active_m_padded(nx, pw);
    const unsigned int slice_size = nx * ny;
    const unsigned int z_start    = is_last_op ? pw : 0u;
    const unsigned int z_end      = is_last_op ? nz - pw : nz;
    const unsigned int y_start    = is_last_op ? pw : 0u;
    const unsigned int n_cols     = is_last_op ? ny - 2u * pw : ny;

    auto kernel = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma, n_cols,
                                          ny, nx, ny, nx);
    if (!kernel) {
        std::cout << "FALLING BACK TO MATMUL Y DIM (old)" << std::endl;
        return matmul_y_dim_old(R, Dyu, u, alpha, sz, workspace, bflag, pw);
    }

    const double *B          = R_scaled + y_start;
    const unsigned int c_off = y_start * nx + pw;

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
        return;
    }
#endif
    for (unsigned int k = z_start; k < z_end; k++) {
        kernel(u + k * slice_size + pw, B, Dyu + k * slice_size + c_off);
    }
}

void matmul_z_dim(const double *__restrict__ R, double *__restrict__ Dzu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, double *__restrict__ workspace,
                  const unsigned int bflag, const unsigned int pw) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // pre-scale D by alpha
    double R_scaled[nz * nz];
    for (unsigned int ii = 0; ii < nz * nz; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    // z is always the last operator in a chain, so only the active region of
    // the output is needed: per active y-row, C(ma, nz_active) = U(ma, nz) *
    // D^T[active rows] with LDA = LDC = nx*ny so the kernel reads/writes at
    // z-stride straight in the 3D array (no gather/scatter)
    const unsigned int ma        = active_m_padded(nx, pw);
    const unsigned int ld_3d     = nx * ny;
    const unsigned int y_start   = pw;
    const unsigned int y_end     = ny - pw;
    const unsigned int z_start   = pw;
    const unsigned int nz_active = nz - 2u * pw;

    auto kernel = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                          nz_active, nz, ld_3d, nz, ld_3d);
    if (!kernel) {
        return matmul_z_dim_old(R, Dzu, u, alpha, sz, workspace, bflag, pw);
    }

    const double *B          = R_scaled + z_start;
    const unsigned int c_off = pw + z_start * ld_3d;

#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
    libxsmm_gemmfunction raw_fn = kernel.kernel();
    if (raw_fn) {
        libxsmm_gemm_param args;
        args.b.primary = (void *)B;
        for (unsigned int j = y_start; j < y_end; j++) {
            args.a.primary = (void *)(u + j * nx + pw);
            args.c.primary = (void *)(Dzu + j * nx + c_off);
            raw_fn(&args);
        }
        return;
    }
#endif
    for (unsigned int j = y_start; j < y_end; j++) {
        kernel(u + j * nx + pw, B, Dzu + j * nx + c_off);
    }
}

}  // namespace dendroderivs
