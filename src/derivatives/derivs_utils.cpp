#include "derivatives/derivs_utils.h"

#include <algorithm>
#include <bitset>

#include "derivatives.h"
#include "libxsmm.h"
#include "refel.h"

// #define DEBUG_COMPACT_DERIVS

namespace dendroderivs {

std::unordered_map<KernelKey, KernelType, KernelKeyHash> kernel_cache_ld;
std::shared_mutex kernel_cache_ld_mutex;

void prewarm_kernel_cache(const std::vector<BlockShape> &shapes,
                          unsigned int pw) {
    for (const auto &s : shapes) {
        // the plan is what every Derivs instance dispatches from; building
        // it here populates the shared cache with exactly those kernels
        const unsigned int sz[3] = {s.nx, s.ny, s.nz};
        (void)build_matmul_plan(sz, pw);
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

MatmulPlan build_matmul_plan(const unsigned int *sz, unsigned int pw) {
    MatmulPlan p;
    p.nx = sz[0];
    p.ny = sz[1];
    p.nz = sz[2];
    p.pw = pw;
    p.ma = active_m_padded(p.nx, pw);
    const int nx = p.nx, ny = p.ny, nz = p.nz, ma = p.ma, ipw = pw;
    // x: D[pw:pw+ma, :] * u, LDA = nx. last: active y columns per active
    // z-slice; intermediate: all ny*nz columns in one GEMM
    p.kx_last = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma,
                                        ny - 2 * ipw, nx, nx, nx, nx);
    p.kx_int  = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma, ny * nz,
                                        nx, nx, nx, nx);
    p.kx_last_acc = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma,
                                            ny - 2 * ipw, nx, nx, nx, nx, true);
    // y: U(ma, ny) * D^T per z-slice, LDA = LDC = nx, LDB = ny
    p.ky_last = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                        ny - 2 * ipw, ny, nx, ny, nx);
    p.ky_int  = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma, ny, ny,
                                        nx, ny, nx);
    // z: strided, LDA = LDC = nx*ny, active z columns only
    p.kz      = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                        nz - 2 * ipw, nz, nx * ny, nz, nx * ny);
    // fused xy: per active z-slice, tmp(ma, ny) = D_x[pw:, :] * u_slice (LDC =
    // ma), then w(ma, ny_active) = tmp * D_y^T[active rows] (LDA = ma, LDC = nx)
    p.kxy1 = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma, ny, nx, nx, nx,
                                     ma);
    p.kxy2 = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                     ny - 2 * ipw, ny, ma, ny, nx);
    // fused xz: per active y-row, tmp(ma, nz) = D_x[pw:, :] * u_at_j (LDB =
    // nx*ny), then w(ma, nz_active) = tmp * D_z^T[active rows] (LDC = nx*ny)
    p.kxz1 = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma, nz, nx, nx,
                                     nx * ny, ma);
    p.kxz2 = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                     nz - 2 * ipw, nz, ma, nz, nx * ny);
    // fused yz: pass 1 per z-slice, tmp_slab(ma, ny_active) = u_slice *
    // D_y^T[active rows] (LDC = ma); pass 2 per active y, w(ma, nz_active) =
    // tmp_at_j (LDA = ma*ny_active, the z stride in tmp) * D_z^T[active rows]
    p.kyz1 = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                     ny - 2 * ipw, ny, nx, ny, ma);
    p.kyz2 = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                     nz - 2 * ipw, nz, ma * (ny - 2 * ipw), nz,
                                     nx * ny);
    // accumulating terminal y / z (C += ...) for summed-axis operators
    p.ky_last_acc = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                            ny - 2 * ipw, ny, nx, ny, nx, true);
    p.kz_acc      = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                            nz - 2 * ipw, nz, nx * ny, nz,
                                            nx * ny, true);
    p.valid   = true;
    return p;
}

// standalone wrappers: scale the operator, fetch the one kernel this call
// needs from the shared cache, apply. Derivs instances bypass these (they
// memoize the plan and the scaled operator) but the addressing is identical
void matmul_x_dim(const double *__restrict__ R, double *__restrict__ Dxu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, const unsigned int bflag,
                  const unsigned int pw, bool is_last_op) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double R_scaled[nx * nx];
    for (unsigned int ii = 0; ii < nx * nx; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    const unsigned int ma = active_m_padded(nx, pw);
    KernelType kernel =
        is_last_op ? get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma,
                                             ny - 2u * pw, nx, nx, nx, nx)
                   : get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_NONE, ma,
                                             ny * nz, nx, nx, nx, nx);
    if (!matmul_x_apply(kernel, R_scaled, Dxu, u, sz, pw, is_last_op)) {
        std::cout << "FALLING BACK TO MATMUL X DIM" << std::endl;
        matmul_x_dim_old(R, Dxu, u, alpha, sz, bflag, pw);
    }
}

void matmul_y_dim(const double *__restrict__ R, double *__restrict__ Dyu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, double *__restrict__ workspace,
                  const unsigned int bflag, const unsigned int pw,
                  bool is_last_op) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];

    double R_scaled[ny * ny];
    for (unsigned int ii = 0; ii < ny * ny; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    const unsigned int ma     = active_m_padded(nx, pw);
    const unsigned int n_cols = is_last_op ? ny - 2u * pw : ny;
    KernelType kernel = get_or_create_kernel_ld(LIBXSMM_GEMM_FLAG_TRANS_B, ma,
                                                n_cols, ny, nx, ny, nx);
    if (!matmul_y_apply(kernel, R_scaled, Dyu, u, sz, pw, is_last_op)) {
        std::cout << "FALLING BACK TO MATMUL Y DIM (old)" << std::endl;
        matmul_y_dim_old(R, Dyu, u, alpha, sz, workspace, bflag, pw);
    }
}

void matmul_z_dim(const double *__restrict__ R, double *__restrict__ Dzu,
                  const double *__restrict__ u, const double alpha,
                  const unsigned int *sz, double *__restrict__ workspace,
                  const unsigned int bflag, const unsigned int pw) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double R_scaled[nz * nz];
    for (unsigned int ii = 0; ii < nz * nz; ii++) {
        R_scaled[ii] = R[ii] * alpha;
    }

    const unsigned int ma = active_m_padded(nx, pw);
    KernelType kernel     = get_or_create_kernel_ld(
        LIBXSMM_GEMM_FLAG_TRANS_B, ma, nz - 2u * pw, nz, nx * ny, nz, nx * ny);
    if (!matmul_z_apply(kernel, R_scaled, Dzu, u, sz, pw)) {
        matmul_z_dim_old(R, Dzu, u, alpha, sz, workspace, bflag, pw);
    }
}

}  // namespace dendroderivs
