#include "derivatives/derivs_utils.h"

#include <algorithm>

namespace dendroderivs {

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

    // if not successful
    int info = *(vars->INFO);
    if (info != 0) {
        std::cerr << "BandedMatrixSolve solve failed. info = " << info
                  << std::endl;
        if (info < 0) {
            std::cerr << "Illegal value in element " << std::abs(info)
                      << std::endl;
        } else if (info > 0 && info <= *(vars->N)) {
            std::cerr << "U(i,i) is exactly zero for i = " << info << std::endl;
        } else {
            std::cerr << "U(i,i) is nonsingular, but rcond is less than "
                         "machine zero."
                      << std::endl;
        }

        // TODO: probably better to throw errors and propagate it up all the way
        // with try catch, fortunately upon full exit the OS will actually clean
        // everything up
        exit(info);
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

void matmul_x_dim(const double *const R, double *const Dxu,
                  const double *const u, const double alpha,
                  const unsigned int *sz) {
    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    static const char TRANSA = 'N';
    static const char TRANSB = 'N';

    const int M              = nx;
    const int N              = ny;
    const int K              = nx;
    // NOTE: LDA = M, LDB = K, and LDC = M

    static const double beta = 0.0;

#ifdef USE_XSMM_MAT_MUL
#ifdef SOLVER_ENABLE_MERGED_BLOCKS
    kernel_type *kernel = m_kernel_storage[simple_hash(M, N)];
#endif
#endif

    for (unsigned int k = 0; k < nz; k++) {
        // avoid pointer arithmitic, use direct pointer location for compiler
        // optimization
        const double *u_slice  = u + k * nx * ny;
        const double *du_slice = Dxu + k * nx * ny;

#ifdef USE_XSMM_MAT_MUL
        // N = ny;
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        // performs C_mn = alpha * A_mk * B_kn + beta * C_mn

        // for the x_der case, m = k = nx
#ifdef SOLVER_ENABLE_MERGED_BLOCKS
        (*kernel)(R_mat_use, u_curr_chunk, du_curr_chunk);
#else
        (*m_kernel_x)(R_mat_use, u_curr_chunk, du_curr_chunk);
#endif

#else
        // this allows us to use const in the functionality
        lapack::dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R, &M,
                               u_slice, &K, &beta, du_slice, &M);
#endif
    }

    // TODO: investigate why the kernel won't take an alpha
#ifdef USE_XSMM_MAT_MUL
#pragma omp simd
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dxu[ii] *= alpha;
    }
#endif
}

void matmul_y_dim(const double *const R, double *const Dyu,
                  const double *const u, const double alpha,
                  const unsigned int *sz, double *const workspace) {
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

#ifdef USE_XSMM_MAT_MUL
#ifdef SOLVER_ENABLE_MERGED_BLOCKS
    kernel_type *kernel = m_kernel_transpose_storage[simple_hash(M, N)];
#endif
#endif

    const unsigned int slice_size = nx * ny;

    for (unsigned int k = 0; k < nz; k++) {
        // avoid pointer arithmitic, use direct pointer location for compiler
        // optimization
        const double *u_slice = u + k * slice_size;
        double *du_slice      = Dyu + k * slice_size;

#ifdef USE_XSMM_MAT_MUL
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

#ifdef SOLVER_ENABLE_MERGED_BLOCKS
        (*kernel)(R_mat_use, u_slice, workspace);
#else
        (*m_kernel_y)(R_mat_use, u_slice, workspace);
#endif

#else

        lapack::dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R, &M,
                               u_slice, &N, &beta, workspace, &M);
#endif

#ifdef __INTEL_MKL__
        // copy back out from workspace using domatcopy if using intel mkl
        mkl_domatcopy('C', 'T', ny, nx, alpha_domatcopy, workspace, ny,
                      du_slice, nx);
#else
        // TODO: see if there's a faster way to copy (i.e. SSE?)
        // the data is transposed so it's much harder to just copy all at
        // once

#pragma omp simd collapse(2)
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                du_slice[INDEX_N2D(i, j, nx)] = workspace[j + i * ny];
            }
        }
#endif
    }

    // NOTE: it is currently faster for these derivatives if we calculate
    // them
#ifdef USE_XSMM_MAT_MUL
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dyu[ii] *= alpha;
    }
#endif
}

void matmul_z_dim(const double *const R, double *const Dzu,
                  const double *const u, const double alpha,
                  const unsigned int *sz, double *const workspace) {
    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    static const char TRANSA = 'N';
    static const char TRANSB = 'T';
    const int M              = nz;
    const int N              = nx;
    const int K              = nz;
    static const double beta = 0.0;

#ifdef USE_XSMM_MAT_MUL
#ifdef SOLVER_ENABLE_MERGED_BLOCKS
    kernel_type *kernel = m_kernel_transpose_storage[simple_hash(M, N)];
#endif
#endif

    double const *workspace_offset = workspace + ny * nz;

    for (unsigned int j = 0; j < ny; j++) {
        // #pragma omp simd collapse(2)
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                workspace[k * nx + i] = u[INDEX_3D(i, j, k)];
            }
        }

        // for (unsigned int k = 0; k < nz; k++) {
        //     // copy the slice of X values over
        //     std::copy_n(&u[INDEX_3D(0, j, k)], nx,
        //                 &workspace[INDEX_N2D(0, k, nx)]);
        // }

#ifdef USE_XSMM_MAT_MUL
#ifdef SOLVER_ENABLE_MERGED_BLOCKS
        (*kernel)(R_mat_use, workspace, workspace_offset);
#else
        // now do the faster math multiplcation
        (*m_kernel_z)(R_mat_use, workspace, workspace_offset);
#endif

#else

        // now we have a transposed matrix to send into dgemm_
        lapack::dgemm_cpp_safe(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R, &M,
                               workspace, &N, &beta, workspace_offset, &M);

#endif

        // #pragma omp simd collapse(2)
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = workspace_offset[k + i * nz];
            }
        }
    }

    // NOTE: it is currently faster for these derivatives if we calculate
    // them
#ifdef USE_XSMM_MAT_MUL
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dzu[ii] *= alpha;
    }
#endif
}

}  // namespace dendroderivs
