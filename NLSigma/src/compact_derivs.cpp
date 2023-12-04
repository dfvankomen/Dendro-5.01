#include "compact_derivs.h"

#include <cstdint>

// #define FASTER_DERIV_CALC_VIA_MATRIX_MULT
// #define PRINT_COMPACT_MATRICES

namespace dendro_cfd {

// initialize a "global" cfd object
CompactFiniteDiff cfd(0, 0);

CompactFiniteDiff::CompactFiniteDiff(const unsigned int num_dim,
                                     const unsigned int padding_size,
                                     const DerType deriv_type,
                                     const DerType2nd second_deriv_type,
                                     const FilterType filter_type) {
    if (deriv_type != CFD_NONE && deriv_type != CFD_P1_O4 &&
        deriv_type != CFD_P1_O6 && deriv_type != CFD_Q1_O6_ETA1 &&
        deriv_type != CFD_KIM_O4 && deriv_type != CFD_HAMR_O4 &&
        deriv_type != CFD_JT_O6 && deriv_type != EXPLCT_FD_O4 &&
        deriv_type != EXPLCT_FD_O6 && deriv_type != EXPLCT_FD_O8) {
        throw std::invalid_argument(
            "Couldn't initialize CFD object, deriv type was not a valid 'base' "
            "type: deriv_type = " +
            std::to_string(deriv_type));
    }

    m_deriv_type = deriv_type;
    m_second_deriv_type = second_deriv_type;
    m_filter_type = filter_type;
    m_curr_dim_size = num_dim;
    m_padding_size = padding_size;

    initialize_cfd_storage();

    if (num_dim == 0) {
        return;
    }

    initialize_cfd_matrix();
    initialize_cfd_filter();
}

CompactFiniteDiff::~CompactFiniteDiff() {
    // make sure we delete the cfd matrix to avoid memory leaks
    delete_cfd_matrices();
}

void CompactFiniteDiff::change_dim_size(const unsigned int dim_size) {
    if (m_curr_dim_size == dim_size) {
        return;
    } else {
        delete_cfd_matrices();

        m_curr_dim_size = dim_size;

        initialize_cfd_storage();

        // if deriv type is none, for some reason, just exit

        initialize_cfd_matrix();
        initialize_cfd_filter();
    }
}

void CompactFiniteDiff::initialize_cfd_storage() {
    // NOTE: 0 indicates that it's initialized with all elements set to 0

    for (CompactDerivValueOrder ii = CompactDerivValueOrder::DERIV_NORM;
         ii < CompactDerivValueOrder::R_MAT_END;
         ii = static_cast<CompactDerivValueOrder>((size_t)ii + 1)) {
        m_RMatrices[ii] = new double[m_curr_dim_size * m_curr_dim_size]();
    }

    // NOTE: the () syntax only works with C++ 11 or greater, may need to
    // use std::fill_n(array, n, 0); to 0 set the data or use std::memset(array,
    // 0, sizeof *array * size)

    m_u1d = new double[m_curr_dim_size];
    m_u2d = new double[m_curr_dim_size * m_curr_dim_size];
    m_du1d = new double[m_curr_dim_size];
    m_du2d = new double[m_curr_dim_size * m_curr_dim_size];
}

void CompactFiniteDiff::initialize_cfd_matrix() {
    // temporary P and Q storage used in calculations
    double *P = new double[m_curr_dim_size * m_curr_dim_size]();
    double *Q = new double[m_curr_dim_size * m_curr_dim_size]();

    // for each cfd matrix that needs to be initialized, we need the "base"
    // matrix, the "left edge" and the "right edge" to be safe.

    // TODO: it might be necessary if the processor knows what boundaries it has
    // but these matrices are small compared to the blocks that they're probably
    // alright plus, these are only calculated once and not over and over again.

    for (CompactDerivValueOrder ii = CompactDerivValueOrder::DERIV_NORM;
         ii < CompactDerivValueOrder::FILT_NORM;
         ii = static_cast<CompactDerivValueOrder>((size_t)ii + 1)) {
        if (m_deriv_type == EXPLCT_FD_O4 &&
            (ii == CompactDerivValueOrder::DERIV_NORM ||
             ii == CompactDerivValueOrder::DERIV_LEFT ||
             ii == CompactDerivValueOrder::DERIV_RIGHT ||
             ii == CompactDerivValueOrder::DERIV_LEFTRIGHT)) {
        } else if (m_deriv_type == EXPLCT_FD_O6 &&
                   (ii == CompactDerivValueOrder::DERIV_NORM ||
                    ii == CompactDerivValueOrder::DERIV_LEFT ||
                    ii == CompactDerivValueOrder::DERIV_RIGHT ||
                    ii == CompactDerivValueOrder::DERIV_LEFTRIGHT)) {
        } else if (m_deriv_type == EXPLCT_FD_O8 &&
                   (ii == CompactDerivValueOrder::DERIV_NORM ||
                    ii == CompactDerivValueOrder::DERIV_LEFT ||
                    ii == CompactDerivValueOrder::DERIV_RIGHT ||
                    ii == CompactDerivValueOrder::DERIV_LEFTRIGHT)) {
        }

        setArrToZero(P, m_curr_dim_size * m_curr_dim_size);
        setArrToZero(Q, m_curr_dim_size * m_curr_dim_size);

        if (ii < CompactDerivValueOrder::DERIV_2ND_NORM &&
            m_deriv_type == CFD_NONE) {
            continue;
        } else if (ii < CompactDerivValueOrder::FILT_NORM &&
                   m_second_deriv_type == CFD2ND_NONE) {
            continue;
        }

        // figure out if it's a left boundary
        bool left_b = (ii == DERIV_LEFT || ii == DERIV_2ND_LEFT ||
                       ii == DERIV_LEFTRIGHT || ii == DERIV_2ND_LEFTRIGHT)
                          ? true
                          : false;
        bool right_b = (ii == DERIV_RIGHT || ii == DERIV_2ND_RIGHT ||
                        ii == DERIV_LEFTRIGHT || ii == DERIV_2ND_LEFTRIGHT)
                           ? true
                           : false;

        // check for explicit filters
        if (m_deriv_type == EXPLCT_FD_O4 || m_deriv_type == EXPLCT_FD_O6 ||
            m_deriv_type == EXPLCT_FD_O8) {
            if (ii == CompactDerivValueOrder::DERIV_NORM ||
                ii == CompactDerivValueOrder::DERIV_LEFT ||
                ii == CompactDerivValueOrder::DERIV_RIGHT ||
                ii == CompactDerivValueOrder::DERIV_LEFTRIGHT) {
                buildDerivExplicitRMatrix(m_RMatrices[ii], m_padding_size,
                                          m_curr_dim_size, m_deriv_type, left_b,
                                          right_b);
                continue;
            }
        }

        if (m_second_deriv_type == EXPLCT2ND_FD_O4 ||
            m_second_deriv_type == EXPLCT2ND_FD_O6 ||
            m_second_deriv_type == EXPLCT2ND_FD_O8) {
            if (ii == CompactDerivValueOrder::DERIV_2ND_NORM ||
                ii == CompactDerivValueOrder::DERIV_2ND_LEFT ||
                ii == CompactDerivValueOrder::DERIV_2ND_RIGHT ||
                ii == CompactDerivValueOrder::DERIV_2ND_LEFTRIGHT) {
                build2ndDerivExplicitRMatrix(
                    m_RMatrices[ii], m_padding_size, m_curr_dim_size,
                    m_second_deriv_type, left_b, right_b);
                continue;
            }
        }

        if (ii == DERIV_NORM || ii == DERIV_RIGHT || ii == DERIV_LEFT ||
            ii == DERIV_LEFTRIGHT) {
            buildPandQMatrices(P, Q, m_padding_size, m_curr_dim_size,
                               m_deriv_type, left_b, right_b);
        } else if (ii == DERIV_2ND_NORM || ii == DERIV_2ND_RIGHT ||
                   ii == DERIV_2ND_LEFT) {
            buildPandQMatrices2ndOrder(P, Q, m_padding_size, m_curr_dim_size,
                                       m_second_deriv_type, left_b, right_b);
        } else if (ii == DERIV_2ND_LEFTRIGHT) {
            // don't bother with second order left right...
            continue;
        } else {
            throw std::out_of_range(
                "Something went wrong when trying to build P and Q "
                "matrices...");
        }

#ifdef PRINT_COMPACT_MATRICES
        std::cout << "\nP MATRIX no=" << ii << std::endl;
        print_square_mat(P, m_curr_dim_size);

        std::cout << "\nQ MATRIX no=" << ii << std::endl;
        print_square_mat(Q, m_curr_dim_size);
#endif

        calculateDerivMatrix(m_RMatrices[ii], P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
        std::cout << "\nDERIV MATRIX no=" << ii << std::endl;
        print_square_mat(m_RMatrices[ii], m_curr_dim_size);
#endif
    }

    delete[] P;
    delete[] Q;
}

void CompactFiniteDiff::initialize_cfd_filter() {
    // exit early on filter none
    if (m_filter_type == FILT_NONE || m_filter_type == FILT_KO_DISS) {
        return;
    }

    // temporary P and Q storage used in calculations
    double *P = new double[m_curr_dim_size * m_curr_dim_size]();
    double *Q = new double[m_curr_dim_size * m_curr_dim_size]();

    for (CompactDerivValueOrder ii = CompactDerivValueOrder::FILT_NORM;
         ii < CompactDerivValueOrder::R_MAT_END;
         ii = static_cast<CompactDerivValueOrder>((size_t)ii + 1)) {
        setArrToZero(P, m_curr_dim_size * m_curr_dim_size);
        setArrToZero(Q, m_curr_dim_size * m_curr_dim_size);

        // figure out if it's a left boundary
        bool left_b = (ii == FILT_LEFT || ii == FILT_LEFTRIGHT) ? true : false;
        bool right_b =
            (ii == FILT_RIGHT || ii == FILT_LEFTRIGHT) ? true : false;

        buildPandQFilterMatrices(P, Q, m_padding_size, m_curr_dim_size,
                                 m_filter_type, left_b, right_b);

#ifdef PRINT_COMPACT_MATRICES
        std::cout << "\nP MATRIX no=" << ii << std::endl;
        print_square_mat(P, m_curr_dim_size);

        std::cout << "\nQ MATRIX no=" << ii << std::endl;
        print_square_mat(Q, m_curr_dim_size);
#endif

        calculateDerivMatrix(m_RMatrices[ii], P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
        std::cout << "\nFILTER MATRIX no=" << ii << std::endl;
        print_square_mat(m_RMatrices[ii], m_curr_dim_size);
#endif
    }

    delete[] P;
    delete[] Q;
}

void CompactFiniteDiff::delete_cfd_matrices() {
    delete[] m_u1d;
    delete[] m_u2d;
    delete[] m_du1d;
    delete[] m_du2d;

    for (CompactDerivValueOrder ii = CompactDerivValueOrder::DERIV_NORM;
         ii < CompactDerivValueOrder::R_MAT_END;
         ii = static_cast<CompactDerivValueOrder>((size_t)ii + 1)) {
        delete[] m_RMatrices[ii];
    }
}

void CompactFiniteDiff::cfd_x(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // std::cout << "Nx, ny, nz: " << nx << " " << ny << " " << nz << std::endl;

    char TRANSA = 'N';
    char TRANSB = 'N';

    int M = nx;
    int N = ny;
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    const double alpha = 1.0 / dx;
#else
    double alpha = 1.0 / dx;
#endif
    int K = nx;

    // NOTE: LDA, LDB, and LDC should be nx, ny, and nz
    // TODO: fix for non-square sizes
    int LDA = nx;
    int LDB = ny;
    int LDC = nx;

    double *u_curr_chunk = (double *)u;
    double *du_curr_chunk = (double *)Dxu;

    double beta = 0.0;

    double *R_mat_use = nullptr;

    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_LEFT)) && !(bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_NORM];
    } else if ((bflag & (1u << OCT_DIR_LEFT)) &&
               !(bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_LEFT)) &&
               (bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_RIGHT];
    } else {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_LEFTRIGHT];
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, M, N, K, 1.0, 0.0);
    assert(kernel);
#endif

    // const libxsmm_mmfunction<double, double, LIBXSMM_PREFETCH_AUTO>
    // xmm(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, LDA, LDB, LDC, alpha,
    // beta);

    for (unsigned int k = 0; k < nz; k++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // N = ny;
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        // performs C_mn = alpha * A_mk * B_kn + beta * C_mn

        // for the x_der case, m = k = nx

        kernel(R_mat_use, u_curr_chunk, du_curr_chunk);

#else

        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R_mat_use, &LDA,
               u_curr_chunk, &LDB, &beta, du_curr_chunk, &LDC);

#endif

        u_curr_chunk += nx * ny;
        du_curr_chunk += nx * ny;
    }

    // TODO: investigate why the kernel won't take 1/dx as its alpha
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dxu[ii] *= 1 / dx;
    }
#endif
}

void CompactFiniteDiff::cfd_y(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'T';
    int M = ny;
    int N = nx;
    int K = ny;

    double alpha = 1.0 / dy;
    double beta = 0.0;

    double *u_curr_chunk = (double *)u;
    double *du_curr_chunk = (double *)Dyu;

    double *R_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_DOWN)) && !(bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_NORM];
    } else if ((bflag & (1u << OCT_DIR_DOWN)) &&
               !(bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_DOWN)) &&
               (bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_RIGHT];
    } else {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_LEFTRIGHT];
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, M, N, K, 1.0, 0.0);
    assert(kernel);
#endif

    for (unsigned int k = 0; k < nz; k++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        kernel(R_mat_use, u_curr_chunk, m_du2d);

#else

        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R_mat_use, &M,
               u_curr_chunk, &K, &beta, m_du2d, &M);

#endif
        // TODO: see if there's a faster way to copy (i.e. SSE?)
        // the data is transposed so it's much harder to just copy all at once
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                Dyu[INDEX_3D(i, j, k)] = m_du2d[j + i * ny];
            }
        }

        // NOTE: this is probably faster on Intel, but for now we'll do the form
        // above libxsmm_otrans(du_curr_chunk, m_du2d, sizeof(double), ny, nx,
        // nx, ny);
        // TODO: mkl's mkl_domatcopy might be even better!

        // update u_curr_chunk
        u_curr_chunk += nx * ny;
        du_curr_chunk += nx * ny;
    }

    // NOTE: it is currently faster for these derivatives if we calculate them
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dyu[ii] *= 1 / dy;
    }
#endif
}

void CompactFiniteDiff::cfd_z(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'N';
    int M = nz;
    int K = nz;
    double alpha = 1.0 / dz;
    double beta = 0.0;

    double *R_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_BACK)) && !(bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_NORM];
    } else if ((bflag & (1u << OCT_DIR_BACK)) &&
               !(bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_BACK)) &&
               (bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_RIGHT];
    } else {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_LEFTRIGHT];
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    int N = nx;
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, M, N, K, 1.0, 0.0);
    assert(kernel);
#else
    int N = 1;
#endif

    for (unsigned int j = 0; j < ny; j++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        for (unsigned int k = 0; k < nz; k++) {
            // copy the slice of X values over
            std::copy_n(&u[INDEX_3D(0, j, k)], nx, &m_u2d[INDEX_N2D(0, k, nx)]);
        }

        // now do the faster math multiplcation
        kernel(R_mat_use, m_u2d, m_du2d);

        // then we just stick it back in, but now in memory it's stored as z0,
        // z1, z2,... then increases in x so we can't just do copy_n
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = m_du2d[k + i * nz];
            }
        }

#else
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                m_u1d[k] = u[INDEX_3D(i, j, k)];
            }
        }

        dgemv_(&TRANSA, &M, &K, &alpha, R_mat_use, &M, m_u1d, &N, &beta, m_du1d,
               &N);

        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = m_du1d[k];
            }
        }

#endif
    }

    // NOTE: it is currently faster for these derivatives if we calculate them
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dzu[ii] *= 1 / dz;
    }
#endif
}

void CompactFiniteDiff::cfd_xx(double *const Dxu, const double *const u,
                               const double dx, const unsigned int *sz,
                               unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // std::cout << "Nx, ny, nz: " << nx << " " << ny << " " << nz << std::endl;

    char TRANSA = 'N';
    char TRANSB = 'N';

    int M = nx;
    int N = ny;
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    const double alpha = 1.0 / (dx * dx);
#else
    double alpha = 1.0 / (dx * dx);
#endif
    int K = nx;

    // NOTE: LDA, LDB, and LDC should be nx, ny, and nz
    // TODO: fix for non-square sizes
    int LDA = nx;
    int LDB = ny;
    int LDC = nx;

    double *u_curr_chunk = (double *)u;
    double *du_curr_chunk = (double *)Dxu;

    double beta = 0.0;

    double *R_mat_use = nullptr;

    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_LEFT)) && !(bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_NORM];
    } else if ((bflag & (1u << OCT_DIR_LEFT)) &&
               !(bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_LEFT)) &&
               (bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_RIGHT];
    } else {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_LEFTRIGHT];
        printf("Uh oh, DERIV_2ND_LEFTRIGHT was reached!");
    }

    // std::cout << "bflag is: " << bflag << " dx: " << dx << " alpha: " <<
    // alpha << std::endl; print_square_mat(R_mat_use, nx); exit(0);

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, M, N, K, 1.0, 0.0);
    assert(kernel);
#endif

    // const libxsmm_mmfunction<double, double, LIBXSMM_PREFETCH_AUTO>
    // xmm(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, LDA, LDB, LDC, alpha,
    // beta);

    for (unsigned int k = 0; k < nz; k++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // N = ny;
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        // performs C_mn = alpha * A_mk * B_kn + beta * C_mn

        // for the x_der case, m = k = nx

        kernel(R_mat_use, u_curr_chunk, du_curr_chunk);

#else

        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R_mat_use, &LDA,
               u_curr_chunk, &LDB, &beta, du_curr_chunk, &LDC);

#endif

        u_curr_chunk += nx * ny;
        du_curr_chunk += nx * ny;
    }

    // TODO: investigate why the kernel won't take 1/dx as its alpha
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dxu[ii] *= alpha;
    }
#endif

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                const int pp = INDEX_3D(i, j, k);
                if (std::isnan(Dxu[pp])) {
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
                    std::cout << "alpha is " << alpha;
                    std::cout << "previous value was " << Dxu[pp - 1]
                              << std::endl;
                    exit(0);
                }
            }
        }
    }
}

void CompactFiniteDiff::cfd_yy(double *const Dyu, const double *const u,
                               const double dy, const unsigned int *sz,
                               unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'T';
    int M = ny;
    int N = nx;
    int K = ny;

    double alpha = 1.0 / (dy * dy);
    double beta = 0.0;

    double *u_curr_chunk = (double *)u;
    double *du_curr_chunk = (double *)Dyu;

    double *R_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_DOWN)) && !(bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_NORM];
    } else if ((bflag & (1u << OCT_DIR_DOWN)) &&
               !(bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_DOWN)) &&
               (bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_RIGHT];
    } else {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_LEFTRIGHT];
        printf("Uh oh, DERIV_2ND_LEFTRIGHT was reached!");
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    // #if 0
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, M, N, K, 1.0, 0.0);
    assert(kernel);
#endif

    for (unsigned int k = 0; k < nz; k++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // #if 0
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        kernel(R_mat_use, u_curr_chunk, m_du2d);

#else

        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R_mat_use, &M,
               u_curr_chunk, &K, &beta, m_du2d, &M);

#endif
        // TODO: see if there's a faster way to copy (i.e. SSE?)
        // the data is transposed so it's much harder to just copy all at once
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                Dyu[INDEX_3D(i, j, k)] = m_du2d[j + i * ny];
            }
        }

        // NOTE: this is probably faster on Intel, but for now we'll do the form
        // above libxsmm_otrans(du_curr_chunk, m_du2d, sizeof(double), ny, nx,
        // nx, ny);
        // TODO: mkl's mkl_domatcopy might be even better!

        // update u_curr_chunk
        u_curr_chunk += nx * ny;
        du_curr_chunk += nx * ny;
    }

    // NOTE: it is currently faster for these derivatives if we calculate them
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dyu[ii] *= alpha;
    }
#endif

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                const int pp = INDEX_3D(i, j, k);
                if (std::isnan(Dyu[pp])) {
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
                    exit(0);
                }
            }
        }
    }
}

void CompactFiniteDiff::cfd_zz(double *const Dzu, const double *const u,
                               const double dz, const unsigned int *sz,
                               unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'N';
    int M = nz;
    int K = nz;
    double alpha = 1.0 / (dz * dz);
    double beta = 0.0;

    double *R_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_BACK)) && !(bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_NORM];
    } else if ((bflag & (1u << OCT_DIR_BACK)) &&
               !(bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_BACK)) &&
               (bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_RIGHT];
    } else {
        R_mat_use = m_RMatrices[CompactDerivValueOrder::DERIV_2ND_LEFTRIGHT];
        printf("Uh oh, DERIV_2ND_LEFTRIGHT was reached!");
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    int N = nx;
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, M, N, K, 1.0, 0.0);
    assert(kernel);
#else
    int N = 1;
#endif

    for (unsigned int j = 0; j < ny; j++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        for (unsigned int k = 0; k < nz; k++) {
            // copy the slice of X values over
            std::copy_n(&u[INDEX_3D(0, j, k)], nx, &m_u2d[INDEX_N2D(0, k, nx)]);
        }

        // now do the faster math multiplcation
        kernel(R_mat_use, m_u2d, m_du2d);

        // then we just stick it back in, but now in memory it's stored as z0,
        // z1, z2,... then increases in x so we can't just do copy_n
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = m_du2d[k + i * nz];
            }
        }

#else
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                m_u1d[k] = u[INDEX_3D(i, j, k)];
            }
        }

        dgemv_(&TRANSA, &M, &K, &alpha, R_mat_use, &M, m_u1d, &N, &beta, m_du1d,
               &N);

        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = m_du1d[k];
            }
        }

#endif
    }

    // NOTE: it is currently faster for these derivatives if we calculate them
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dzu[ii] *= alpha;
    }
#endif

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                const int pp = INDEX_3D(i, j, k);
                if (std::isnan(Dzu[pp])) {
                    std::cout << "NAN detected function " << __func__
                              << " file: " << __FILE__ << " line: " << __LINE__
                              << std::endl;
                    exit(0);
                }
            }
        }
    }
}

void CompactFiniteDiff::filter_cfd_x(double *const u, double *const filtx_work,
                                     const double dx, const unsigned int *sz,
                                     unsigned bflag) {
    if (m_filter_type == FILT_NONE || m_filter_type == FILT_KO_DISS) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // copy u to filtx_work
    std::copy_n(u, nx * ny * nz, filtx_work);

    char TRANSA = 'N';
    char TRANSB = 'N';

    int M = nx;
    int N = ny;
    int K = nx;

    // NOTE: LDA, LDB, and LDC should be nx, ny, and nz
    // TODO: fix for non-square sizes
    int LDA = nx;
    int LDB = ny;
    int LDC = nx;

    double *u_curr_chunk = (double *)u;
    double *filtu_curr_chunk = (double *)filtx_work;

    double alpha = 1.0;
    // TODO: beta should actuall be a parameter!

    double *RF_mat_use = nullptr;

    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_LEFT)) && !(bflag & (1u << OCT_DIR_RIGHT))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_NORM];
    } else if ((bflag & (1u << OCT_DIR_LEFT)) &&
               !(bflag & (1u << OCT_DIR_RIGHT))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_LEFT)) &&
               (bflag & (1u << OCT_DIR_RIGHT))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_RIGHT];

    } else {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_LEFTRIGHT];
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, M, N, K, 1.0, m_beta_filt);
    assert(kernel);
#endif

    for (unsigned int k = 0; k < nx; k++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        // performs C_mn = alpha * A_mk * B_kn + beta * C_mn

        // for the x_der case, m = k = nx
        kernel(RF_mat_use, u_curr_chunk, filtu_curr_chunk);

#else
        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, RF_mat_use, &LDA,
               u_curr_chunk, &LDB, &m_beta_filt, filtu_curr_chunk, &LDC);

#endif
        u_curr_chunk += nx * ny;
        filtu_curr_chunk += nx * ny;
    }

    // we don't want B to overwrite C other wise we end up with errors
    std::copy_n(filtx_work, nx * ny * nz, u);
}

void CompactFiniteDiff::filter_cfd_y(double *const u, double *const filty_work,
                                     const double dy, const unsigned int *sz,
                                     unsigned bflag) {
    if (m_filter_type == FILT_NONE || m_filter_type == FILT_KO_DISS) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // copy u to filtx_work
    // std::copy_n(u, nx * ny * nz, filty_work);

    char TRANSA = 'N';
    char TRANSB = 'T';
    int M = ny;
    int N = nx;
    int K = ny;
    double alpha = 1.0;

    // TODO: beta needs to be a parameter
    double beta = 1.0;

    double *u_curr_chunk = (double *)u;
    double *filtu_curr_chunk = (double *)filty_work;

    double *RF_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_DOWN)) && !(bflag & (1u << OCT_DIR_UP))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_NORM];
    } else if ((bflag & (1u << OCT_DIR_DOWN)) &&
               !(bflag & (1u << OCT_DIR_UP))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_DOWN)) &&
               (bflag & (1u << OCT_DIR_UP))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_RIGHT];
    } else {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_LEFTRIGHT];
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, M, N, K, 1.0, m_beta_filt);
    assert(kernel);
#endif

    for (unsigned int k = 0; k < nz; k++) {
        // transpose into filty_work as a copy
        for (unsigned int j = 0; j < ny; j++) {
            for (unsigned int i = 0; i < nx; i++) {
                filty_work[j + i * ny] = u_curr_chunk[i + j * nx];
            }
        }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        kernel(RF_mat_use, u_curr_chunk, filty_work);

#else
        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, RF_mat_use, &M,
               u_curr_chunk, &K, &m_beta_filt, filty_work, &M);

#endif

        // then transpose right back
        // TODO: see if there's a faster way to copy (i.e. SSE?)
        // the data is transposed so it's much harder to just copy all at once
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                // u[INDEX_3D(i, j, k)] += filty_work[j + i * ny];
                u_curr_chunk[i + j * nx] = filty_work[j + i * ny];
            }
        }
        u_curr_chunk += nx * ny;
    }
}

void CompactFiniteDiff::filter_cfd_z(double *const u, double *const filtz_work,
                                     const double dz, const unsigned int *sz,
                                     unsigned bflag) {
    if (m_filter_type == FILT_NONE || m_filter_type == FILT_KO_DISS) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'N';
    int M = nz;
    int N = 1;
    int K = nz;
    double alpha = 1.0;
    double beta = 1.0;

    double *RF_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_BACK)) && !(bflag & (1u << OCT_DIR_FRONT))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_NORM];
    } else if ((bflag & (1u << OCT_DIR_BACK)) &&
               !(bflag & (1u << OCT_DIR_FRONT))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_LEFT];
    } else if (!(bflag & (1u << OCT_DIR_BACK)) &&
               (bflag & (1u << OCT_DIR_FRONT))) {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_RIGHT];
    } else {
        RF_mat_use = m_RMatrices[CompactDerivValueOrder::FILT_LEFTRIGHT];
    }

    for (unsigned int j = 0; j < ny; j++) {
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                m_u1d[k] = u[INDEX_3D(i, j, k)];
                filtz_work[k] = u[INDEX_3D(i, j, k)];
            }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
            dgemv_(&TRANSA, &M, &K, &alpha, RF_mat_use, &M, m_u1d, &N, &beta,
                   filtz_work, &N);

            for (unsigned int k = 0; k < nz; k++) {
                u[INDEX_3D(i, j, k)] = filtz_work[k];
            }
#else
            dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, RF_mat_use, &M, m_u1d,
                   &K, &m_beta_filt, filtz_work, &M);

            for (int k = 0; k < nz; k++) {
                u[INDEX_3D(i, j, k)] = filtz_work[k];
            }

#endif
        }
    }
}

DerType getDerTypeForEdges(const DerType derivtype,
                           const BoundaryType boundary) {
    DerType doptions_CFD_P1_O4[4] = {CFD_P1_O4, CFD_DRCHLT_ORDER_4,
                                     CFD_P1_O4_CLOSE, CFD_P1_O4_L4_CLOSE};
    DerType doptions_CFD_P1_O6[4] = {CFD_P1_O6, CFD_DRCHLT_ORDER_6,
                                     CFD_P1_O6_CLOSE, CFD_P1_O6_L6_CLOSE};
    DerType doptions_CFD_Q1_O6_ETA1[4] = {CFD_Q1_O6_ETA1, CFD_DRCHLT_Q6,
                                          CFD_Q1_O6_ETA1_CLOSE,
                                          CFD_P1_O6_L6_CLOSE};

    // the doptions to use
    DerType *doptions;

    switch (derivtype) {
        case CFD_P1_O4:
            doptions = doptions_CFD_P1_O4;
            break;
        case CFD_P1_O6:
            doptions = doptions_CFD_P1_O6;
            break;
        case CFD_Q1_O6_ETA1:
            doptions = doptions_CFD_Q1_O6_ETA1;
            break;

        default:
            throw std::invalid_argument(
                "Invalid type of CFD derivative called! derivtype=" +
                std::to_string(derivtype));
            break;
    }

    switch (boundary) {
        case BLOCK_CFD_DIRICHLET:
        case BLOCK_PHYS_BOUNDARY:
            return doptions[1];
        case BLOCK_CFD_CLOSURE:
            return doptions[2];
        case BLOCK_CFD_LOPSIDE_CLOSURE:
            return doptions[3];
        default:
            return doptions[1];
    }
}

DerType2nd get2ndDerTypeForEdges(const DerType2nd derivtype,
                                 const BoundaryType boundary) {
    DerType2nd doptions_CFD2ND_P2_O4[4] = {CFD2ND_P2_O4, CFD2ND_DRCHLT_ORDER_4,
                                           CFD2ND_P2_O4_CLOSE,
                                           CFD2ND_P2_O4_L4_CLOSE};
    DerType2nd doptions_CFD2ND_P2_O6[4] = {CFD2ND_P2_O6, CFD2ND_DRCHLT_ORDER_6,
                                           CFD2ND_P2_O6_CLOSE,
                                           CFD2ND_P2_O6_L6_CLOSE};
    DerType2nd doptions_CFD2ND_Q2_O6_ETA1[4] = {
        CFD2ND_Q2_O6_ETA1, CFD2ND_DRCHLT_Q6, CFD2ND_Q2_O6_ETA1_CLOSE,
        CFD2ND_P2_O6_L6_CLOSE};

    // the doptions to use
    DerType2nd *doptions;

    switch (derivtype) {
        case CFD2ND_P2_O4:
            doptions = doptions_CFD2ND_P2_O4;
            break;
        case CFD2ND_P2_O6:
            doptions = doptions_CFD2ND_P2_O6;
            break;
        case CFD2ND_Q2_O6_ETA1:
            doptions = doptions_CFD2ND_Q2_O6_ETA1;
            break;

        default:
            throw std::invalid_argument(
                "Invalid type of CFD2ND derivative called! derivtype=" +
                std::to_string(derivtype));
            break;
    }

    switch (boundary) {
        case BLOCK_CFD_DIRICHLET:
        case BLOCK_PHYS_BOUNDARY:
            return doptions[2];
            // return doptions[1];
        case BLOCK_CFD_CLOSURE:
            return doptions[2];
        case BLOCK_CFD_LOPSIDE_CLOSURE:
            return doptions[3];
        default:
            return doptions[1];
    }
}

void buildPandQMatrices(double *P, double *Q, const uint32_t padding,
                        const uint32_t n, const DerType derivtype,
                        const bool is_left_edge, const bool is_right_edge) {
    // NOTE: we're pretending that all of the "mpi" or "block" boundaries
    // are treated equally. We only need to account for physical "left" and
    // "right" edges

    // NOTE: (2) we're also assuming that P and Q are initialized to **zero**.
    // There are no guarantees in this function if they are not.
    // std::cout << derivtype << " is the deriv type" << std::endl;

    uint32_t curr_n = n;
    uint32_t i_start = 0;
    uint32_t i_end = n;
    uint32_t j_start = 0;
    uint32_t j_end = n;

    if (is_left_edge) {
        // initialize the "diagonal" in the padding to 1
        for (uint32_t ii = 0; ii < padding; ii++) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_start += padding;
        j_start += padding;
        curr_n -= padding;
    }

    if (is_right_edge) {
        // initialize bottom "diagonal" in padding to 1 as well
        for (uint32_t ii = n - 1; ii >= n - padding; ii--) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_end -= padding;
        j_end -= padding;
        curr_n -= padding;
    }

    // std::cout << "i : " << i_start << " " << i_end << std::endl;
    // std::cout << "j : " << j_start << " " << j_end << std::endl;

    // NOTE: when at the "edges", we need a temporary array that can be copied
    // over
    double *tempP = nullptr;
    double *tempQ = nullptr;

    if (is_left_edge or is_right_edge) {
        // initialize tempP to be a "smaller" square matrix for use
        tempP = new double[curr_n * curr_n]();
        tempQ = new double[curr_n * curr_n]();
    } else {
        // just use the same pointer value, then no need to adjust later even
        tempP = P;
        tempQ = Q;
    }

    if (derivtype == CFD_P1_O4 || derivtype == CFD_P1_O6 ||
        derivtype == CFD_Q1_O6_ETA1) {
        // NOTE: this is only for the NONISOTROPIC matrices!!!

        // now build up the method object that will be used to calculate the
        // in-between values
        CFDMethod method(derivtype);

        int ibgn = 0;
        int iend = 0;

        DerType leftEdgeDtype;
        DerType rightEdgeDtype;

        if (is_left_edge) {
            leftEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_PHYS_BOUNDARY);
        } else {
            // TODO: update the boundary type based on what we want to build in
            leftEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_CFD_DIRICHLET);
        }

        if (is_right_edge) {
            rightEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_PHYS_BOUNDARY);
        } else {
            // TODO: update the boundary type based on what we want to build in
            rightEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_CFD_DIRICHLET);
        }

        buildMatrixLeft(tempP, tempQ, &ibgn, leftEdgeDtype, padding, curr_n);
        buildMatrixRight(tempP, tempQ, &iend, rightEdgeDtype, padding, curr_n);

        for (int i = ibgn; i <= iend; i++) {
            for (int k = -method.Ld; k <= method.Rd; k++) {
                if (!(i > -1) && !(i < curr_n)) {
                    if (is_left_edge or is_right_edge) {
                        delete[] tempP;
                        delete[] tempQ;
                    }
                    throw std::out_of_range(
                        "I is either less than zero or greater than curr_n! "
                        "i=" +
                        std::to_string(i) +
                        " curr_n=" + std::to_string(curr_n));
                }
                if (!((i + k) > -1) && !((i + k) < curr_n)) {
                    if (is_left_edge or is_right_edge) {
                        delete[] tempP;
                        delete[] tempQ;
                    }
                    throw std::out_of_range(
                        "i + k is either less than 1 or greater than curr_n! "
                        "i=" +
                        std::to_string(i + k) + " k=" + std::to_string(k) +
                        " curr_n=" + std::to_string(curr_n));
                }

                tempP[INDEX_N2D(i, i + k, curr_n)] =
                    method.alpha[k + method.Ld];
            }
            for (int k = -method.Lf; k <= method.Rf; k++) {
                if (!(i > -1) && !(i < curr_n)) {
                    throw std::out_of_range(
                        "(i is either less than zero or greater than curr_n! "
                        "i=" +
                        std::to_string(i) +
                        " curr_n=" + std::to_string(curr_n));
                }
                if (!((i + k) > -1) && !((i + k) < curr_n)) {
                    throw std::out_of_range(
                        "i + k is either less than 1 or greater than curr_n! "
                        "i=" +
                        std::to_string(i + k) + " k=" + std::to_string(k) +
                        " curr_n=" + std::to_string(curr_n));
                }

                tempQ[INDEX_N2D(i, i + k, curr_n)] = method.a[k + method.Lf];
            }
        }
    } else if (derivtype == CFD_KIM_O4) {
        // build Kim4 P and Q

        initializeKim4PQ(tempP, tempQ, curr_n);
    } else if (derivtype == CFD_HAMR_O4) {
        // build HAMR 4 P
        HAMRDeriv4_dP(tempP, curr_n);

        // then build Q
        HAMRDeriv4_dQ(tempQ, curr_n);
    } else if (derivtype == CFD_JT_O6) {
        // build JTP Deriv P
        JTPDeriv6_dP(tempP, curr_n);

        // then build Q
        JTPDeriv6_dQ(tempQ, curr_n);
    } else if (derivtype == CFD_NONE) {
        // just.... do nothing... keep them at zeros
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "dendro_cfd::buildPandQMatrices should never be called with a "
            "CFD_NONE deriv type!");
    } else {
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "The CFD deriv type was not one of the valid options. derivtype=" +
            std::to_string(derivtype));
    }

    // copy the values back in
    // NOTE: the use of j and i assumes ROW-MAJOR order, but it will just copy a
    // square matrix in no matter what, so it's not a big issue
    if (is_left_edge or is_right_edge) {
        // then memcopy the "chunks" to where they go inside the matrix
        uint32_t temp_arr_i = 0;
        // iterate over the rows
        for (uint32_t jj = j_start; jj < j_end; jj++) {
            // ii will only go from empty rows we actually need to fill...
            // j will start at "j_start" and go until "j_end" where we need to
            // fill memory start index of our main array

            uint32_t temp_start = INDEX_N2D(0, temp_arr_i, curr_n);
            // uint32_t temp_end = INDEX_N2D(curr_n - 1, temp_arr_i, curr_n);

            std::copy_n(&tempP[temp_start], curr_n, &P[INDEX_2D(i_start, jj)]);
            std::copy_n(&tempQ[temp_start], curr_n, &Q[INDEX_2D(i_start, jj)]);

            // increment temp_arr "row" value
            temp_arr_i++;
        }
        // clear up our temporary arrays we don't need
        delete[] tempP;
        delete[] tempQ;
    }
    // NOTE: tempP doesn't need to be deleted if it was not initialized,
    // so we don't need to delete it unless we're dealing with left/right edges
}

void buildPandQMatrices2ndOrder(double *P, double *Q, const uint32_t padding,
                                const uint32_t n, const DerType2nd derivtype,
                                const bool is_left_edge,
                                const bool is_right_edge) {
    // NOTE: we're pretending that all of the "mpi" or "block" boundaries
    // are treated equally. We only need to account for physical "left" and
    // "right" edges

    // NOTE: (2) we're also assuming that P and Q are initialized to **zero**.
    // There are no guarantees in this function if they are not.
    // std::cout << derivtype << " is the deriv type" << std::endl;

    uint32_t curr_n = n;
    uint32_t i_start = 0;
    uint32_t i_end = n;
    uint32_t j_start = 0;
    uint32_t j_end = n;

    if (is_left_edge) {
        // initialize the "diagonal" in the padding to 1
        for (uint32_t ii = 0; ii < padding; ii++) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_start += padding;
        j_start += padding;
        curr_n -= padding;
    }

    if (is_right_edge) {
        // initialize bottom "diagonal" in padding to 1 as well
        for (uint32_t ii = n - 1; ii >= n - padding; ii--) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_end -= padding;
        j_end -= padding;
        curr_n -= padding;
    }

    // std::cout << "i : " << i_start << " " << i_end << std::endl;
    // std::cout << "j : " << j_start << " " << j_end << std::endl;

    // NOTE: when at the "edges", we need a temporary array that can be copied
    // over
    double *tempP = nullptr;
    double *tempQ = nullptr;

    if (is_left_edge or is_right_edge) {
        // initialize tempP to be a "smaller" square matrix for use
        tempP = new double[curr_n * curr_n]();
        tempQ = new double[curr_n * curr_n]();
    } else {
        // just use the same pointer value, then no need to adjust later even
        tempP = P;
        tempQ = Q;
    }

    if (derivtype == CFD2ND_P2_O4 || derivtype == CFD2ND_P2_O6 ||
        derivtype == CFD2ND_Q2_O6_ETA1) {
        // NOTE: this is only for the NONISOTROPIC matrices!!!

        // now build up the method object that will be used to calculate the
        // in-between values
        CFDMethod2nd method(derivtype);

        int ibgn = 0;
        int iend = 0;

        DerType2nd leftEdgeDtype;
        DerType2nd rightEdgeDtype;

        if (is_left_edge) {
            leftEdgeDtype = get2ndDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_PHYS_BOUNDARY);
        } else {
            // TODO: update the boundary type based on what we want to build in
            leftEdgeDtype = get2ndDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_CFD_DIRICHLET);
        }

        if (is_right_edge) {
            rightEdgeDtype = get2ndDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_PHYS_BOUNDARY);
        } else {
            // TODO: update the boundary type based on what we want to build in
            rightEdgeDtype = get2ndDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_CFD_DIRICHLET);
        }

        buildMatrixLeft2nd(tempP, tempQ, &ibgn, leftEdgeDtype, padding, curr_n);
        buildMatrixRight2nd(tempP, tempQ, &iend, rightEdgeDtype, padding,
                            curr_n);

        for (int i = ibgn; i <= iend; i++) {
            for (int k = -method.Ld; k <= method.Rd; k++) {
                if (!(i > -1) && !(i < curr_n)) {
                    if (is_left_edge or is_right_edge) {
                        delete[] tempP;
                        delete[] tempQ;
                    }
                    throw std::out_of_range(
                        "I is either less than zero or greater than curr_n! "
                        "i=" +
                        std::to_string(i) +
                        " curr_n=" + std::to_string(curr_n));
                }
                if (!((i + k) > -1) && !((i + k) < curr_n)) {
                    if (is_left_edge or is_right_edge) {
                        delete[] tempP;
                        delete[] tempQ;
                    }
                    throw std::out_of_range(
                        "i + k is either less than 1 or greater than curr_n! "
                        "i=" +
                        std::to_string(i + k) + " k=" + std::to_string(k) +
                        " curr_n=" + std::to_string(curr_n));
                }

                tempP[INDEX_N2D(i, i + k, curr_n)] =
                    method.alpha[k + method.Ld];
            }
            for (int k = -method.Lf; k <= method.Rf; k++) {
                if (!(i > -1) && !(i < curr_n)) {
                    throw std::out_of_range(
                        "(i is either less than zero or greater than curr_n! "
                        "i=" +
                        std::to_string(i) +
                        " curr_n=" + std::to_string(curr_n));
                }
                if (!((i + k) > -1) && !((i + k) < curr_n)) {
                    throw std::out_of_range(
                        "i + k is either less than 1 or greater than curr_n! "
                        "i=" +
                        std::to_string(i + k) + " k=" + std::to_string(k) +
                        " curr_n=" + std::to_string(curr_n));
                }

                tempQ[INDEX_N2D(i, i + k, curr_n)] = method.a[k + method.Lf];
            }
        }
    } else if (derivtype == CFD2ND_KIM_O4) {
        // build Kim4 P and Q
        throw dendro_cfd::CFDNotImplemented(
            "Kim 2nd Order is not yet implemented!");

        // initializeKim4PQ(tempP, tempQ, curr_n);
    } else if (derivtype == CFD2ND_HAMR_O4) {
        // build HAMR 4 P
        throw dendro_cfd::CFDNotImplemented(
            "HAMR 2nd Order is not yet implemented!");
        // HAMRDeriv4_dP(tempP, curr_n);

        // then build Q
        // HAMRDeriv4_dQ(tempQ, curr_n);
    } else if (derivtype == CFD2ND_JT_O6) {
        // build JTP Deriv P
        throw dendro_cfd::CFDNotImplemented(
            "JT 2nd Order is not yet implemented!");
        // JTPDeriv6_dP(tempP, curr_n);

        // then build Q
        // JTPDeriv6_dQ(tempQ, curr_n);
    } else if (derivtype == CFD2ND_NONE) {
        // just.... do nothing... keep them at zeros
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "dendro_cfd::buildPandQMatrices should never be called with a "
            "CFD_NONE deriv type!");
    } else {
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "The CFD deriv type was not one of the valid options. derivtype=" +
            std::to_string(derivtype));
    }

    // copy the values back in
    // NOTE: the use of j and i assumes ROW-MAJOR order, but it will just copy a
    // square matrix in no matter what, so it's not a big issue
    if (is_left_edge or is_right_edge) {
        // then memcopy the "chunks" to where they go inside the matrix
        uint32_t temp_arr_i = 0;
        // iterate over the rows
        for (uint32_t jj = j_start; jj < j_end; jj++) {
            // ii will only go from empty rows we actually need to fill...
            // j will start at "j_start" and go until "j_end" where we need to
            // fill memory start index of our main array

            uint32_t temp_start = INDEX_N2D(0, temp_arr_i, curr_n);
            // uint32_t temp_end = INDEX_N2D(curr_n - 1, temp_arr_i, curr_n);

            std::copy_n(&tempP[temp_start], curr_n, &P[INDEX_2D(i_start, jj)]);
            std::copy_n(&tempQ[temp_start], curr_n, &Q[INDEX_2D(i_start, jj)]);

            // increment temp_arr "row" value
            temp_arr_i++;
        }
        // clear up our temporary arrays we don't need
        delete[] tempP;
        delete[] tempQ;
    }
    // NOTE: tempP doesn't need to be deleted if it was not initialized,
    // so we don't need to delete it unless we're dealing with left/right edges
}

void buildPandQFilterMatrices(double *P, double *Q, const uint32_t padding,
                              const uint32_t n, const FilterType filtertype,
                              const bool is_left_edge,
                              const bool is_right_edge) {
    // NOTE: we're pretending that all of the "mpi" or "block" boundaries
    // are treated equally. We only need to account for physical "left" and
    // "right" edges

    // NOTE: (2) we're also assuming that P and Q are initialized to **zero**.
    // There are no guarantees in this function if they are not.
    // std::cout << filtertype << " is the filter type" << std::endl;

    uint32_t curr_n = n;
    uint32_t i_start = 0;
    uint32_t i_end = n;
    uint32_t j_start = 0;
    uint32_t j_end = n;

    if (is_left_edge) {
        // initialize the "diagonal" in the padding to 1
        for (uint32_t ii = 0; ii < padding; ii++) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_start += padding;
        j_start += padding;
        curr_n -= padding;
    }

    if (is_right_edge) {
        // initialize bottom "diagonal" in padding to 1 as well
        for (uint32_t ii = n - 1; ii >= n - padding; ii--) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_end -= padding;
        j_end -= padding;
        curr_n -= padding;
    }

    // std::cout << "i : " << i_start << " " << i_end << std::endl;
    // std::cout << "j : " << j_start << " " << j_end << std::endl;

    // NOTE: when at the "edges", we need a temporary array that can be copied
    // over
    double *tempP = nullptr;
    double *tempQ = nullptr;

    if (is_left_edge or is_right_edge) {
        // initialize tempP to be a "smaller" square matrix for use
        tempP = new double[curr_n * curr_n]();
        tempQ = new double[curr_n * curr_n]();
    } else {
        // just use the same pointer value, then no need to adjust later even
        tempP = P;
        tempQ = Q;
    }

    if (filtertype == FILT_KIM_6) {
        // build Kim4 P and Q

        initializeKim6FilterPQ(tempP, tempQ, curr_n);
    } else if (filtertype == FILT_JT_6) {
        // TODO: NOT CURRENTLY IMPLEMENTED
        std::cerr << "WARNING: The JT 6 filter is not yet ready! This will "
                     "lead to unexpected results!"
                  << std::endl;
    } else if (filtertype == FILT_JT_8) {
        // TODO: NOT CURRENTLY IMPLEMENTED
        std::cerr << "WARNING: The JT 8 filter is not yet ready! This will "
                     "lead to unexpected results!"
                  << std::endl;
    } else if (filtertype == FILT_NONE || filtertype == FILT_KO_DISS) {
        // just.... do nothing... keep them at zeros
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "dendro_cfd::buildPandQFilterMatrices should never be called with "
            "a "
            "CFD_NONE deriv type!");
    } else {
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "The filter type was not one of the valid options. filtertype=" +
            std::to_string(filtertype));
    }

    // copy the values back in
    // NOTE: the use of j and i assumes ROW-MAJOR order, but it will just copy a
    // square matrix in no matter what, so it's not a big issue
    if (is_left_edge or is_right_edge) {
        // then memcopy the "chunks" to where they go inside the matrix
        uint32_t temp_arr_i = 0;
        // iterate over the rows
        for (uint32_t jj = j_start; jj < j_end; jj++) {
            // ii will only go from empty rows we actually need to fill...
            // j will start at "j_start" and go until "j_end" where we need to
            // fill memory start index of our main array

            uint32_t temp_start = INDEX_N2D(0, temp_arr_i, curr_n);
            // uint32_t temp_end = INDEX_N2D(curr_n - 1, temp_arr_i, curr_n);

            std::copy_n(&tempP[temp_start], curr_n, &P[INDEX_2D(i_start, jj)]);
            std::copy_n(&tempQ[temp_start], curr_n, &Q[INDEX_2D(i_start, jj)]);

            // increment temp_arr "row" value
            temp_arr_i++;
        }
        // clear up our temporary arrays we don't need
        delete[] tempP;
        delete[] tempQ;
    }
    // NOTE: tempP doesn't need to be deleted if it was not initialized,
    // so we don't need to delete it unless we're dealing with left/right edges
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

    // memcpy is faster than the for loops!
    std::memcpy(Pinv, P, n * n * sizeof(double));

    int lwork = n * n;
    double *work = new double[lwork];

    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        throw std::runtime_error("Matrix inversion failed: info=" +
                                 std::to_string(info));
    }

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "P INVERSE" << std::endl;
    print_square_mat(Pinv, n);
#endif

    mulMM(D, Pinv, Q, n, n);

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
}

void mulMM(double *C, double *A, double *B, int na, int nb) {
    /*  M = number of rows of A and C
        N = number of columns of B and C
        K = number of columns of A and rows of B
    */

    char TA[4], TB[4];
    double ALPHA = 1.0;
    double BETA = 0.0;
    sprintf(TA, "N");
    sprintf(TB, "N");
    int M = na;
    int N = nb;
    int K = na;
    int LDA = na;
    int LDB = na;
    int LDC = na;

    dgemm_(TA, TB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void setArrToZero(double *arr, const int n) {
    for (uint16_t ii = 0; ii < n; ii++) {
        arr[ii] = 0.0;
    }
}

void buildMatrixLeft(double *P, double *Q, int *xib, const DerType dtype,
                     const int nghosts, const int n) {
    int ib = 0;

    switch (dtype) {
        case CFD_DRCHLT_ORDER_4: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 3.0;

            Q[INDEX_2D(0, 0)] = -17.0 / 6.0;
            Q[INDEX_2D(0, 1)] = 3.0 / 2.0;
            Q[INDEX_2D(0, 2)] = 3.0 / 2.0;
            Q[INDEX_2D(0, 3)] = -1.0 / 6.0;
            ib = 1;
        } break;

        case CFD_DRCHLT_ORDER_6: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 5.0;

            P[INDEX_2D(1, 0)] = 2.0 / 11.0;
            P[INDEX_2D(1, 1)] = 1.0;
            P[INDEX_2D(1, 2)] = 2.0 / 11.0;

            Q[INDEX_2D(0, 0)] = -197.0 / 60.0;
            Q[INDEX_2D(0, 1)] = -5.0 / 12.0;
            Q[INDEX_2D(0, 2)] = 5.0;
            Q[INDEX_2D(0, 3)] = -5.0 / 3.0;
            Q[INDEX_2D(0, 4)] = 5.0 / 12.0;
            Q[INDEX_2D(0, 5)] = -1.0 / 20.0;

            Q[INDEX_2D(1, 0)] = -20.0 / 33.0;
            Q[INDEX_2D(1, 1)] = -35.0 / 132.0;
            Q[INDEX_2D(1, 2)] = 34.0 / 33.0;
            Q[INDEX_2D(1, 3)] = -7.0 / 33.0;
            Q[INDEX_2D(1, 4)] = 2.0 / 33.0;
            Q[INDEX_2D(1, 5)] = -1.0 / 132.0;
            ib = 2;
        } break;

        case CFD_DRCHLT_Q6: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 5.0;

            P[INDEX_2D(1, 0)] = 2.0 / 11.0;
            P[INDEX_2D(1, 1)] = 1.0;
            P[INDEX_2D(1, 2)] = 2.0 / 11.0;

            P[INDEX_2D(2, 1)] = 1.0 / 3.0;
            P[INDEX_2D(2, 2)] = 1.0;
            P[INDEX_2D(2, 3)] = 1.0 / 3.0;

            Q[INDEX_2D(0, 0)] = -197.0 / 60.0;
            Q[INDEX_2D(0, 1)] = -5.0 / 12.0;
            Q[INDEX_2D(0, 2)] = 5.0;
            Q[INDEX_2D(0, 3)] = -5.0 / 3.0;
            Q[INDEX_2D(0, 4)] = 5.0 / 12.0;
            Q[INDEX_2D(0, 5)] = -1.0 / 20.0;

            Q[INDEX_2D(1, 0)] = -20.0 / 33.0;
            Q[INDEX_2D(1, 1)] = -35.0 / 132.0;
            Q[INDEX_2D(1, 2)] = 34.0 / 33.0;
            Q[INDEX_2D(1, 3)] = -7.0 / 33.0;
            Q[INDEX_2D(1, 4)] = 2.0 / 33.0;
            Q[INDEX_2D(1, 5)] = -1.0 / 132.0;

            Q[INDEX_2D(2, 0)] = -1.0 / 36.0;
            Q[INDEX_2D(2, 1)] = -14.0 / 18.0;
            Q[INDEX_2D(3, 2)] = 0.0;
            Q[INDEX_2D(2, 3)] = 14.0 / 18.0;
            Q[INDEX_2D(2, 4)] = 1.0 / 36.0;

            ib = 3;
        } break;

        case CFD_P1_O4_CLOSE: {
            if (nghosts < 3) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "3! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t1 = 1.0 / 72.0;
            Q[INDEX_2D(ib, ib - 3)] = -t1;
            Q[INDEX_2D(ib, ib - 2)] = 10.0 * t1;
            Q[INDEX_2D(ib, ib - 1)] = -53.0 * t1;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 53.0 * t1;
            Q[INDEX_2D(ib, ib + 2)] = -10.0 * t1;
            Q[INDEX_2D(ib, ib + 3)] = t1;
            ib += 1;
        } break;

        case CFD_P1_O6_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t2 = 1.0 / 300.0;
            Q[INDEX_2D(ib, ib - 4)] = t2;
            Q[INDEX_2D(ib, ib - 3)] = -11.0 * t2;
            Q[INDEX_2D(ib, ib - 2)] = 59.0 * t2;
            Q[INDEX_2D(ib, ib - 1)] = -239.0 * t2;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 239.0 * t2;
            Q[INDEX_2D(ib, ib + 2)] = -59.0 * t2;
            Q[INDEX_2D(ib, ib + 3)] = 11.0 * t2;
            Q[INDEX_2D(ib, ib + 4)] = -t2;
            ib += 1;
        } break;

        case CFD_P1_O4_L4_CLOSE: {
            if (nghosts < 1) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "1! nghosts = " +
                    std::to_string(nghosts));
            }

            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t3 = 1.0 / 12.0;
            Q[INDEX_2D(ib, ib - 1)] = -3.0 * t3;
            Q[INDEX_2D(ib, ib)] = -10.0 * t3;
            Q[INDEX_2D(ib, ib + 1)] = 18.0 * t3;
            Q[INDEX_2D(ib, ib + 2)] = -6.0 * t3;
            Q[INDEX_2D(ib, ib + 3)] = t3;

            ib += 1;
        } break;

        case CFD_P1_O6_L6_CLOSE: {
            if (nghosts < 2) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "2! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            const double t4 = 1.0 / 60.0;
            P[INDEX_2D(ib, ib)] = 1.0;

            Q[INDEX_2D(ib, ib - 2)] = 2.0 * t4;
            Q[INDEX_2D(ib, ib - 1)] = -24.0 * t4;
            Q[INDEX_2D(ib, ib)] = -35.0 * t4;
            Q[INDEX_2D(ib, ib + 1)] = 80.0 * t4;
            Q[INDEX_2D(ib, ib + 2)] = -30.0 * t4;
            Q[INDEX_2D(ib, ib + 3)] = 8.0 * t4;
            Q[INDEX_2D(ib, ib + 4)] = -1.0 * t4;

            ib += 1;
        } break;

        case CFD_Q1_O6_ETA1_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at "
                    "least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            Q[INDEX_2D(ib, ib - 4)] = 0.0035978349;
            Q[INDEX_2D(ib, ib - 3)] = -0.038253676;
            Q[INDEX_2D(ib, ib - 2)] = 0.20036969;
            Q[INDEX_2D(ib, ib - 1)] = -0.80036969;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 0.80036969;
            Q[INDEX_2D(ib, ib + 2)] = -0.20036969;
            Q[INDEX_2D(ib, ib + 3)] = 0.038253676;
            Q[INDEX_2D(ib, ib + 4)] = -0.0035978349;
            ib += 1;
        } break;

            // NOTE: in original initcfd.c file from David Neilsen, this was
            // repeated in the if statement, but in an elif, so it's unreachable
            // anyway since this value is handled in the same way above case
            // CFD_P1_O4_L4_CLOSE: ...

        default:
            throw std::invalid_argument(
                "Unknown derivative type for initializing CFD matrices! "
                "dtype=" +
                std::to_string(dtype));
            break;
    }
    // update xib
    *xib = ib;
}

void buildMatrixRight(double *P, double *Q, int *xie, const DerType dtype,
                      const int nghosts, const int n) {
    int ie = n - 1;

    switch (dtype) {
        case CFD_DRCHLT_ORDER_4: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 3.0;

            Q[INDEX_2D(n - 1, n - 1)] = 17.0 / 6.0;
            Q[INDEX_2D(n - 1, n - 2)] = -3.0 / 2.0;
            Q[INDEX_2D(n - 1, n - 3)] = -3.0 / 2.0;
            Q[INDEX_2D(n - 1, n - 4)] = 1.0 / 6.0;
            ie = n - 2;
        } break;

        case CFD_DRCHLT_ORDER_6: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 5.0;

            P[INDEX_2D(n - 2, n - 1)] = 2.0 / 11.0;
            P[INDEX_2D(n - 2, n - 2)] = 1.0;
            P[INDEX_2D(n - 2, n - 3)] = 2.0 / 11.0;

            Q[INDEX_2D(n - 1, n - 1)] = 197.0 / 60.0;
            Q[INDEX_2D(n - 1, n - 2)] = 5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 3)] = -5.0;
            Q[INDEX_2D(n - 1, n - 4)] = 5.0 / 3.0;
            Q[INDEX_2D(n - 1, n - 5)] = -5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 6)] = 1.0 / 20.0;

            Q[INDEX_2D(n - 2, n - 1)] = 20.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 2)] = 35.0 / 132.0;
            Q[INDEX_2D(n - 2, n - 3)] = -34.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 4)] = 7.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 5)] = -2.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 6)] = 1.0 / 132.0;
            ie = n - 3;
        } break;

        case CFD_DRCHLT_Q6: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 5.0;

            P[INDEX_2D(n - 2, n - 1)] = 2.0 / 11.0;
            P[INDEX_2D(n - 2, n - 2)] = 1.0;
            P[INDEX_2D(n - 2, n - 3)] = 2.0 / 11.0;

            P[INDEX_2D(n - 3, n - 2)] = 1.0 / 3.0;
            P[INDEX_2D(n - 3, n - 3)] = 1.0;
            P[INDEX_2D(n - 3, n - 4)] = 1.0 / 3.0;

            Q[INDEX_2D(n - 1, n - 1)] = 197.0 / 60.0;
            Q[INDEX_2D(n - 1, n - 2)] = 5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 3)] = -5.0;
            Q[INDEX_2D(n - 1, n - 4)] = 5.0 / 3.0;
            Q[INDEX_2D(n - 1, n - 5)] = -5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 6)] = 1.0 / 20.0;

            Q[INDEX_2D(n - 2, n - 1)] = 20.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 2)] = 35.0 / 132.0;
            Q[INDEX_2D(n - 2, n - 3)] = -34.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 4)] = 7.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 5)] = -2.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 6)] = 1.0 / 132.0;

            Q[INDEX_2D(n - 3, n - 1)] = 1.0 / 36.0;
            Q[INDEX_2D(n - 3, n - 2)] = 14.0 / 18.0;
            Q[INDEX_2D(n - 3, n - 3)] = 0.0;
            Q[INDEX_2D(n - 3, n - 4)] = -14.0 / 18.0;
            Q[INDEX_2D(n - 3, n - 5)] = -1.0 / 36.0;

            ie = n - 4;
        } break;

        case CFD_P1_O4_CLOSE: {
            if (nghosts < 3) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "3! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t1 = 1.0 / 72.0;
            Q[INDEX_2D(ie, ie - 3)] = -t1;
            Q[INDEX_2D(ie, ie - 2)] = 10.0 * t1;
            Q[INDEX_2D(ie, ie - 1)] = -53.0 * t1;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 53.0 * t1;
            Q[INDEX_2D(ie, ie + 2)] = -10.0 * t1;
            Q[INDEX_2D(ie, ie + 3)] = t1;
            ie -= 1;
        } break;

        case CFD_P1_O6_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t2 = 1.0 / 300.0;
            Q[INDEX_2D(ie, ie - 4)] = t2;
            Q[INDEX_2D(ie, ie - 3)] = -11.0 * t2;
            Q[INDEX_2D(ie, ie - 2)] = 59.0 * t2;
            Q[INDEX_2D(ie, ie - 1)] = -239.0 * t2;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 239.0 * t2;
            Q[INDEX_2D(ie, ie + 2)] = -59.0 * t2;
            Q[INDEX_2D(ie, ie + 3)] = 11.0 * t2;
            Q[INDEX_2D(ie, ie + 4)] = -t2;
            ie -= 1;
        } break;

        case CFD_P1_O4_L4_CLOSE: {
            if (nghosts < 1) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "1! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t3 = 1.0 / 12.0;
            Q[INDEX_2D(ie, ie + 1)] = 3.0 * t3;
            Q[INDEX_2D(ie, ie)] = 10.0 * t3;
            Q[INDEX_2D(ie, ie - 1)] = -18.0 * t3;
            Q[INDEX_2D(ie, ie - 2)] = 6.0 * t3;
            Q[INDEX_2D(ie, ie - 3)] = -t3;

            ie -= 1;
        } break;

        case CFD_P1_O6_L6_CLOSE: {
            if (nghosts < 2) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "2! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            const double t4 = 1.0 / 60.0;
            P[INDEX_2D(ie, ie)] = 1.0;

            Q[INDEX_2D(ie, ie + 2)] = -2.0 * t4;
            Q[INDEX_2D(ie, ie + 1)] = 24.0 * t4;
            Q[INDEX_2D(ie, ie)] = 35.0 * t4;
            Q[INDEX_2D(ie, ie - 1)] = -80.0 * t4;
            Q[INDEX_2D(ie, ie - 2)] = 30.0 * t4;
            Q[INDEX_2D(ie, ie - 3)] = -8.0 * t4;
            Q[INDEX_2D(ie, ie - 4)] = 1.0 * t4;

            ie -= 1;
        } break;

        case CFD_Q1_O6_ETA1_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at "
                    "least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            Q[INDEX_2D(ie, ie - 4)] = 0.0035978349;
            Q[INDEX_2D(ie, ie - 3)] = -0.038253676;
            Q[INDEX_2D(ie, ie - 2)] = 0.20036969;
            Q[INDEX_2D(ie, ie - 1)] = -0.80036969;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 0.80036969;
            Q[INDEX_2D(ie, ie + 2)] = -0.20036969;
            Q[INDEX_2D(ie, ie + 3)] = 0.038253676;
            Q[INDEX_2D(ie, ie + 4)] = -0.0035978349;
            ie -= 1;
        } break;

        default:
            break;
    }
    // update xib
    *xie = ie;
}

void buildMatrixLeft2nd(double *P, double *Q, int *xib, const DerType2nd dtype,
                        const int nghosts, const int n) {
    int ib = 0;

    switch (dtype) {
        case CFD2ND_DRCHLT_ORDER_4: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 10.0;

            Q[INDEX_2D(0, 0)] = 145.0 / 12.0;
            Q[INDEX_2D(0, 1)] = -76.0 / 3.0;
            Q[INDEX_2D(0, 2)] = 29.0 / 2.0;
            Q[INDEX_2D(0, 3)] = -4.0 / 3.0;
            Q[INDEX_2D(0, 4)] = 1.0 / 12.0;
            ib = 1;
        } break;

        case CFD2ND_DRCHLT_ORDER_6: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 126.0 / 11.0;

            P[INDEX_2D(1, 0)] = 11.0 / 128.0;
            P[INDEX_2D(1, 1)] = 1.0;
            P[INDEX_2D(1, 2)] = 11.0 / 128.0;

            Q[INDEX_2D(0, 0)] = 2077.0 / 157.0;
            Q[INDEX_2D(0, 1)] = -2943.0 / 110.0;
            Q[INDEX_2D(0, 2)] = 573.0 / 44.0;
            Q[INDEX_2D(0, 3)] = 167.0 / 99.0;
            Q[INDEX_2D(0, 4)] = -18.0 / 11.0;
            Q[INDEX_2D(0, 5)] = 57.0 / 110.0;
            Q[INDEX_2D(0, 6)] = -131.0 / 1980.0;

            Q[INDEX_2D(1, 0)] = 585.0 / 512.0;
            Q[INDEX_2D(1, 1)] = -141.0 / 64.0;
            Q[INDEX_2D(1, 2)] = 459.0 / 512.0;
            Q[INDEX_2D(1, 3)] = 9.0 / 32.0;
            Q[INDEX_2D(1, 4)] = -81.0 / 512.0;
            Q[INDEX_2D(1, 5)] = 3.0 / 64.0;
            Q[INDEX_2D(1, 6)] = -3.0 / 512.0;
            ib = 2;
        } break;

        case CFD2ND_DRCHLT_Q6: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 126.0 / 11.0;

            P[INDEX_2D(1, 0)] = 11.0 / 128.0;
            P[INDEX_2D(1, 1)] = 1.0;
            P[INDEX_2D(1, 2)] = 11.0 / 128.0;

            P[INDEX_2D(2, 1)] = 2.0 / 11.0;
            P[INDEX_2D(2, 2)] = 1.0;
            P[INDEX_2D(2, 3)] = 2.0 / 11.0;

            Q[INDEX_2D(0, 0)] = 2077.0 / 157.0;
            Q[INDEX_2D(0, 1)] = -2943.0 / 110.0;
            Q[INDEX_2D(0, 2)] = 573.0 / 44.0;
            Q[INDEX_2D(0, 3)] = 167.0 / 99.0;
            Q[INDEX_2D(0, 4)] = -18.0 / 11.0;
            Q[INDEX_2D(0, 5)] = 57.0 / 110.0;
            Q[INDEX_2D(0, 6)] = -131.0 / 1980.0;

            Q[INDEX_2D(1, 0)] = 585.0 / 512.0;
            Q[INDEX_2D(1, 1)] = -141.0 / 64.0;
            Q[INDEX_2D(1, 2)] = 459.0 / 512.0;
            Q[INDEX_2D(1, 3)] = 9.0 / 32.0;
            Q[INDEX_2D(1, 4)] = -81.0 / 512.0;
            Q[INDEX_2D(1, 5)] = 3.0 / 64.0;
            Q[INDEX_2D(1, 6)] = -3.0 / 512.0;

            Q[INDEX_2D(2, 0)] = 3.0 / 44.0;
            Q[INDEX_2D(2, 1)] = 48.0 / 44.0;
            Q[INDEX_2D(2, 2)] = -102.0 / 44.0;
            Q[INDEX_2D(2, 3)] = 48.0 / 44.0;
            Q[INDEX_2D(2, 4)] = 3.0 / 44.0;

            ib = 3;
        } break;

        case CFD2ND_P2_O4_CLOSE: {
            if (nghosts < 3) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "3! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t1 = 1.0 / 144.0;
            Q[INDEX_2D(ib, ib - 3)] = t1;
            Q[INDEX_2D(ib, ib - 2)] = -18.0 * t1;
            Q[INDEX_2D(ib, ib - 1)] = 207.0 * t1;
            Q[INDEX_2D(ib, ib)] = -380.0 * t1;
            Q[INDEX_2D(ib, ib + 1)] = 207.0 * t1;
            Q[INDEX_2D(ib, ib + 2)] = -18.0 * t1;
            Q[INDEX_2D(ib, ib + 3)] = t1;
            ib += 1;
        } break;

        case CFD2ND_P2_O6_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t1 = 1.0 / 675.0;
            Q[INDEX_2D(ib, ib - 4)] = -t1;
            Q[INDEX_2D(ib, ib - 3)] = 0.5 * 31.0 * t1;
            Q[INDEX_2D(ib, ib - 2)] = -0.25 * 517.0 * t1;
            Q[INDEX_2D(ib, ib - 1)] = 0.5 * 2137.0 * t1;
            Q[INDEX_2D(ib, ib)] = -0.5 * 3815.0 * t1;
            Q[INDEX_2D(ib, ib + 1)] = 0.5 * 2137.0 * t1;
            Q[INDEX_2D(ib, ib + 2)] = -0.25 * 517.0 * t1;
            Q[INDEX_2D(ib, ib + 3)] = 0.5 * 31.0 * t1;
            Q[INDEX_2D(ib, ib + 4)] = -t1;
            ib += 1;
        } break;

        case CFD2ND_P2_O4_L4_CLOSE: {
            if (nghosts < 1) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "1! nghosts = " +
                    std::to_string(nghosts));
            }
            // FIXME: This needs to be coded and fixed!

            throw dendro_cfd::CFDNotImplemented(
                "2nd Order P2 L4 Close is not yet implemented!");
        } break;

        case CFD2ND_P2_O6_L6_CLOSE: {
            if (nghosts < 2) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "2! nghosts = " +
                    std::to_string(nghosts));
            }

            throw dendro_cfd::CFDNotImplemented(
                "2nd Order P2 L4 Close is not yet implemented!");

        } break;

        case CFD2ND_Q2_O6_ETA1_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at "
                    "least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            Q[INDEX_2D(ib, ib - 4)] = -0.0018142695;
            Q[INDEX_2D(ib, ib - 3)] = 0.025625267;
            Q[INDEX_2D(ib, ib - 2)] = -0.20079955;
            Q[INDEX_2D(ib, ib - 1)] = 1.6015991;
            Q[INDEX_2D(ib, ib)] = -2.8492211;
            Q[INDEX_2D(ib, ib + 1)] = 1.6015991;
            Q[INDEX_2D(ib, ib + 2)] = -0.20079955;
            Q[INDEX_2D(ib, ib + 3)] = 0.025625267;
            Q[INDEX_2D(ib, ib + 4)] = -0.0018142695;
            ib += 1;
        } break;

            // NOTE: in original initcfd.c file from David Neilsen, this was
            // repeated in the if statement, but in an elif, so it's unreachable
            // anyway since this value is handled in the same way above case
            // CFD_P1_O4_L4_CLOSE: ...

        default:
            throw std::invalid_argument(
                "Unknown derivative type for initializing CFD matrices! "
                "dtype=" +
                std::to_string(dtype));
            break;
    }
    // update xib
    *xib = ib;
}

void buildMatrixRight2nd(double *P, double *Q, int *xie, const DerType2nd dtype,
                         const int nghosts, const int n) {
    int ie = n - 1;

    switch (dtype) {
        case CFD2ND_DRCHLT_ORDER_4: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 10.0;

            Q[INDEX_2D(n - 1, n - 1)] = 145.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 2)] = -76.0 / 3.0;
            Q[INDEX_2D(n - 1, n - 3)] = 29.0 / 2.0;
            Q[INDEX_2D(n - 1, n - 4)] = -4.0 / 3.0;
            Q[INDEX_2D(n - 1, n - 5)] = 1.0 / 12.0;
            ie = n - 2;
        } break;

        case CFD2ND_DRCHLT_ORDER_6: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 126.0 / 11.0;

            P[INDEX_2D(n - 2, n - 1)] = 11.0 / 128.0;
            P[INDEX_2D(n - 2, n - 2)] = 1.0;
            P[INDEX_2D(n - 2, n - 3)] = 11.0 / 128.0;

            Q[INDEX_2D(n - 1, n - 1)] = 2077.0 / 157.0;
            Q[INDEX_2D(n - 1, n - 2)] = -2943.0 / 110.0;
            Q[INDEX_2D(n - 1, n - 3)] = 573.0 / 44.0;
            Q[INDEX_2D(n - 1, n - 4)] = 167.0 / 99.0;
            Q[INDEX_2D(n - 1, n - 5)] = -18.0 / 11.0;
            Q[INDEX_2D(n - 1, n - 6)] = 57.0 / 110.0;
            Q[INDEX_2D(n - 1, n - 7)] = -131.0 / 1980.0;

            Q[INDEX_2D(n - 2, n - 1)] = 585.0 / 512.0;
            Q[INDEX_2D(n - 2, n - 2)] = -141.0 / 64.0;
            Q[INDEX_2D(n - 2, n - 3)] = 459.0 / 512.0;
            Q[INDEX_2D(n - 2, n - 4)] = 9.0 / 32.0;
            Q[INDEX_2D(n - 2, n - 5)] = -81.0 / 512.0;
            Q[INDEX_2D(n - 2, n - 6)] = 3.0 / 64.0;
            Q[INDEX_2D(n - 2, n - 7)] = -3.0 / 512.0;
            ie = n - 3;
        } break;

        case CFD2ND_DRCHLT_Q6: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 126.0 / 11.0;

            P[INDEX_2D(n - 2, n - 1)] = 11.0 / 128.0;
            P[INDEX_2D(n - 2, n - 2)] = 1.0;
            P[INDEX_2D(n - 2, n - 3)] = 11.0 / 128.0;

            P[INDEX_2D(n - 3, n - 2)] = 2.0 / 11.0;
            P[INDEX_2D(n - 3, n - 3)] = 1.0;
            P[INDEX_2D(n - 3, n - 4)] = 2.0 / 11.0;

            Q[INDEX_2D(n - 1, n - 1)] = 2077.0 / 157.0;
            Q[INDEX_2D(n - 1, n - 2)] = -2943.0 / 110.0;
            Q[INDEX_2D(n - 1, n - 3)] = 573.0 / 44.0;
            Q[INDEX_2D(n - 1, n - 4)] = 167.0 / 99.0;
            Q[INDEX_2D(n - 1, n - 5)] = -18.0 / 11.0;
            Q[INDEX_2D(n - 1, n - 6)] = 57.0 / 110.0;
            Q[INDEX_2D(n - 1, n - 7)] = -131.0 / 1980.0;

            Q[INDEX_2D(n - 2, n - 1)] = 585.0 / 512.0;
            Q[INDEX_2D(n - 2, n - 2)] = -141.0 / 64.0;
            Q[INDEX_2D(n - 2, n - 3)] = 459.0 / 512.0;
            Q[INDEX_2D(n - 2, n - 4)] = 9.0 / 32.0;
            Q[INDEX_2D(n - 2, n - 5)] = -81.0 / 512.0;
            Q[INDEX_2D(n - 2, n - 6)] = 3.0 / 64.0;
            Q[INDEX_2D(n - 2, n - 7)] = -3.0 / 512.0;

            Q[INDEX_2D(n - 3, n - 1)] = 3.0 / 44.0;
            Q[INDEX_2D(n - 3, n - 2)] = 48.0 / 44.0;
            Q[INDEX_2D(n - 3, n - 3)] = -102.0 / 44.0;
            Q[INDEX_2D(n - 3, n - 4)] = 48.0 / 44.0;
            Q[INDEX_2D(n - 3, n - 5)] = 3.0 / 44.0;

            ie = n - 4;
        } break;

        case CFD2ND_P2_O4_CLOSE: {
            if (nghosts < 3) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "3! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t1 = 1.0 / 144.0;
            Q[INDEX_2D(ie, ie - 3)] = t1;
            Q[INDEX_2D(ie, ie - 2)] = -18.0 * t1;
            Q[INDEX_2D(ie, ie - 1)] = 207.0 * t1;
            Q[INDEX_2D(ie, ie)] = -380.0 * t1;
            Q[INDEX_2D(ie, ie + 1)] = 207.0 * t1;
            Q[INDEX_2D(ie, ie + 2)] = -18.0 * t1;
            Q[INDEX_2D(ie, ie + 3)] = t1;
            ie -= 1;
        } break;

        case CFD2ND_P2_O6_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t1 = 1.0 / 675.0;
            Q[INDEX_2D(ie, ie - 4)] = -t1;
            Q[INDEX_2D(ie, ie - 3)] = 0.5 * 31.0 * t1;
            Q[INDEX_2D(ie, ie - 2)] = -0.25 * 517.0 * t1;
            Q[INDEX_2D(ie, ie - 1)] = 0.5 * 2137.0 * t1;
            Q[INDEX_2D(ie, ie)] = -0.5 * 3815.0 * t1;
            Q[INDEX_2D(ie, ie + 1)] = 0.5 * 2137.0 * t1;
            Q[INDEX_2D(ie, ie + 2)] = -0.25 * 517.0 * t1;
            Q[INDEX_2D(ie, ie + 3)] = 0.5 * 31.0 * t1;
            Q[INDEX_2D(ie, ie + 4)] = -t1;
            ie -= 1;
        } break;

        case CFD2ND_P2_O4_L4_CLOSE: {
            if (nghosts < 1) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "1! nghosts = " +
                    std::to_string(nghosts));
            }

            // FIXME
            throw dendro_cfd::CFDNotImplemented(
                "Second order Right Side P2_O4_L4_CLOSE not implemented!");

        } break;

        case CFD_P1_O6_L6_CLOSE: {
            if (nghosts < 2) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "2! nghosts = " +
                    std::to_string(nghosts));
            }
            // FIXME
            throw dendro_cfd::CFDNotImplemented(
                "Second order Right Side CFD_P1_O6_L6_CLOSE not implemented!");
        } break;

        case CFD2ND_Q2_O6_ETA1_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at "
                    "least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            Q[INDEX_2D(ie, ie - 4)] = -0.0018142695;
            Q[INDEX_2D(ie, ie - 3)] = 0.025625267;
            Q[INDEX_2D(ie, ie - 2)] = -0.20079955;
            Q[INDEX_2D(ie, ie - 1)] = 1.6015991;
            Q[INDEX_2D(ie, ie)] = -2.8492211;
            Q[INDEX_2D(ie, ie + 1)] = 1.6015991;
            Q[INDEX_2D(ie, ie + 2)] = -0.20079955;
            Q[INDEX_2D(ie, ie + 3)] = 0.025625267;
            Q[INDEX_2D(ie, ie + 4)] = -0.0018142695;
            ie -= 1;
        } break;

        default:
            break;
    }
    // update xib
    *xie = ie;
}

void initializeKim4PQ(double *P, double *Q, int n) {
    const double alpha = 0.5862704032801503;
    const double beta = 9.549533555017055e-2;

    const double a1 = 0.6431406736919156;
    const double a2 = 0.2586011023495066;
    const double a3 = 7.140953479797375e-3;

    const double y00 = 0.0;
    const double y10 = 8.360703307833438e-2;
    const double y20 = 3.250008295108466e-2;
    const double y01 = 5.912678614078549;
    const double y11 = 0.0;
    const double y21 = 0.3998040493524358;
    const double y02 = 3.775623951744012;
    const double y12 = 2.058102869495757;
    const double y22 = 0.0;
    const double y03 = 0.0;
    const double y13 = 0.9704052014790193;
    const double y23 = 0.7719261277615860;
    const double y04 = 0.0;
    const double y14 = 0.0;
    const double y24 = 0.1626635931256900;

    const double b10 = -0.3177447290722621;
    const double b20 = -0.1219006056449124;
    const double b01 = -3.456878182643609;
    const double b21 = -0.6301651351188667;
    const double b02 = 5.839043358834730;
    const double b12 = -2.807631929593225e-2;
    const double b03 = 1.015886726041007;
    const double b13 = 1.593461635747659;
    const double b23 = 0.6521195063966084;
    const double b04 = -0.2246526470654333;
    const double b14 = 0.2533027046976367;
    const double b24 = 0.3938843551210350;
    const double b05 = 8.564940889936562e-2;
    const double b15 = -3.619652460174756e-2;
    const double b25 = 1.904944407973912e-2;
    const double b06 = -1.836710059356763e-2;
    const double b16 = 4.080281419108407e-3;
    const double b26 = -1.027260523947668e-3;

    const double b00 = -(b01 + b02 + b03 + b04 + b05 + b06);
    const double b11 = -(b10 + b12 + b13 + b14 + b15 + b16);
    const double b22 = -(b20 + b21 + b23 + b24 + b25 + b26);

    const int nd = n * n;

    for (int i = 3; i < n - 3; i++) {
        P[INDEX_2D(i, i - 2)] = beta;
        P[INDEX_2D(i, i - 1)] = alpha;
        P[INDEX_2D(i, i)] = 1.0;
        P[INDEX_2D(i, i + 1)] = alpha;
        P[INDEX_2D(i, i + 2)] = beta;
    }

    P[INDEX_2D(0, 0)] = 1.0;
    P[INDEX_2D(0, 1)] = y01;
    P[INDEX_2D(0, 2)] = y02;

    P[INDEX_2D(1, 0)] = y10;
    P[INDEX_2D(1, 1)] = 1.0;
    P[INDEX_2D(1, 2)] = y12;
    P[INDEX_2D(1, 3)] = y13;

    P[INDEX_2D(2, 0)] = y20;
    P[INDEX_2D(2, 1)] = y21;
    P[INDEX_2D(2, 2)] = 1.0;
    P[INDEX_2D(2, 3)] = y23;
    P[INDEX_2D(2, 4)] = y24;

    P[INDEX_2D(n - 3, n - 5)] = y24;
    P[INDEX_2D(n - 3, n - 4)] = y23;
    P[INDEX_2D(n - 3, n - 3)] = 1.0;
    P[INDEX_2D(n - 3, n - 2)] = y21;
    P[INDEX_2D(n - 3, n - 1)] = y20;

    P[INDEX_2D(n - 2, n - 4)] = y13;
    P[INDEX_2D(n - 2, n - 3)] = y12;
    P[INDEX_2D(n - 2, n - 2)] = 1.0;
    P[INDEX_2D(n - 2, n - 1)] = y10;

    P[INDEX_2D(n - 1, n - 3)] = y02;
    P[INDEX_2D(n - 1, n - 2)] = y01;
    P[INDEX_2D(n - 1, n - 1)] = 1.0;

    for (int i = 3; i < n - 3; i++) {
        Q[INDEX_2D(i, i - 3)] = -a3;
        Q[INDEX_2D(i, i - 2)] = -a2;
        Q[INDEX_2D(i, i - 1)] = -a1;
        Q[INDEX_2D(i, i)] = 0.0;
        Q[INDEX_2D(i, i + 1)] = a1;
        Q[INDEX_2D(i, i + 2)] = a2;
        Q[INDEX_2D(i, i + 3)] = a3;
    }

    Q[INDEX_2D(0, 0)] = b00;
    Q[INDEX_2D(0, 1)] = b01;
    Q[INDEX_2D(0, 2)] = b02;
    Q[INDEX_2D(0, 3)] = b03;
    Q[INDEX_2D(0, 4)] = b04;
    Q[INDEX_2D(0, 5)] = b05;
    Q[INDEX_2D(0, 6)] = b06;

    Q[INDEX_2D(1, 0)] = b10;
    Q[INDEX_2D(1, 1)] = b11;
    Q[INDEX_2D(1, 2)] = b12;
    Q[INDEX_2D(1, 3)] = b13;
    Q[INDEX_2D(1, 4)] = b14;
    Q[INDEX_2D(1, 5)] = b15;
    Q[INDEX_2D(1, 6)] = b16;

    Q[INDEX_2D(2, 0)] = b20;
    Q[INDEX_2D(2, 1)] = b21;
    Q[INDEX_2D(2, 2)] = b22;
    Q[INDEX_2D(2, 3)] = b23;
    Q[INDEX_2D(2, 4)] = b24;
    Q[INDEX_2D(2, 5)] = b25;
    Q[INDEX_2D(2, 6)] = b26;

    Q[INDEX_2D(n - 3, n - 1)] = -b20;
    Q[INDEX_2D(n - 3, n - 2)] = -b21;
    Q[INDEX_2D(n - 3, n - 3)] = -b22;
    Q[INDEX_2D(n - 3, n - 4)] = -b23;
    Q[INDEX_2D(n - 3, n - 5)] = -b24;
    Q[INDEX_2D(n - 3, n - 6)] = -b25;
    Q[INDEX_2D(n - 3, n - 7)] = -b26;

    Q[INDEX_2D(n - 2, n - 1)] = -b10;
    Q[INDEX_2D(n - 2, n - 2)] = -b11;
    Q[INDEX_2D(n - 2, n - 3)] = -b12;
    Q[INDEX_2D(n - 2, n - 4)] = -b13;
    Q[INDEX_2D(n - 2, n - 5)] = -b14;
    Q[INDEX_2D(n - 2, n - 6)] = -b15;
    Q[INDEX_2D(n - 2, n - 7)] = -b16;

    Q[INDEX_2D(n - 1, n - 1)] = -b00;
    Q[INDEX_2D(n - 1, n - 2)] = -b01;
    Q[INDEX_2D(n - 1, n - 3)] = -b02;
    Q[INDEX_2D(n - 1, n - 4)] = -b03;
    Q[INDEX_2D(n - 1, n - 5)] = -b04;
    Q[INDEX_2D(n - 1, n - 6)] = -b05;
    Q[INDEX_2D(n - 1, n - 7)] = -b06;
}

void initializeKim6FilterPQ(double *P, double *Q, int n) {
    const double alphaF = 0.6651452077642562;
    const double betaF = 0.1669709584471488;
    const double aF1 = 8.558206326059179e-4;
    const double aF2 = -3.423282530423672e-4;
    const double aF3 = 5.705470884039454e-5;
    const double aF0 = -2.0 * (aF1 + aF2 + aF3);

    const double yF00 = 0.0;
    const double yF10 = 0.7311329755609861;
    const double yF20 = 0.1681680891936087;
    const double yF01 = 0.3412746505356879;
    const double yF11 = 0.0;
    const double yF21 = 0.6591595540319565;
    const double yF02 = 0.2351300295562464;
    const double yF12 = 0.6689728401317021;
    const double yF22 = 0.0;
    const double yF03 = 0.0;
    const double yF13 = 0.1959510121583215;
    const double yF23 = 0.6591595540319565;
    const double yF04 = 0.0;
    const double yF14 = 0.0;
    const double yF24 = 0.1681680891936087;

    const double bF20 = -2.81516723801634e-4;
    const double bF21 = 1.40758361900817e-3;
    const double bF23 = 2.81516723801634e-3;
    const double bF24 = -1.40758361900817e-3;
    const double bF25 = 2.81516723801634e-4;
    const double bF22 = -(bF20 + bF21 + bF23 + bF24 + bF25);

    const int nd = n * n;

    for (int i = 3; i < n - 3; i++) {
        P[INDEX_2D(i, i - 2)] = betaF;
        P[INDEX_2D(i, i - 1)] = alphaF;
        P[INDEX_2D(i, i)] = 1.0;
        P[INDEX_2D(i, i + 1)] = alphaF;
        P[INDEX_2D(i, i + 2)] = betaF;
    }

    P[INDEX_2D(0, 0)] = 1.0;
    P[INDEX_2D(0, 1)] = yF01;
    P[INDEX_2D(0, 2)] = yF02;

    P[INDEX_2D(1, 0)] = yF10;
    P[INDEX_2D(1, 1)] = 1.0;
    P[INDEX_2D(1, 2)] = yF12;
    P[INDEX_2D(1, 3)] = yF13;

    P[INDEX_2D(2, 0)] = yF20;
    P[INDEX_2D(2, 1)] = yF21;
    P[INDEX_2D(2, 2)] = 1.0;
    P[INDEX_2D(2, 3)] = yF23;
    P[INDEX_2D(2, 4)] = yF24;

    P[INDEX_2D(n - 3, n - 5)] = yF24;
    P[INDEX_2D(n - 3, n - 4)] = yF23;
    P[INDEX_2D(n - 3, n - 3)] = 1.0;
    P[INDEX_2D(n - 3, n - 2)] = yF21;
    P[INDEX_2D(n - 3, n - 1)] = yF20;

    P[INDEX_2D(n - 2, n - 4)] = yF13;
    P[INDEX_2D(n - 2, n - 3)] = yF12;
    P[INDEX_2D(n - 2, n - 2)] = 1.0;
    P[INDEX_2D(n - 2, n - 1)] = yF10;

    P[INDEX_2D(n - 1, n - 3)] = yF02;
    P[INDEX_2D(n - 1, n - 2)] = yF01;
    P[INDEX_2D(n - 1, n - 1)] = 1.0;

    for (int i = 0; i < nd; i++) {
        Q[i] = 0.0;
    }
    for (int i = 3; i < n - 3; i++) {
        Q[INDEX_2D(i, i - 3)] = aF3;
        Q[INDEX_2D(i, i - 2)] = aF2;
        Q[INDEX_2D(i, i - 1)] = aF1;
        Q[INDEX_2D(i, i)] = aF0;
        Q[INDEX_2D(i, i + 1)] = aF1;
        Q[INDEX_2D(i, i + 2)] = aF2;
        Q[INDEX_2D(i, i + 3)] = aF3;
    }

    Q[INDEX_2D(0, 0)] = 0.0;
    Q[INDEX_2D(0, 1)] = 0.0;
    Q[INDEX_2D(0, 2)] = 0.0;
    Q[INDEX_2D(0, 3)] = 0.0;

    Q[INDEX_2D(1, 0)] = 0.0;
    Q[INDEX_2D(1, 1)] = 0.0;
    Q[INDEX_2D(1, 2)] = 0.0;
    Q[INDEX_2D(1, 3)] = 0.0;
    Q[INDEX_2D(1, 4)] = 0.0;

    Q[INDEX_2D(2, 0)] = bF20;
    Q[INDEX_2D(2, 1)] = bF21;
    Q[INDEX_2D(2, 2)] = bF22;
    Q[INDEX_2D(2, 3)] = bF23;
    Q[INDEX_2D(2, 4)] = bF24;
    Q[INDEX_2D(2, 5)] = bF25;

    Q[INDEX_2D(n - 3, n - 6)] = bF25;
    Q[INDEX_2D(n - 3, n - 5)] = bF24;
    Q[INDEX_2D(n - 3, n - 4)] = bF23;
    Q[INDEX_2D(n - 3, n - 3)] = bF22;
    Q[INDEX_2D(n - 3, n - 2)] = bF21;
    Q[INDEX_2D(n - 3, n - 1)] = bF20;

    Q[INDEX_2D(n - 2, n - 5)] = 0.0;
    Q[INDEX_2D(n - 2, n - 4)] = 0.0;
    Q[INDEX_2D(n - 2, n - 3)] = 0.0;
    Q[INDEX_2D(n - 2, n - 2)] = 0.0;
    Q[INDEX_2D(n - 2, n - 1)] = 0.0;

    Q[INDEX_2D(n - 1, n - 4)] = 0.0;
    Q[INDEX_2D(n - 1, n - 3)] = 0.0;
    Q[INDEX_2D(n - 1, n - 2)] = 0.0;
    Q[INDEX_2D(n - 1, n - 1)] = 0.0;
}

void print_square_mat(double *m, const uint32_t n) {
    // assumes "col" order in memory
    // J is the row!
    for (uint16_t i = 0; i < n; i++) {
        printf("%3d : ", i);
        // I is the column!
        for (uint16_t j = 0; j < n; j++) {
            printf("%8.3f ", m[INDEX_2D(i, j)]);
        }
        printf("\n");
    }
}

void buildDerivExplicitRMatrix(double *R, const unsigned int padding,
                               const unsigned int n, const DerType deriv_type,
                               const bool is_left_edge,
                               const bool is_right_edge) {
    uint32_t curr_n = n;
    uint32_t i_start = 0;
    uint32_t i_end = n;
    uint32_t j_start = 0;
    uint32_t j_end = n;

    if (is_left_edge) {
        // initialize the "diagonal" in the padding to 1
        for (uint32_t ii = 0; ii < padding; ii++) {
            R[INDEX_2D(ii, ii)] = 1.0;
        }
        i_start += padding;
        j_start += padding;
        curr_n -= padding;
    }

    if (is_right_edge) {
        // initialize bottom "diagonal" in padding to 1 as well
        for (uint32_t ii = n - 1; ii >= n - padding; ii--) {
            R[INDEX_2D(ii, ii)] = 1.0;
        }
        i_end -= padding;
        j_end -= padding;
        curr_n -= padding;
    }

    double *tempR = nullptr;

    if (is_left_edge or is_right_edge) {
        // initialize tempR to be a "smaller" square matrix for use
        tempR = new double[curr_n * curr_n]();
    } else {
        // just use the same pointer value, then no need to adjust later even
        tempR = R;
    }

    // NOTE: the right edge/left edge on the following functions are always true
    // in this case
    if (deriv_type == EXPLCT_FD_O4) {
        buildDerivExplicit4thOrder(tempR, curr_n, true, true);
    } else if (deriv_type == EXPLCT_FD_O6) {
        buildDerivExplicit6thOrder(tempR, curr_n, true, true);
    } else if (deriv_type == EXPLCT_FD_O8) {
        buildDerivExplicit8thOrder(tempR, curr_n, true, true);
    } else {
        if (is_left_edge or is_right_edge) {
            delete[] tempR;
        }
        throw std::invalid_argument(
            "Explicit derivative type was invalid. deriv_type=" +
            std::to_string(deriv_type));
    }

    // copy the values back in
    // NOTE: the use of j and i assumes ROW-MAJOR order, but it will just copy a
    // square matrix in no matter what, so it's not a big issue
    if (is_left_edge or is_right_edge) {
        // then memcopy the "chunks" to where they go inside the matrix
        uint32_t temp_arr_i = 0;
        // iterate over the rows
        for (uint32_t jj = j_start; jj < j_end; jj++) {
            // ii will only go from empty rows we actually need to fill...
            // j will start at "j_start" and go until "j_end" where we need to
            // fill memory start index of our main array

            uint32_t temp_start = INDEX_N2D(0, temp_arr_i, curr_n);
            // uint32_t temp_end = INDEX_N2D(curr_n - 1, temp_arr_i, curr_n);

            std::copy_n(&tempR[temp_start], curr_n, &R[INDEX_2D(i_start, jj)]);

            // increment temp_arr "row" value
            temp_arr_i++;
        }
        // clear up our temporary arrays we don't need
        delete[] tempR;
    }
    // NOTE: tempR doesn't need to be deleted if it was not initialized,
    // so we don't need to delete it unless we're dealing with left/right edges

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nDERIV MATRIX R" << std::endl;
    print_square_mat(R, n);
#endif
}

void build2ndDerivExplicitRMatrix(double *R, const unsigned int padding,
                                  const unsigned int n,
                                  const DerType2nd deriv_type,
                                  const bool is_left_edge,
                                  const bool is_right_edge) {
    uint32_t curr_n = n;
    uint32_t i_start = 0;
    uint32_t i_end = n;
    uint32_t j_start = 0;
    uint32_t j_end = n;

    if (is_left_edge) {
        // initialize the "diagonal" in the padding to 1
        for (uint32_t ii = 0; ii < padding; ii++) {
            R[INDEX_2D(ii, ii)] = 1.0;
        }
        i_start += padding;
        j_start += padding;
        curr_n -= padding;
    }

    if (is_right_edge) {
        // initialize bottom "diagonal" in padding to 1 as well
        for (uint32_t ii = n - 1; ii >= n - padding; ii--) {
            R[INDEX_2D(ii, ii)] = 1.0;
        }
        i_end -= padding;
        j_end -= padding;
        curr_n -= padding;
    }

    double *tempR = nullptr;

    if (is_left_edge or is_right_edge) {
        // initialize tempR to be a "smaller" square matrix for use
        tempR = new double[curr_n * curr_n]();
    } else {
        // just use the same pointer value, then no need to adjust later even
        tempR = R;
    }

    // NOTE: the right edge/left edge on the following functions are always true
    // or else we don't fill it out properly!
    if (deriv_type == EXPLCT2ND_FD_O4) {
        build2ndDerivExplicit4thOrder(tempR, curr_n, true, true);
    } else if (deriv_type == EXPLCT2ND_FD_O6) {
        build2ndDerivExplicit6thOrder(tempR, curr_n, true, true);
    } else if (deriv_type == EXPLCT2ND_FD_O8) {
        build2ndDerivExplicit8thOrder(tempR, curr_n, true, true);
    } else {
        if (is_left_edge or is_right_edge) {
            delete[] tempR;
        }
        throw std::invalid_argument(
            "Explicit derivative type was invalid. deriv_type=" +
            std::to_string(deriv_type));
    }

    // copy the values back in
    // NOTE: the use of j and i assumes ROW-MAJOR order, but it will just copy a
    // square matrix in no matter what, so it's not a big issue
    if (is_left_edge or is_right_edge) {
        // then memcopy the "chunks" to where they go inside the matrix
        uint32_t temp_arr_i = 0;
        // iterate over the rows
        for (uint32_t jj = j_start; jj < j_end; jj++) {
            // ii will only go from empty rows we actually need to fill...
            // j will start at "j_start" and go until "j_end" where we need to
            // fill memory start index of our main array

            uint32_t temp_start = INDEX_N2D(0, temp_arr_i, curr_n);
            // uint32_t temp_end = INDEX_N2D(curr_n - 1, temp_arr_i, curr_n);

            std::copy_n(&tempR[temp_start], curr_n, &R[INDEX_2D(i_start, jj)]);

            // increment temp_arr "row" value
            temp_arr_i++;
        }
        // clear up our temporary arrays we don't need
        delete[] tempR;
    }
    // NOTE: tempR doesn't need to be deleted if it was not initialized,
    // so we don't need to delete it unless we're dealing with left/right edges

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\n2ND DERIV MATRIX R" << std::endl;
    print_square_mat(R, n);
#endif
}

void buildDerivExplicit4thOrder(double *R, const unsigned int n,
                                bool is_left_edge = false,
                                bool is_right_edge = false) {
    // this matrix assumes that the whole matrix should be filled
    // send in a smaller n for other values

    if (is_left_edge && is_right_edge) {
        if (n < 4) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 4th order deriv "
                "matrix "
                "for left and right edge to be set, n must be >= "
                "4, and it is currently " +
                std::to_string(n));
        }
    } else if (is_left_edge || is_right_edge) {
        if (n < 2) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 4th order deriv "
                "matrix "
                "for left or right edge to be set, n must be >= "
                "2, and it is currently " +
                std::to_string(n));
        }
    }

    uint32_t start = 0;
    // n is now an *INDEX* for the "row" we want to use, not related to n
    uint32_t end = n - 1;

    // always assuming that we're starting with row 0, even if it's a right edge
    if (is_left_edge) {
        // first available row
        R[INDEX_2D(start, start)] = -3.0 / 2.0;  // this is the point
        R[INDEX_2D(start, start + 1)] = 4.0 / 2.0;
        R[INDEX_2D(start, start + 2)] = -1.0 / 2.0;

        // second row
        R[INDEX_2D(start + 1, start)] = -1.0 / 2.0;
        // R[INDEX_2D(start + 1, start + 1)] = 0.0 / 2.0; // this is the point
        R[INDEX_2D(start + 1, start + 2)] = 1.0 / 2.0;

        start += 2;
    }

    if (is_right_edge) {
        // first available row
        R[INDEX_2D(end, end)] = 3.0 / 2.0;  // this is the point
        R[INDEX_2D(end, end - 1)] = -4.0 / 2.0;
        R[INDEX_2D(end, end - 2)] = 1.0 / 2.0;

        // second row
        R[INDEX_2D(end - 1, end)] = 1.0 / 2.0;
        // R[INDEX_2D(end - 1, end - 1)] = 0.0 / 2.0; // this is the point
        R[INDEX_2D(end - 1, end - 2)] = -1.0 / 2.0;

        end -= 2;
    }

    // have to include "end", so add 1
    for (unsigned int i = start; i < end + 1; i++) {
        R[INDEX_2D(i, i - 2)] = 1.0 / 12.0;
        R[INDEX_2D(i, i - 1)] = -8.0 / 12.0;
        R[INDEX_2D(i, i - 0)] = 0.0;
        R[INDEX_2D(i, i + 1)] = 8.0 / 12.0;
        R[INDEX_2D(i, i + 2)] = -1.0 / 12.0;
    }
}

void buildDerivExplicit6thOrder(double *R, const unsigned int n,
                                bool is_left_edge = false,
                                bool is_right_edge = false) {
    // this matrix assumes that the whole matrix should be filled
    // send in a smaller n for other values

    if (is_left_edge && is_right_edge) {
        if (n < 6) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order deriv "
                "matrix "
                "for left and right edge to be set, n must be >= "
                "6, and it is currently " +
                std::to_string(n));
        }
    } else if (is_left_edge || is_right_edge) {
        if (n < 3) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order deriv "
                "matrix "
                "for left or right edge to be set, n must be >= "
                "3, and it is currently " +
                std::to_string(n));
        }
    }

    uint32_t start = 0;
    // n is now an *INDEX* for the "row" we want to use, not related to n
    uint32_t end = n - 1;

    // always assuming that we're starting with row 0, even if it's a right edge
    if (is_left_edge) {
        // first available row
        R[INDEX_2D(start, start)] = -25.0 / 12.0;
        R[INDEX_2D(start, start + 1)] = 48.0 / 12.0;
        R[INDEX_2D(start, start + 2)] = -36.0 / 12.0;
        R[INDEX_2D(start, start + 3)] = 16.0 / 12.0;
        R[INDEX_2D(start, start + 4)] = -3.0 / 12.0;

        // second row
        R[INDEX_2D(start + 1, start)] = -3.0 / 12.0;
        R[INDEX_2D(start + 1, start + 1)] = -10.0 / 12.0;
        R[INDEX_2D(start + 1, start + 2)] = 18.0 / 12.0;
        R[INDEX_2D(start + 1, start + 3)] = -6.0 / 12.0;
        R[INDEX_2D(start + 1, start + 4)] = 1.0 / 12.0;

        // third row
        R[INDEX_2D(start + 2, start)] = 1.0 / 12.0;
        R[INDEX_2D(start + 2, start + 1)] = -8.0 / 12.0;
        // 2 is empty
        R[INDEX_2D(start + 2, start + 3)] = +8.0 / 12.0;
        R[INDEX_2D(start + 2, start + 4)] = -1.0 / 12.0;

        start += 3;
    }

    if (is_right_edge) {
        // first available row
        R[INDEX_2D(end, end)] = 25.0 / 12.0;
        R[INDEX_2D(end, end - 1)] = -48.0 / 12.0;
        R[INDEX_2D(end, end - 2)] = 36.0 / 12.0;
        R[INDEX_2D(end, end - 3)] = -16.0 / 12.0;
        R[INDEX_2D(end, end - 4)] = 3.0 / 12.0;

        // second row
        R[INDEX_2D(end - 1, end)] = 3.0 / 12.0;
        R[INDEX_2D(end - 1, end - 1)] = 10.0 / 12.0;
        R[INDEX_2D(end - 1, end - 2)] = -18.0 / 12.0;
        R[INDEX_2D(end - 1, end - 3)] = 6.0 / 12.0;
        R[INDEX_2D(end - 1, end - 4)] = -1.0 / 12.0;

        // third row
        R[INDEX_2D(end - 2, end)] = 1.0 / 12.0;
        R[INDEX_2D(end - 2, end - 1)] = -8.0 / 12.0;
        // 2 is empty
        R[INDEX_2D(end - 2, end - 3)] = 8.0 / 12.0;
        R[INDEX_2D(end - 2, end - 4)] = -1.0 / 12.0;

        end -= 3;
    }

    // have to include "end", so add 1
    for (unsigned int i = start; i < end + 1; i++) {
        R[INDEX_2D(i, i - 3)] = -1.0 / 60.0;
        R[INDEX_2D(i, i - 2)] = 9.0 / 60.0;
        R[INDEX_2D(i, i - 1)] = -45.0 / 60.0;
        R[INDEX_2D(i, i - 0)] = 0.0;
        R[INDEX_2D(i, i + 1)] = 45.0 / 60.0;
        R[INDEX_2D(i, i + 2)] = -9.0 / 60.0;
        R[INDEX_2D(i, i + 3)] = 1.0 / 60.0;
    }
}

void buildDerivExplicit8thOrder(double *R, const unsigned int n,
                                bool is_left_edge = false,
                                bool is_right_edge = false) {
    // this matrix assumes that the whole matrix should be filled
    // send in a smaller n for other values

    if (is_left_edge && is_right_edge) {
        if (n < 8) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 8th order deriv "
                "matrix "
                "for left and right edge to be set, n must be >= "
                "8, and it is currently " +
                std::to_string(n));
        }
    } else if (is_left_edge || is_right_edge) {
        if (n < 4) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 8th order deriv "
                "matrix "
                "for left or right edge to be set, n must be >= "
                "4, and it is currently " +
                std::to_string(n));
        }
    }

    uint32_t start = 0;
    // n is now an *INDEX* for the "row" we want to use, not related to n
    uint32_t end = n - 1;

    // always assuming that we're starting with row 0, even if it's a right edge
    if (is_left_edge) {
        // first available row
        R[INDEX_2D(start, start)] = -147.0 / 60.0;  // this is the point
        R[INDEX_2D(start, start + 1)] = 360.0 / 60.0;
        R[INDEX_2D(start, start + 2)] = -450.0 / 60.0;
        R[INDEX_2D(start, start + 3)] = 400.0 / 60.0;
        R[INDEX_2D(start, start + 4)] = -225.0 / 60.0;
        R[INDEX_2D(start, start + 5)] = 72.0 / 60.0;
        R[INDEX_2D(start, start + 6)] = -10.0 / 60.0;

        // second row
        R[INDEX_2D(start + 1, start)] = -10.0 / 60.0;
        R[INDEX_2D(start + 1, start + 1)] = -77.0 / 60.0;  // this is the point
        R[INDEX_2D(start + 1, start + 2)] = 150.0 / 60.0;
        R[INDEX_2D(start + 1, start + 3)] = -100.0 / 60.0;
        R[INDEX_2D(start + 1, start + 4)] = 50.0 / 60.0;
        R[INDEX_2D(start + 1, start + 5)] = -15.0 / 60.0;
        R[INDEX_2D(start + 1, start + 6)] = 1.0 / 60.0;

        // third row
        R[INDEX_2D(start + 2, start)] = 2.0 / 60.0;
        R[INDEX_2D(start + 2, start + 1)] = -24.0 / 60.0;
        R[INDEX_2D(start + 2, start + 2)] = -35.0 / 60.0;  // this is the point
        R[INDEX_2D(start + 2, start + 3)] = 80.0 / 60.0;
        R[INDEX_2D(start + 2, start + 4)] = -30.0 / 60.0;
        R[INDEX_2D(start + 2, start + 5)] = 8.0 / 60.0;
        R[INDEX_2D(start + 2, start + 6)] = -1.0 / 60.0;

        // fourth row
        R[INDEX_2D(start + 3, start)] = -1.0 / 60.0;
        R[INDEX_2D(start + 3, start + 1)] = 9.0 / 60.0;
        R[INDEX_2D(start + 3, start + 2)] = -45.0 / 60.0;
        // R[INDEX_2D(start + 3, start + 3)] = 0.0 / 60.0; // this is the point
        R[INDEX_2D(start + 3, start + 4)] = 45.0 / 60.0;
        R[INDEX_2D(start + 3, start + 5)] = -9.0 / 60.0;
        R[INDEX_2D(start + 3, start + 6)] = 1.0 / 60.0;

        start += 4;
    }

    if (is_right_edge) {
        // first available row
        R[INDEX_2D(end, end)] = 147.0 / 12.0;  // this is the point
        R[INDEX_2D(end, end - 1)] = -360.0 / 12.0;
        R[INDEX_2D(end, end - 2)] = 450.0 / 12.0;
        R[INDEX_2D(end, end - 3)] = -400.0 / 12.0;
        R[INDEX_2D(end, end - 4)] = 225.0 / 12.0;
        R[INDEX_2D(end, end - 5)] = -72.0 / 12.0;
        R[INDEX_2D(end, end - 6)] = 10.0 / 12.0;

        // second row
        R[INDEX_2D(end - 1, end)] = 10.0 / 12.0;
        R[INDEX_2D(end - 1, end - 1)] = 77.0 / 12.0;  // this is the point
        R[INDEX_2D(end - 1, end - 2)] = -150.0 / 12.0;
        R[INDEX_2D(end - 1, end - 3)] = 100.0 / 12.0;
        R[INDEX_2D(end - 1, end - 4)] = -50.0 / 12.0;
        R[INDEX_2D(end - 1, end - 5)] = 15.0 / 12.0;
        R[INDEX_2D(end - 1, end - 6)] = -2.0 / 12.0;

        // third row
        R[INDEX_2D(end - 2, end)] = -2.0 / 12.0;
        R[INDEX_2D(end - 2, end - 1)] = 24.0 / 12.0;
        R[INDEX_2D(end - 2, end - 2)] = 35.0 / 12.0;  // this is the point
        R[INDEX_2D(end - 2, end - 3)] = -80.0 / 12.0;
        R[INDEX_2D(end - 2, end - 4)] = 30.0 / 12.0;
        R[INDEX_2D(end - 2, end - 5)] = -8.0 / 12.0;
        R[INDEX_2D(end - 2, end - 6)] = 1.0 / 12.0;

        // fourth row
        R[INDEX_2D(end - 3, end)] = 1.0 / 12.0;
        R[INDEX_2D(end - 3, end - 1)] = -9.0 / 12.0;
        R[INDEX_2D(end - 3, end - 2)] = 45.0 / 12.0;
        // R[INDEX_2D(end - 3, end - 3)] = 0.0 / 12.0; // this is the point
        R[INDEX_2D(end - 3, end - 4)] = -45.0 / 12.0;
        R[INDEX_2D(end - 3, end - 5)] = 9.0 / 12.0;
        R[INDEX_2D(end - 3, end - 6)] = -1.0 / 12.0;

        end -= 4;
    }

    // have to include "end", so add 1
    for (unsigned int i = start; i < end + 1; i++) {
        R[INDEX_2D(i, i - 4)] = 9.0 / 2520.0;
        R[INDEX_2D(i, i - 3)] = -96.0 / 2520.0;
        R[INDEX_2D(i, i - 2)] = 504.0 / 2520.0;
        R[INDEX_2D(i, i - 1)] = -2016.0 / 2520.0;
        R[INDEX_2D(i, i - 0)] = 0.0;
        R[INDEX_2D(i, i + 1)] = 2016.0 / 2520.0;
        R[INDEX_2D(i, i + 2)] = -504.0 / 2520.0;
        R[INDEX_2D(i, i + 3)] = 96.0 / 2520.0;
        R[INDEX_2D(i, i + 4)] = -9.0 / 2520.0;
    }
}

void buildKOExplicitFilter(double *R, const unsigned int n,
                           const unsigned int padding, const unsigned int order,
                           bool is_left_edge = false,
                           bool is_right_edge = false) {
    // assuming R is n x n
    //
    if (order == 6) {
        buildKOExplicit6thOrder(R, n, padding, is_left_edge, is_right_edge);
    } else if (order == 8) {
        buildKOExplicit8thOrder(R, n, padding, is_left_edge, is_right_edge);
    }
}

void buildKOExplicit6thOrder(double *R, const unsigned int n,
                             const unsigned int padding,
                             bool is_left_edge = false,
                             bool is_right_edge = false) {
    // fill the top padding rows as identity, assume matrix is 0'd

    if (padding < 3) {
        throw std::invalid_argument(
            "There isn't enough padding in the explicit 6th order KO matrix, "
            "padding must be >= 3, and it is currently " +
            std::to_string(padding));
    }

    if (is_left_edge && is_right_edge) {
        if (n - (padding * 2) < 6) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order KO matrix "
                "for left and right edge to be set, n - 2 * padding must be > "
                "6, and it is currently " +
                std::to_string(n - (2 * padding)));
        }
    } else if (is_left_edge || is_right_edge) {
        if (n - (padding * 2) < 3) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order KO matrix "
                "for left or right edge to be set, n - 2 * padding must be > "
                "3, and it is currently " +
                std::to_string(n - (2 * padding)));
        }
    }

    for (unsigned int i = 0; i < padding; i++) {
        R[INDEX_2D(i, i)] = 1.0;
        R[INDEX_2D(n - i - 1, n - i - 1)] = 1.0;
    }

    uint32_t start = padding;
    // n is now an *INDEX* for the "row" we want to use, not related to n
    uint32_t end = n - padding - 1;

    const double invSMR3 = 48.0 / (59.0 * 64.0);  // times dx!
    const double invSMR2 = 48.0 / (43.0 * 64.0);
    const double invSMR1 = 48.0 / (49.0 * 64.0);

    // always assuming that we're starting with row 0, even if it's a right edge
    if (is_left_edge) {
        // first available row
        R[INDEX_2D(start, start)] = 1 * invSMR3;
        R[INDEX_2D(start, start + 1)] = -3.0 * invSMR3;
        R[INDEX_2D(start, start + 2)] = 3.0 * invSMR3;
        R[INDEX_2D(start, start + 3)] = -1.0 * invSMR3;

        // second row
        R[INDEX_2D(start + 1, start)] = 1.0 * invSMR2;
        R[INDEX_2D(start + 1, start + 1)] = -6.0 * invSMR2;
        R[INDEX_2D(start + 1, start + 2)] = 12.0 * invSMR2;
        R[INDEX_2D(start + 1, start + 3)] = -10.0 * invSMR2;
        R[INDEX_2D(start + 1, start + 4)] = 3.0 * invSMR2;

        // third row
        R[INDEX_2D(start + 2, start)] = 1.0 * invSMR1;
        R[INDEX_2D(start + 2, start + 1)] = -6.0 * invSMR1;
        R[INDEX_2D(start + 2, start + 2)] = 15.0 * invSMR1;
        R[INDEX_2D(start + 2, start + 3)] = -19.0 * invSMR1;
        R[INDEX_2D(start + 2, start + 4)] = 12.0 * invSMR1;
        R[INDEX_2D(start + 2, start + 5)] = 3.0 * invSMR1;

        start += 3;
    }

    if (is_right_edge) {
        // first available row
        R[INDEX_2D(end, end)] = 1 * invSMR3;
        R[INDEX_2D(end, end - 1)] = -3.0 * invSMR3;
        R[INDEX_2D(end, end - 2)] = 3.0 * invSMR3;
        R[INDEX_2D(end, end - 3)] = -1.0 * invSMR3;

        // second row
        R[INDEX_2D(end - 1, end)] = 1.0 * invSMR2;
        R[INDEX_2D(end - 1, end - 1)] = -6.0 * invSMR2;
        R[INDEX_2D(end - 1, end - 2)] = 12.0 * invSMR2;
        R[INDEX_2D(end - 1, end - 3)] = -10.0 * invSMR2;
        R[INDEX_2D(end - 1, end - 4)] = 3.0 * invSMR2;

        // third row
        R[INDEX_2D(end - 2, end)] = 1.0 * invSMR1;
        R[INDEX_2D(end - 2, end - 1)] = -6.0 * invSMR1;
        R[INDEX_2D(end - 2, end - 2)] = 15.0 * invSMR1;
        R[INDEX_2D(end - 2, end - 3)] = -19.0 * invSMR1;
        R[INDEX_2D(end - 2, end - 4)] = 12.0 * invSMR1;
        R[INDEX_2D(end - 2, end - 5)] = 3.0 * invSMR1;

        end -= 3;
    }

    // have to include "end", so add 1
    for (unsigned int i = start; i < end + 1; i++) {
        R[INDEX_2D(i, i - 3)] = 1.0 / 64.0;
        R[INDEX_2D(i, i - 2)] = -6.0 / 64.0;
        R[INDEX_2D(i, i - 1)] = 15.0 / 64.0;
        R[INDEX_2D(i, i - 0)] = -20.0 / 64.0;
        R[INDEX_2D(i, i + 1)] = 15.0 / 64.0;
        R[INDEX_2D(i, i + 2)] = -6.0 / 64.0;
        R[INDEX_2D(i, i + 3)] = 1.0 / 64.0;
    }
}

void buildKOExplicit8thOrder(double *R, const unsigned int n,
                             const unsigned int padding,
                             bool is_left_edge = false,
                             bool is_right_edge = false) {
    // fill the top padding rows as identity, assume matrix is 0'd

    // FIXME: THIS MIGHT NOT BE COMPLETELY CORRECT, IT IS UNTESTED
    // NOTE: these have come directly from the derivs.cpp files

    if (padding < 4) {
        throw std::invalid_argument(
            "There isn't enough padding in the explicit 6th order KO matrix, "
            "padding must be >= 4, and it is currently " +
            std::to_string(padding));
    }

    if (is_left_edge && is_right_edge) {
        if (n - (padding * 2) < 8) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order KO matrix "
                "for left and right edge to be set, n - 2 * padding must be > "
                "8, and it is currently " +
                std::to_string(n - (2 * padding)));
        }
    } else if (is_left_edge || is_right_edge) {
        if (n - (padding * 2) < 4) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order KO matrix "
                "for left or right edge to be set, n - 2 * padding must be > "
                "4, and it is currently " +
                std::to_string(n - (2 * padding)));
        }
    }
    for (unsigned int i = 0; i < padding; i++) {
        R[INDEX_2D(i, i)] = 1.0;
        R[INDEX_2D(n - i - 1, n - i - 1)] = 1.0;
    }

    uint32_t start = padding;
    // n is now an *INDEX* for the "row" we want to use, not related to n
    uint32_t end = n - padding - 1;

    const double invSMR4 = 48.0 / (17.0 * 256.0);  // TIMES dx!
    const double invSMR3 = 48.0 / (59.0 * 256.0);
    const double invSMR2 = 48.0 / (43.0 * 256.0);
    const double invSMR1 = 48.0 / (49.0 * 256.0);

    // always assuming that we're starting with row 0, even if it's a right edge
    if (is_left_edge) {
        // first available row
        R[INDEX_2D(start, start)] = -1.0 * invSMR4;
        R[INDEX_2D(start, start + 1)] = 4.0 * invSMR4;
        R[INDEX_2D(start, start + 2)] = -6.0 * invSMR4;
        R[INDEX_2D(start, start + 3)] = 4.0 * invSMR4;
        R[INDEX_2D(start, start + 4)] = -1.0 * invSMR4;

        R[INDEX_2D(start + 1, start)] = 3.0 * invSMR3;
        R[INDEX_2D(start + 1, start + 1)] = -11.0 * invSMR3;
        R[INDEX_2D(start + 1, start + 2)] = 15.0 * invSMR3;
        R[INDEX_2D(start + 1, start + 3)] = -9.0 * invSMR3;
        R[INDEX_2D(start + 1, start + 4)] = -2.0 * invSMR3;
        // Why isn't this one used?
        // R[INDEX_2D(start + 1, start + 5)] = xxx * invSMR3;

        // second row
        R[INDEX_2D(start + 2, start)] = -3.0 * invSMR2;
        R[INDEX_2D(start + 2, start + 1)] = 9.0 * invSMR2;
        R[INDEX_2D(start + 2, start + 2)] = -8.0 * invSMR2;
        R[INDEX_2D(start + 2, start + 3)] = 3.0 * invSMR2;
        R[INDEX_2D(start + 2, start + 4)] = -1.0 * invSMR2;
        // why isn't this one used?
        // R[INDEX_2D(start + 2, start + 5)] = xxx * invSMR2;

        // third row
        R[INDEX_2D(start + 3, start)] = 1.0 * invSMR1;
        R[INDEX_2D(start + 3, start + 1)] = -1.0 * invSMR1;
        R[INDEX_2D(start + 3, start + 2)] = -6.0 * invSMR1;
        R[INDEX_2D(start + 3, start + 3)] = 15.0 * invSMR1;
        R[INDEX_2D(start + 3, start + 4)] = -14.0 * invSMR1;
        R[INDEX_2D(start + 3, start + 5)] = 6.0 * invSMR1;
        R[INDEX_2D(start + 3, start + 6)] = -1.0 * invSMR1;

        start += 4;
    }

    if (is_right_edge) {
        // first available row
        // first available row
        R[INDEX_2D(end, end)] = -1.0 * invSMR4;
        R[INDEX_2D(end, end - 1)] = 4.0 * invSMR4;
        R[INDEX_2D(end, end - 2)] = -6.0 * invSMR4;
        R[INDEX_2D(end, end - 3)] = 4.0 * invSMR4;
        R[INDEX_2D(end, end - 4)] = -1.0 * invSMR4;

        R[INDEX_2D(end - 1, end)] = 3.0 * invSMR3;
        R[INDEX_2D(end - 1, end - 1)] = -11.0 * invSMR3;
        R[INDEX_2D(end - 1, end - 2)] = 15.0 * invSMR3;
        R[INDEX_2D(end - 1, end - 3)] = -9.0 * invSMR3;
        R[INDEX_2D(end - 1, end - 4)] = -2.0 * invSMR3;
        // Why isn't this one used?
        // R[INDEX_2D(end - 1, end - 5)] = xxx * invSMR3;

        // second row
        R[INDEX_2D(end - 2, end)] = -3.0 * invSMR2;
        R[INDEX_2D(end - 2, end - 1)] = 9.0 * invSMR2;
        R[INDEX_2D(end - 2, end - 2)] = -8.0 * invSMR2;
        R[INDEX_2D(end - 2, end - 3)] = 3.0 * invSMR2;
        R[INDEX_2D(end - 2, end - 4)] = -1.0 * invSMR2;
        // why isn't this one used?
        // R[INDEX_2D(end - 2, end - 5)] = xxx * invSMR2;

        // third row
        R[INDEX_2D(end - 3, end)] = 1.0 * invSMR1;
        R[INDEX_2D(end - 3, end - 1)] = -1.0 * invSMR1;
        R[INDEX_2D(end - 3, end - 2)] = -6.0 * invSMR1;
        R[INDEX_2D(end - 3, end - 3)] = 15.0 * invSMR1;
        R[INDEX_2D(end - 3, end - 4)] = -14.0 * invSMR1;
        R[INDEX_2D(end - 3, end - 5)] = 6.0 * invSMR1;
        R[INDEX_2D(end - 3, end - 6)] = -1.0 * invSMR1;

        end -= 4;
    }

    // have to include "end", so add 1
    for (unsigned int i = start; i < end + 1; i++) {
        R[INDEX_2D(i, i - 4)] = -1.0 / 256.0;
        R[INDEX_2D(i, i - 3)] = 8.0 / 256.0;
        R[INDEX_2D(i, i - 2)] = -28.0 / 256.0;
        R[INDEX_2D(i, i - 1)] = 56.0 / 256.0;
        R[INDEX_2D(i, i - 0)] = -70.0 / 256.0;
        R[INDEX_2D(i, i + 1)] = 56.0 / 256.0;
        R[INDEX_2D(i, i + 2)] = -28.0 / 256.0;
        R[INDEX_2D(i, i + 3)] = 8.0 / 256.0;
        R[INDEX_2D(i, i + 4)] = -1.0 / 256.0;
    }
}

void build2ndDerivExplicit4thOrder(double *R, const unsigned int n,
                                   bool is_left_edge = false,
                                   bool is_right_edge = false) {
    // this matrix assumes that the whole matrix should be filled
    // send in a smaller n for other values

    if (is_left_edge && is_right_edge) {
        if (n < 4) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order 2nd deriv "
                "matrix "
                "for left and right edge to be set, n must be >= "
                "4, and it is currently " +
                std::to_string(n));
        }
    } else if (is_left_edge || is_right_edge) {
        if (n < 2) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order 2nd deriv "
                "matrix "
                "for left or right edge to be set, n must be >= "
                "2, and it is currently " +
                std::to_string(n));
        }
    }

    uint32_t start = 0;
    // n is now an *INDEX* for the "row" we want to use, not related to n
    uint32_t end = n - 1;

    // always assuming that we're starting with row 0, even if it's a right edge
    // NOTE: this is using the 644 stencil structure from derivs.cpp
    if (is_left_edge) {
        // first available row
        R[INDEX_2D(start, start)] = 2.0 / 1.0;  // POINT 0
        R[INDEX_2D(start, start + 1)] = -5.0 / 1.0;
        R[INDEX_2D(start, start + 2)] = 4.0 / 1.0;
        R[INDEX_2D(start, start + 3)] = -1.0 / 1.0;

        // second row
        R[INDEX_2D(start + 1, start)] = 1.0 / 1.0;
        R[INDEX_2D(start + 1, start + 1)] = -2.0 / 1.0;  // POINT 0
        R[INDEX_2D(start + 1, start + 2)] = 1.0 / 1.0;

        start += 2;
    }

    if (is_right_edge) {
        // first available row
        R[INDEX_2D(end, end)] = 2.0 / 1.0;  // POINT 0
        R[INDEX_2D(end, end - 1)] = -5.0 / 1.0;
        R[INDEX_2D(end, end - 2)] = 4.0 / 1.0;
        R[INDEX_2D(end, end - 3)] = -1.0 / 1.0;

        // second row
        R[INDEX_2D(end - 1, end)] = 1.0 / 1.0;
        R[INDEX_2D(end - 1, end - 1)] = -2.0 / 1.0;  // POINT 0
        R[INDEX_2D(end - 1, end - 2)] = 1.0 / 1.0;

        end -= 2;
    }

    // have to include "end", so add 1
    for (unsigned int i = start; i < end + 1; i++) {
        R[INDEX_2D(i, i - 2)] = -1.0 / 12.0;
        R[INDEX_2D(i, i - 1)] = 16.0 / 12.0;
        R[INDEX_2D(i, i - 0)] = -30.0 / 12.0;
        R[INDEX_2D(i, i + 1)] = 16.0 / 12.0;
        R[INDEX_2D(i, i + 2)] = -1.0 / 12.0;
    }
}

void build2ndDerivExplicit6thOrder(double *R, const unsigned int n,
                                   bool is_left_edge = false,
                                   bool is_right_edge = false) {
    // this matrix assumes that the whole matrix should be filled
    // send in a smaller n for other values

    if (is_left_edge && is_right_edge) {
        if (n < 6) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order 2nd deriv "
                "matrix "
                "for left and right edge to be set, n must be >= "
                "6, and it is currently " +
                std::to_string(n));
        }
    } else if (is_left_edge || is_right_edge) {
        if (n < 3) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 6th order 2nd deriv "
                "matrix "
                "for left or right edge to be set, n must be >= "
                "3, and it is currently " +
                std::to_string(n));
        }
    }

    uint32_t start = 0;
    // n is now an *INDEX* for the "row" we want to use, not related to n
    uint32_t end = n - 1;

    // always assuming that we're starting with row 0, even if it's a right edge
    // NOTE: this is using the 644 stencil structure from derivs.cpp
    if (is_left_edge) {
        // first available row
        R[INDEX_2D(start, start)] = 45.0 / 12.0;  // POINT 0
        R[INDEX_2D(start, start + 1)] = -154.0 / 12.0;
        R[INDEX_2D(start, start + 2)] = 214.0 / 12.0;
        R[INDEX_2D(start, start + 3)] = -156.0 / 12.0;
        R[INDEX_2D(start, start + 4)] = 61.0 / 12.0;
        R[INDEX_2D(start, start + 5)] = -10.0 / 12.0;

        // second row
        R[INDEX_2D(start + 1, start)] = 10.0 / 12.0;
        R[INDEX_2D(start + 1, start + 1)] = -15.0 / 12.0;  // POINT 0
        R[INDEX_2D(start + 1, start + 2)] = -4.0 / 12.0;
        R[INDEX_2D(start + 1, start + 3)] = 14.0 / 12.0;
        R[INDEX_2D(start + 1, start + 4)] = -6.0 / 12.0;
        R[INDEX_2D(start + 1, start + 5)] = 1.0 / 12.0;

        // third row
        R[INDEX_2D(start + 2, start)] = -1.0 / 12.0;
        R[INDEX_2D(start + 2, start + 1)] = 16.0 / 12.0;
        R[INDEX_2D(start + 2, start + 2)] = -30.0 / 12.0;  // POINT 0
        R[INDEX_2D(start + 2, start + 3)] = 16.0 / 12.0;
        R[INDEX_2D(start + 2, start + 4)] = -1.0 / 12.0;
        // 5 is empty

        start += 3;
    }

    if (is_right_edge) {
        // first available row
        R[INDEX_2D(end, end)] = 45.0 / 12.0;  // POINT 0
        R[INDEX_2D(end, end - 1)] = -154.0 / 12.0;
        R[INDEX_2D(end, end - 2)] = 214.0 / 12.0;
        R[INDEX_2D(end, end - 3)] = -156.0 / 12.0;
        R[INDEX_2D(end, end - 4)] = 61.0 / 12.0;
        R[INDEX_2D(end, end - 5)] = -10.0 / 12.0;

        // second row
        R[INDEX_2D(end - 1, end)] = 10.0 / 12.0;
        R[INDEX_2D(end - 1, end - 1)] = -15.0 / 12.0;  // POINT 0
        R[INDEX_2D(end - 1, end - 2)] = -4.0 / 12.0;
        R[INDEX_2D(end - 1, end - 3)] = 14.0 / 12.0;
        R[INDEX_2D(end - 1, end - 4)] = -6.0 / 12.0;
        R[INDEX_2D(end - 1, end - 5)] = 1.0 / 12.0;

        // third row
        R[INDEX_2D(end - 2, end)] = 1.0 / 12.0;
        R[INDEX_2D(end - 2, end - 1)] = 16.0 / 12.0;
        R[INDEX_2D(end - 2, end - 2)] = -30.0 / 12.0;  // POINT 0
        R[INDEX_2D(end - 2, end - 3)] = 16.0 / 12.0;
        R[INDEX_2D(end - 2, end - 4)] = -1.0 / 12.0;
        // 5 is empty

        end -= 3;
    }

    // have to include "end", so add 1
    for (unsigned int i = start; i < end + 1; i++) {
        R[INDEX_2D(i, i - 3)] = 2.0 / 180.0;
        R[INDEX_2D(i, i - 2)] = -27.0 / 180.0;
        R[INDEX_2D(i, i - 1)] = 270.0 / 180.0;
        R[INDEX_2D(i, i - 0)] = -490.0 / 180.0;
        R[INDEX_2D(i, i + 1)] = 270.0 / 180.0;
        R[INDEX_2D(i, i + 2)] = -27.0 / 180.0;
        R[INDEX_2D(i, i + 3)] = 2.0 / 180.0;
    }
}

void build2ndDerivExplicit8thOrder(double *R, const unsigned int n,
                                   bool is_left_edge = false,
                                   bool is_right_edge = false) {
    // this matrix assumes that the whole matrix should be filled
    // send in a smaller n for other values

    if (is_left_edge && is_right_edge) {
        if (n < 8) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 8th order 2nd deriv "
                "matrix "
                "for left and right edge to be set, n must be >= "
                "8, and it is currently " +
                std::to_string(n));
        }
    } else if (is_left_edge || is_right_edge) {
        if (n < 4) {
            throw std::invalid_argument(
                "There isn't enough space in the explicit 8th order 2nd deriv "
                "matrix "
                "for left or right edge to be set, n must be >= "
                "3, and it is currently " +
                std::to_string(n));
        }
    }

    uint32_t start = 0;
    // n is now an *INDEX* for the "row" we want to use, not related to n
    uint32_t end = n - 1;

    // always assuming that we're starting with row 0, even if it's a right edge

    // NOTE: this uses 8666's implementation since we can use it all in the
    // matrix!
    if (is_left_edge) {
        // NOTE: none of these points extend beyond the "middle" of the block
        // first available row
        R[INDEX_2D(start, start)] = 938.0 / 180.0;  // POINT 0
        R[INDEX_2D(start, start + 1)] = -4014.0 / 180.0;
        R[INDEX_2D(start, start + 2)] = 7911.0 / 180.0;
        R[INDEX_2D(start, start + 3)] = -9490.0 / 180.0;
        R[INDEX_2D(start, start + 4)] = 7380.0 / 180.0;
        R[INDEX_2D(start, start + 5)] = -3618.0 / 180.0;
        R[INDEX_2D(start, start + 6)] = 1019.0 / 180.0;
        R[INDEX_2D(start, start + 7)] = -126.0 / 180.0;
        // ^ a (totally) shifted second order stencil (FIXME)

        // second row
        R[INDEX_2D(start + 1, start)] = 126.0 / 180.0;
        R[INDEX_2D(start + 1, start + 1)] = -70.0 / 180.0;  // POINT 0
        R[INDEX_2D(start + 1, start + 2)] = -486.0 / 180.0;
        R[INDEX_2D(start + 1, start + 3)] = 855.0 / 180.0;
        R[INDEX_2D(start + 1, start + 4)] = -670.0 / 180.0;
        R[INDEX_2D(start + 1, start + 5)] = 324.0 / 180.0;
        R[INDEX_2D(start + 1, start + 6)] = -90.0 / 180.0;
        R[INDEX_2D(start + 1, start + 7)] = 11.0 / 180.0;

        // third row
        R[INDEX_2D(start + 1, start)] = -11.0 / 180.0;
        R[INDEX_2D(start + 1, start + 1)] = 214.0 / 180.0;
        R[INDEX_2D(start + 1, start + 2)] = -378.0 / 180.0;  // POINT 0
        R[INDEX_2D(start + 1, start + 3)] = 130.0 / 180.0;
        R[INDEX_2D(start + 1, start + 4)] = 85.0 / 180.0;
        R[INDEX_2D(start + 1, start + 5)] = -54.0 / 180.0;
        R[INDEX_2D(start + 1, start + 6)] = 16.0 / 180.0;
        R[INDEX_2D(start + 1, start + 7)] = -2.0 / 180.0;

        // fourth row
        R[INDEX_2D(start + 3, start)] = 2.0 / 180.0;
        R[INDEX_2D(start + 3, start + 1)] = -27.0 / 180.0;
        R[INDEX_2D(start + 3, start + 2)] = 270.0 / 180.0;  // POINT 0
        R[INDEX_2D(start + 3, start + 3)] = -490.0 / 180.0;
        R[INDEX_2D(start + 3, start + 4)] = 270.0 / 180.0;
        R[INDEX_2D(start + 3, start + 5)] = -27.0 / 180.0;
        R[INDEX_2D(start + 3, start + 6)] = 2.0 / 180.0;
        // 7th is empty

        start += 4;
    }

    if (is_right_edge) {
        // first available row
        R[INDEX_2D(end, end)] = 938.0 / 180.0;  // POINT 0
        R[INDEX_2D(end, end - 1)] = -4014.0 / 180.0;
        R[INDEX_2D(end, end - 2)] = 7911.0 / 180.0;
        R[INDEX_2D(end, end - 3)] = -9490.0 / 180.0;
        R[INDEX_2D(end, end - 4)] = 7380.0 / 180.0;
        R[INDEX_2D(end, end - 5)] = -3618.0 / 180.0;
        R[INDEX_2D(end, end - 6)] = 1019.0 / 180.0;
        R[INDEX_2D(end, end - 7)] = -126.0 / 180.0;

        // second row
        R[INDEX_2D(end - 1, end)] = 126.0 / 180.0;
        R[INDEX_2D(end - 1, end - 1)] = -70.0 / 180.0;  // POINT 0
        R[INDEX_2D(end - 1, end - 2)] = -486.0 / 180.0;
        R[INDEX_2D(end - 1, end - 3)] = 855.0 / 180.0;
        R[INDEX_2D(end - 1, end - 4)] = -670.0 / 180.0;
        R[INDEX_2D(end - 1, end - 5)] = 324.0 / 180.0;
        R[INDEX_2D(end - 1, end - 6)] = -90.0 / 180.0;
        R[INDEX_2D(end - 1, end - 7)] = 11.0 / 180.0;

        // third row
        R[INDEX_2D(end - 2, end)] = -11.0 / 180.0;
        R[INDEX_2D(end - 2, end - 1)] = 214.0 / 180.0;
        R[INDEX_2D(end - 2, end - 2)] = -378.0 / 180.0;  // POINT 0
        R[INDEX_2D(end - 2, end - 3)] = 130.0 / 180.0;
        R[INDEX_2D(end - 2, end - 4)] = 85.0 / 180.0;
        R[INDEX_2D(end - 2, end - 5)] = -54.0 / 180.0;
        R[INDEX_2D(end - 2, end - 6)] = 16.0 / 180.0;
        R[INDEX_2D(end - 2, end - 7)] = -2.0 / 180.0;

        // fourth row
        R[INDEX_2D(end - 2, end)] = 2.0 / 180.0;
        R[INDEX_2D(end - 2, end - 1)] = -27.0 / 180.0;
        R[INDEX_2D(end - 2, end - 2)] = 270.0 / 180.0;
        R[INDEX_2D(end - 2, end - 3)] = -490.0 / 180.0;  // POINT 0
        R[INDEX_2D(end - 2, end - 4)] = 270.0 / 180.0;
        R[INDEX_2D(end - 2, end - 5)] = -27.0 / 180.0;
        R[INDEX_2D(end - 2, end - 6)] = 2.0 / 180.0;
        // 7th is empty

        end -= 3;
    }

    // have to include "end", so add 1
    for (unsigned int i = start; i < end + 1; i++) {
        R[INDEX_2D(i, i - 4)] = -9.0 / 540.0;
        R[INDEX_2D(i, i - 3)] = 128.0 / 540.0;
        R[INDEX_2D(i, i - 2)] = -1008.0 / 540.0;
        R[INDEX_2D(i, i - 1)] = 8064.0 / 540.0;
        R[INDEX_2D(i, i - 0)] = -14350.0 / 540.0;
        R[INDEX_2D(i, i + 1)] = 8064.0 / 540.0;
        R[INDEX_2D(i, i + 2)] = -1008.0 / 540.0;
        R[INDEX_2D(i, i + 3)] = 128.0 / 540.0;
        R[INDEX_2D(i, i + 4)] = -9.0 / 540.0;
    }
}

}  // namespace dendro_cfd

void HAMRDeriv4_dP(double *P, int n) {
    double *tempP = new double[n * n];
    // Define the variables
    double a00 = 1.0;
    double a10 = 0.1023343303;
    double a20 = 0.0347468867;

    double a01 = 9.4133049605;
    double a11 = 1.0;
    double a21 = 0.4064246796;

    double a02 = 10.7741034803;
    double a12 = 1.8854940182;
    double a22 = 1.0;

    double a03 = 0.0;
    double a13 = 0.8582327249;
    double a23 = 0.7683583302;

    double a04 = 0.0;
    double a14 = 0.0;
    double a24 = 0.1623349133;

    double alpha = 0.5747612151;
    double beta1 = 0.0879324249;
    // Set the diagonal values of the array
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index =
                i * n + j;  // Calculate the 1D index for the (i, j) element
            if (i == j) {
                // Main diagonal
                tempP[index] = 1.0;
            } else if (i == j + 1 || i == j - 1) {
                // Adjacent diagonals
                tempP[index] = alpha;
            } else if (i == j + 2 || i == j - 2) {
                // Adjacent super-diagonals
                tempP[index] = beta1;
            } else {
                tempP[index] = 0.0;
            }
        }
    }

    // Set specific values in the array
    tempP[1] = a01;
    tempP[2] = a02;
    tempP[3] = 0.0;

    tempP[n] = a10;
    tempP[n + 2] = a12;
    tempP[n + 3] = a13;

    tempP[2 * n] = a20;
    tempP[2 * n + 1] = a21;
    tempP[2 * n + 3] = a23;
    tempP[2 * n + 4] = a24;

    tempP[(n - 3) * n + (n - 5)] = a24;
    tempP[(n - 3) * n + (n - 4)] = a23;
    tempP[(n - 3) * n + (n - 2)] = a21;
    tempP[(n - 3) * n + (n - 1)] = a20;

    tempP[(n - 2) * n + (n - 4)] = a13;
    tempP[(n - 2) * n + (n - 3)] = a12;
    tempP[(n - 2) * n + (n - 1)] = a10;

    tempP[(n - 1) * n + (n - 3)] = a02;
    tempP[(n - 1) * n + (n - 2)] = a01;

    // compute the transpose because i need to test this
    // TODO: fix the actual population
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[INDEX_2D(i, j)] = tempP[INDEX_2D(j, i)];
        }
    }

    delete[] tempP;
}

void HAMRDeriv4_dQ(double *Q, int n) {
    double *tempQ = new double[n * n];
    // Seting the constants
    double a = 1.3069171114;
    double b = 0.9828406281;
    double c = 0.0356295405;

    double p10 = -0.3586079596;
    double p20 = -0.1274311870;

    double p01 = -10.3169611301;
    double p21 = -0.6299599564;

    double p02 = 9.6807767746;
    double p12 = 0.0834751059;

    double p03 = 5.4529053045;
    double p13 = 1.4235697122;
    double p23 = 0.6498856630;

    double p04 = -1.51919598290;
    double p14 = 0.2245783548;
    double p24 = 0.3919470424;

    double p05 = 0.4834876759;
    double p15 = -0.0358453729;
    double p25 = 0.0189402158;

    double p06 = -0.0927590566;
    double p16 = 0.0052970021;
    double p26 = -0.0008894789;

    // from equation 13
    double p00 = -(p01 + p02 + p03 + p04 + p05 + p06);
    double p11 = -(p10 + p12 + p13 + p14 + p15 + p16);
    double p22 = -(p20 + p21 + p23 + p24 + p25 + p26);

    // Set the diagonal values of the array
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index =
                i * n + j;  // Calculate the 1D index for the (i, j) element
            if (i == j) {
                // Main diagonal
                tempQ[index] = 0.0;
            } else if (i == j - 1) {
                // Adjacent diagonals
                tempQ[index] = a / 2.0;
            } else if (i == j + 1) {
                // Adjacent diagonals
                tempQ[index] = -a / 2.0;
            } else if (i == j - 2) {
                // Super diagonals
                tempQ[index] = b / 4.0;
            } else if (i == j + 2) {
                // Super diagonals
                tempQ[index] = -b / 4.0;
            } else if (i == j - 3) {
                // Super +1 diagonals
                tempQ[index] = c / 6.0;
            } else if (i == j + 3) {
                // Super +1 diagonals
                tempQ[index] = -c / 6.0;
            } else {
                tempQ[index] = 0.0;
            }
        }
    }

    // Set specific values in the array
    tempQ[0] = p00;
    tempQ[1] = p01;
    tempQ[2] = p02;
    tempQ[3] = p03;
    tempQ[4] = p04;
    tempQ[5] = p05;
    tempQ[6] = p06;

    tempQ[n] = p10;
    tempQ[n + 1] = p11;
    tempQ[n + 2] = p12;
    tempQ[n + 3] = p13;
    tempQ[n + 4] = p14;
    tempQ[n + 5] = p15;
    tempQ[n + 6] = p16;

    tempQ[2 * n] = p20;
    tempQ[2 * n + 1] = p21;
    tempQ[2 * n + 2] = p22;
    tempQ[2 * n + 3] = p23;
    tempQ[2 * n + 4] = p24;
    tempQ[2 * n + 5] = p25;
    tempQ[2 * n + 6] = p26;

    tempQ[(n - 3) * n + (n - 7)] = -p26;
    tempQ[(n - 3) * n + (n - 6)] = -p25;
    tempQ[(n - 3) * n + (n - 5)] = -p24;
    tempQ[(n - 3) * n + (n - 4)] = -p23;
    tempQ[(n - 3) * n + (n - 3)] = -p22;
    tempQ[(n - 3) * n + (n - 2)] = -p21;
    tempQ[(n - 3) * n + (n - 1)] = -p20;

    tempQ[(n - 2) * n + (n - 7)] = -p16;
    tempQ[(n - 2) * n + (n - 6)] = -p15;
    tempQ[(n - 2) * n + (n - 5)] = -p14;
    tempQ[(n - 2) * n + (n - 4)] = -p13;
    tempQ[(n - 2) * n + (n - 3)] = -p12;
    tempQ[(n - 2) * n + (n - 2)] = -p11;
    tempQ[(n - 2) * n + (n - 1)] = -p10;

    tempQ[(n - 1) * n + (n - 7)] = -p06;
    tempQ[(n - 1) * n + (n - 6)] = -p05;
    tempQ[(n - 1) * n + (n - 5)] = -p04;
    tempQ[(n - 1) * n + (n - 4)] = -p03;
    tempQ[(n - 1) * n + (n - 3)] = -p02;
    tempQ[(n - 1) * n + (n - 2)] = -p01;
    tempQ[(n - 1) * n + (n - 1)] = -p00;

    // compute the transpose because i need to test this
    // TODO: fix the actual population
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[INDEX_2D(i, j)] = tempQ[INDEX_2D(j, i)];
        }
    }

    delete[] tempQ;
}

bool initHAMRDeriv4(double *R, const unsigned int n) {
    double *P = new double[n * n];
    double *Q = new double[n * n];
    HAMRDeriv4_dP(P, n);  // define the matrix P using the function provided

    // Define the matrix Q
    HAMRDeriv4_dQ(Q, n);

    // Compute the LU decomposition of the matrix P
    int *ipiv = new int[n];
    int info;
    int nx = n;  // lapack needs fortran-compatible ints, not const unsigned
    dgetrf_(&nx, &nx, P, &nx, ipiv, &info);

    if (info != 0) {
        std::cerr << "LU factorization failed: " << info << std::endl;
        delete[] ipiv;
        return 1;
    }

    // Compute the inverse of the matrix P
    double *Pinv = new double[n * n];
    std::memcpy(Pinv, P, n * n * sizeof(double));
    int lwork = n * n;
    double *work = new double[lwork];
    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        std::cerr << "Matrix inversion failed: " << info << std::endl;
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        return 1;
    }

    // Compute the product of the inverted matrix Pinv and matrix Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            R[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                R[i * n + j] += Pinv[i * n + k] * Q[k * n + j];
            }
        }
    }

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
    delete[] P;
    delete[] Q;

    return 0;
}

void JTPDeriv6_dP(double *P, int n) {
    double *tempP = new double[n * n];

    double alpha = 17.0 / 57.0;
    double beta = -1.0 / 114.0;

    // Initialize the matrix to zeros
    for (int i = 0; i < n * n; i++) {
        tempP[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Main diagonal
                tempP[i * n + j] = 1;
            } else if (i == j + 1 || i == j - 1) {
                // Adjacent diagonals
                tempP[i * n + j] = alpha;
            } else if (i == j + 2 || i == j - 2) {
                // Adjacent super-diagonals
                tempP[i * n + j] = beta;
            } else {
                tempP[i * n + j] = 0;
            }
        }
    }

    tempP[0 * n + 1] = 8.0;
    tempP[0 * n + 2] = 6.0;
    tempP[(n - 1) * n + (n - 3)] = 6.0;
    tempP[(n - 1) * n + (n - 2)] = 8.0;

    // Set diagonal values to 1
    for (int i = 0; i < n; i++) {
        tempP[i * n + i] = 1.0;
    }

    // compute the transpose because i need to test this
    // TODO: fix the actual population
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[INDEX_2D(i, j)] = tempP[INDEX_2D(j, i)];
        }
    }

    delete[] tempP;
}
void JTPDeriv6_dQ(double *Q, int n) {
    double *tempQ = new double[n * n];

    double hm1 = 1.0 / 1.0;

    // Initialize the matrix to zeros
    for (int i = 0; i < n * n; i++) {
        tempQ[i] = 0.0;
    }

    double a = -43.0 / 12.0 * hm1;
    double b = -20.0 / 3.0 * hm1;
    double c = 9.0 * hm1;
    double d = 4.0 / 3.0 * hm1;
    double e = -1.0 / 12.0 * hm1;

    double t1 = 30.0 / 19.0 * 0.5 * hm1;

    tempQ[(n - 1) * n + (n - 5)] = -e;
    tempQ[(n - 1) * n + (n - 4)] = -d;
    tempQ[(n - 1) * n + (n - 3)] = -c;
    tempQ[(n - 1) * n + (n - 2)] = -b;
    tempQ[(n - 1) * n + (n - 1)] = -a;

    tempQ[0 * n + 0] = a;
    tempQ[0 * n + 1] = b;
    tempQ[0 * n + 2] = c;
    tempQ[0 * n + 3] = d;
    tempQ[0 * n + 4] = e;

    for (int i = 1; i < n - 1; i++) {
        tempQ[i * n + (i - 1)] = -t1;
        tempQ[i * n + (i + 1)] = t1;
    }

    // compute the transpose because i need to test this
    // TODO: fix the actual population
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[INDEX_2D(i, j)] = tempQ[INDEX_2D(j, i)];
        }
    }

    delete[] tempQ;
}
bool initJTPDeriv6(double *R, const unsigned int n) {
    double *P = new double[n * n];
    double *Q = new double[n * n];
    JTPDeriv6_dP(P, n);  // define the matrix P using the function provided

    // Define the matrix Q
    JTPDeriv6_dQ(Q, n);

    // Compute the LU decomposition of the matrix P
    int *ipiv = new int[n];
    int info;
    int nx = n;  // lapack needs fortran-compatible ints, not const unsigned
    dgetrf_(&nx, &nx, P, &nx, ipiv, &info);

    if (info != 0) {
        std::cerr << "LU factorization failed: " << info << std::endl;
        delete[] ipiv;
        return 1;
    }

    // Compute the inverse of the matrix P
    double *Pinv = new double[n * n];
    std::memcpy(Pinv, P, n * n * sizeof(double));
    int lwork = n * n;
    double *work = new double[lwork];
    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        std::cerr << "Matrix inversion failed: " << info << std::endl;
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        return 1;
    }

    // Compute the product of the inverted matrix Pinv and matrix Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            R[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                R[i * n + j] += Pinv[i * n + k] * Q[k * n + j];
            }
        }
    }

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
    delete[] P;
    delete[] Q;

    return 0;
}

// THESE ARE THE OLDER KIM DERIVATIVES

void KimDeriv4_dP(double *P, int n) {
    double *tempP = new double[n * n];
    // defining the constants
    double g10 = 0.11737546726594537;
    double g20 = -0.067477420334188354;
    double g01 = 9.279310823736082;
    double g21 = -0.1945509344676567;
    double g02 = 9.8711877434133051;
    double g12 = 0.92895849448052303;
    double g13 = -0.067839996199150834;
    double g23 = 1.279565347145571;
    double g24 = 0.20842348769505742;

    double alpha = 0.5862704032801503;
    double beta = 0.09549533555017055;

    // Set the diagonal values of the matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Main diagonal
                tempP[i * n + j] = 1;
                // tempP[INDEX_2D(i, j)] = 1.0;
            } else if (i == j + 1 || i == j - 1) {
                // Adjacent diagonals
                tempP[i * n + j] = alpha;
                // tempP[INDEX_2D(i, j)] = alpha;
            } else if (i == j + 2 || i == j - 2) {
                // Adjacent super-diagonals
                tempP[i * n + j] = beta;
                // tempP[INDEX_2D(i, j)] = beta;
            } else {
                tempP[i * n + j] = 0;
                // tempP[INDEX_2D(i, j)] = 0.0;
            }
        }
    }
    tempP[0 * n + 1] = g01;
    tempP[0 * n + 2] = g02;
    // tempP[INDEX_2D(0, 1)] = g01;
    // tempP[INDEX_2D(0, 2)] = g02;

    tempP[1 * n + 0] = g10;
    tempP[1 * n + 2] = g12;
    tempP[1 * n + 3] = g13;
    // tempP[INDEX_2D(1, 0)] = g10;
    // tempP[INDEX_2D(1, 2)] = g12;
    // tempP[INDEX_2D(1, 3)] = g13;

    tempP[2 * n + 0] = g20;
    tempP[2 * n + 1] = g21;
    tempP[2 * n + 3] = g23;
    tempP[2 * n + 4] = g24;
    // tempP[INDEX_2D(2, 0)] = g20;
    // tempP[INDEX_2D(2, 1)] = g21;
    // tempP[INDEX_2D(2, 3)] = g23;
    // tempP[INDEX_2D(2, 3)] = g24;

    tempP[(n - 3) * n + (n - 5)] = g24;
    tempP[(n - 3) * n + (n - 4)] = g23;
    tempP[(n - 3) * n + (n - 2)] = g21;
    tempP[(n - 3) * n + (n - 1)] = g20;
    // tempP[INDEX_2D(n - 3, n - 5)] = g24;
    // tempP[INDEX_2D(n - 3, n - 4)] = g23;
    // tempP[INDEX_2D(n - 3, n - 2)] = g21;
    // tempP[INDEX_2D(n - 3, n - 1)] = g20;

    tempP[(n - 2) * n + (n - 4)] = g13;
    tempP[(n - 2) * n + (n - 3)] = g12;
    tempP[(n - 2) * n + (n - 1)] = g10;
    // tempP[INDEX_2D(n - 2, n - 4)] = g13;
    // tempP[INDEX_2D(n - 2, n - 3)] = g12;
    // tempP[INDEX_2D(n - 2, n - 1)] = g10;

    tempP[(n - 1) * n + (n - 3)] = g02;
    tempP[(n - 1) * n + (n - 2)] = g01;
    // tempP[INDEX_2D(n - 1, n - 3)] = g02;
    // tempP[INDEX_2D(n - 1, n - 2)] = g01;

    // compute the transpose because i need to test this
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[INDEX_2D(i, j)] = tempP[INDEX_2D(j, i)];
        }
    }

    delete[] tempP;
}

void KimDeriv4_dQ(double *Q, int n) {
    double *tempQ = new double[n * n];

    // defining some important variables
    double b10 = -0.4197688256685424;
    double b20 = 0.20875393530974462;

    double b01 = -9.9196421679170452;
    double b21 = -0.36722447739446801;

    double b02 = 10.088151775649886;
    double b12 = 1.1593253854830003;

    double b03 = 4.1769460418803268;
    double b13 = 0.31685797023808876;
    double b23 = 0.98917602108458036;

    double b04 = -0.82222305192207212;
    double b14 = -0.096453054902842381;
    double b24 = 0.63518969715000262;

    double b05 = 0.14757709267988142;
    double b15 = 0.015579947274307879;
    double b25 = 0.0042145635666246068;

    double b06 = -0.014332365879513103;
    double b16 = -0.0014553614585464077;
    double b26 = 0.0010111910030585999;

    double a1 = 0.6431406736919156;
    double a2 = 0.2586011023495066;
    double a3 = 0.007140953479797375;

    double b00 = -(b01 + b02 + b03 + b04 + b05 + b06);
    double b11 = -(b10 + b12 + b13 + b14 + b15 + b16);
    double b22 = -(b20 + b21 + b23 + b24 + b25 + b26);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // i * n + j
                // Main diagonal
                tempQ[i * n + j] = 0.0;
                // tempQ[INDEX_2D(i, j)] = 0.0;
            } else if (i == j - 1) {
                // Adjacent diagonals
                tempQ[i * n + j] = a1;
                // tempQ[INDEX_2D(i, j)] = a1;
            } else if (i == j + 1) {
                // Adjacent diagonals
                tempQ[i * n + j] = -a1;
                // tempQ[INDEX_2D(i, j)] = -a1;
            } else if (i == j - 2) {
                // Super diagonals
                tempQ[i * n + j] = a2;
                // tempQ[INDEX_2D(i, j)] = a2;
            } else if (i == j + 2) {
                // Super diagonals on
                tempQ[i * n + j] = -a2;
                // tempQ[INDEX_2D(i, j)] = -a2;
            } else if (i == j - 3) {
                // Super +1 diagonals on
                tempQ[i * n + j] = a3;
                // tempQ[INDEX_2D(i, j)] = a3;
            } else if (i == j + 3) {
                // Super +1 diagonals on
                tempQ[i * n + j] = -a3;
                // tempQ[INDEX_2D(i, j)] = -a3;
            } else {
                tempQ[i * n + j] = 0.0;
                // tempQ[INDEX_2D(i, j)] = 0.0;
            }
        }
    }

    tempQ[0 * n + 0] = b00;
    tempQ[0 * n + 1] = b01;
    tempQ[0 * n + 2] = b02;
    tempQ[0 * n + 3] = b03;
    tempQ[0 * n + 4] = b04;
    tempQ[0 * n + 5] = b05;
    tempQ[0 * n + 6] = b06;
    // tempQ[INDEX_2D(0, 0)] = b00;
    // tempQ[INDEX_2D(0, 1)] = b01;
    // tempQ[INDEX_2D(0, 2)] = b02;
    // tempQ[INDEX_2D(0, 3)] = b03;
    // tempQ[INDEX_2D(0, 4)] = b04;
    // tempQ[INDEX_2D(0, 5)] = b05;
    // tempQ[INDEX_2D(0, 6)] = b06;

    tempQ[1 * n + 0] = b10;
    tempQ[1 * n + 1] = b11;
    tempQ[1 * n + 2] = b12;
    tempQ[1 * n + 3] = b13;
    tempQ[1 * n + 4] = b14;
    tempQ[1 * n + 5] = b15;
    tempQ[1 * n + 6] = b16;
    // tempQ[INDEX_2D(1, 0)] = b10;
    // tempQ[INDEX_2D(1, 1)] = b11;
    // tempQ[INDEX_2D(1, 2)] = b12;
    // tempQ[INDEX_2D(1, 3)] = b13;
    // tempQ[INDEX_2D(1, 4)] = b14;
    // tempQ[INDEX_2D(1, 5)] = b15;
    // tempQ[INDEX_2D(1, 6)] = b16;

    tempQ[2 * n + 0] = b20;
    tempQ[2 * n + 1] = b21;
    tempQ[2 * n + 2] = b22;
    tempQ[2 * n + 3] = b23;
    tempQ[2 * n + 4] = b24;
    tempQ[2 * n + 5] = b25;
    tempQ[2 * n + 6] = b26;
    // tempQ[INDEX_2D(2, 0)] = b20;
    // tempQ[INDEX_2D(2, 1)] = b21;
    // tempQ[INDEX_2D(2, 2)] = b22;
    // tempQ[INDEX_2D(2, 3)] = b23;
    // tempQ[INDEX_2D(2, 4)] = b24;
    // tempQ[INDEX_2D(2, 5)] = b25;
    // tempQ[INDEX_2D(2, 6)] = b26;

    tempQ[(n - 3) * n + (n - 7)] = -b26;
    tempQ[(n - 3) * n + (n - 6)] = -b25;
    tempQ[(n - 3) * n + (n - 5)] = -b24;
    tempQ[(n - 3) * n + (n - 4)] = -b23;
    tempQ[(n - 3) * n + (n - 3)] = -b22;
    tempQ[(n - 3) * n + (n - 2)] = -b21;
    tempQ[(n - 3) * n + (n - 1)] = -b20;
    // tempQ[INDEX_2D(n - 3, n - 7)] = -b26;
    // tempQ[INDEX_2D(n - 3, n - 6)] = -b25;
    // tempQ[INDEX_2D(n - 3, n - 5)] = -b24;
    // tempQ[INDEX_2D(n - 3, n - 4)] = -b23;
    // tempQ[INDEX_2D(n - 3, n - 3)] = -b22;
    // tempQ[INDEX_2D(n - 3, n - 2)] = -b21;
    // tempQ[INDEX_2D(n - 3, n - 1)] = -b20;

    tempQ[(n - 2) * n + (n - 7)] = -b16;
    tempQ[(n - 2) * n + (n - 6)] = -b15;
    tempQ[(n - 2) * n + (n - 5)] = -b14;
    tempQ[(n - 2) * n + (n - 4)] = -b13;
    tempQ[(n - 2) * n + (n - 3)] = -b12;
    tempQ[(n - 2) * n + (n - 2)] = -b11;
    tempQ[(n - 2) * n + (n - 1)] = -b10;
    // tempQ[INDEX_2D(n-2, n-7)] = -b16;
    // tempQ[INDEX_2D(n-2, n-6)] = -b15;
    // tempQ[INDEX_2D(n-2, n-5)] = -b14;
    // tempQ[INDEX_2D(n-2, n-4)] = -b13;
    // tempQ[INDEX_2D(n-2, n-3)] = -b12;
    // tempQ[INDEX_2D(n-2, n-2)] = -b11;
    // tempQ[INDEX_2D(n-2, n-1)] = -b10;

    tempQ[(n - 1) * n + (n - 7)] = -b06;
    tempQ[(n - 1) * n + (n - 6)] = -b05;
    tempQ[(n - 1) * n + (n - 5)] = -b04;
    tempQ[(n - 1) * n + (n - 4)] = -b03;
    tempQ[(n - 1) * n + (n - 3)] = -b02;
    tempQ[(n - 1) * n + (n - 2)] = -b01;
    tempQ[(n - 1) * n + (n - 1)] = -b00;
    // tempQ[INDEX_2D(n-1, n-7)] = -b06;
    // tempQ[INDEX_2D(n-1, n-6)] = -b05;
    // tempQ[INDEX_2D(n-1, n-5)] = -b04;
    // tempQ[INDEX_2D(n-1, n-4)] = -b03;
    // tempQ[INDEX_2D(n-1, n-3)] = -b02;
    // tempQ[INDEX_2D(n-1, n-2)] = -b01;
    // tempQ[(INDEX_2D(n-1, n-1)] = -b00;

    // compute the transpose because i need to test this
    // TODO: fix the actual population
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[INDEX_2D(i, j)] = tempQ[INDEX_2D(j, i)];
        }
    }

    delete[] tempQ;
}

bool initKimDeriv4(double *R, const unsigned int n) {
    double *P = new double[n * n];
    double *Q = new double[n * n];
    KimDeriv4_dP(P, n);  // define the matrix P using the function provided

    // Define the matrix Q
    KimDeriv4_dQ(Q, n);

    // Compute the LU decomposition of the matrix P
    int *ipiv = new int[n];
    int info;
    int nx = n;  // lapack needs fortran-compatible ints, not const unsigned
    dgetrf_(&nx, &nx, P, &nx, ipiv, &info);

    if (info != 0) {
        std::cerr << "LU factorization failed: " << info << std::endl;
        delete[] ipiv;
        return 1;
    }

    // Compute the inverse of the matrix P
    double *Pinv = new double[n * n];
    std::memcpy(Pinv, P, n * n * sizeof(double));
    int lwork = n * n;
    double *work = new double[lwork];
    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        std::cerr << "Matrix inversion failed: " << info << std::endl;
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        return 1;
    }

    // Compute the product of the inverted matrix Pinv and matrix Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            R[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                R[i * n + j] += Pinv[i * n + k] * Q[k * n + j];
            }
        }
    }

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
    delete[] P;
    delete[] Q;

    return 0;
}

void Kim_Filter_Deriv4_dPf(double *Pf, int n) {
    // defining the constants
    double gf10 = 0.7311329755609861;
    double gf20 = 0.1681680891936087;

    double gf01 = 0.3412746505356879;
    double gf21 = 0.6591595540319565;

    double gf02 = 0.2351300295562464;
    double gf12 = 0.6689728401317021;
    double gf13 = 0.1959510121583215;
    double gf23 = 0.6591595540319565;
    double gf24 = 0.1681680891936087;

    double alphaf = 0.6651452077642562;
    double betaf = 0.1669709584471488;

    // Create an nxn matrix with all elements initialized to zero

    // Set the diagonal values of the matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Main diagonal
                Pf[i * n + j] = 1;
            } else if (i == j + 1 || i == j - 1) {
                // Adjacent diagonals
                Pf[i * n + j] = alphaf;
            } else if (i == j + 2 || i == j - 2) {
                // Adjacent super-diagonals
                Pf[i * n + j] = betaf;
            } else {
                Pf[i * n + j] = 0;
            }
        }
    }

    Pf[0 * n + 1] = gf01;
    Pf[0 * n + 2] = gf02;

    Pf[1 * n + 0] = gf10;
    Pf[1 * n + 2] = gf12;
    Pf[1 * n + 3] = gf13;

    Pf[2 * n + 0] = gf20;
    Pf[2 * n + 1] = gf21;
    Pf[2 * n + 3] = gf23;
    Pf[2 * n + 4] = gf24;

    Pf[(n - 3) * n + (n - 5)] = gf24;
    Pf[(n - 3) * n + (n - 4)] = gf23;
    Pf[(n - 3) * n + (n - 2)] = gf21;
    Pf[(n - 3) * n + (n - 1)] = gf20;

    Pf[(n - 2) * n + (n - 4)] = gf13;
    Pf[(n - 2) * n + (n - 3)] = gf12;
    Pf[(n - 2) * n + (n - 1)] = gf10;

    Pf[(n - 1) * n + (n - 3)] = gf02;
    Pf[(n - 1) * n + (n - 2)] = gf01;
}

void Kim_Filter_Deriv4_dQf(double *Qf, int n) {
    // defining some important variables
    double bf20 = -2.81516723801634e-4;
    double bf21 = 1.40758361900817e-3;

    double bf23 = 2.81516723801634e-3;

    double bf24 = -1.40758361900817e-3;
    double bf25 = 2.81516723801634e-4;

    double alphaf1 = 8.558206326059179e-4;
    double alphaf2 = -3.423282530423672e-4;
    double alphaf3 = 5.705470884039454e-5;

    double bf22 = -(bf20 + bf21 + bf23 + bf24 + bf25);
    double alphaf0 = -2 * (alphaf1 + alphaf2 + alphaf3);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Main diagonal
                Qf[i * n + j] = alphaf0;
            } else if (i == j - 1) {
                // Adjacent diagonals
                Qf[i * n + j] = alphaf1;
            } else if (i == j + 1) {
                // Adjacent diagonals
                Qf[i * n + j] = alphaf1;
            } else if (i == j - 2) {
                // Super diagonals
                Qf[i * n + j] = alphaf2;
            } else if (i == j + 2) {
                // Super diagonals on
                Qf[i * n + j] = alphaf2;
            } else if (i == j - 3) {
                // Super +1 diagonals on
                Qf[i * n + j] = alphaf3;
            } else if (i == j + 3) {
                // Super +1 diagonals on
                Qf[i * n + j] = alphaf3;
            } else {
                Qf[i * n + j] = 0.0;
            }
        }
    }

    Qf[0 * n + 0] = 0;
    Qf[0 * n + 1] = 0;
    Qf[0 * n + 2] = 0;
    Qf[0 * n + 3] = 0;
    Qf[0 * n + 4] = 0;
    Qf[0 * n + 5] = 0;
    Qf[0 * n + 6] = 0;

    Qf[1 * n + 0] = 0;
    Qf[1 * n + 1] = 0;
    Qf[1 * n + 2] = 0;
    Qf[1 * n + 3] = 0;
    Qf[1 * n + 4] = 0;
    Qf[1 * n + 5] = 0;
    Qf[1 * n + 6] = 0;

    Qf[2 * n + 0] = bf20;
    Qf[2 * n + 1] = bf21;
    Qf[2 * n + 2] = bf22;
    Qf[2 * n + 3] = bf23;
    Qf[2 * n + 4] = bf24;
    Qf[2 * n + 5] = bf25;

    Qf[(n - 3) * n + (n - 6)] = bf25;
    Qf[(n - 3) * n + (n - 5)] = bf24;
    Qf[(n - 3) * n + (n - 4)] = bf23;
    Qf[(n - 3) * n + (n - 3)] = bf22;
    Qf[(n - 3) * n + (n - 2)] = bf21;
    Qf[(n - 3) * n + (n - 1)] = bf20;

    Qf[(n - 2) * n + (n - 7)] = 0.0;
    Qf[(n - 2) * n + (n - 6)] = 0.0;
    Qf[(n - 2) * n + (n - 5)] = 0.0;
    Qf[(n - 2) * n + (n - 4)] = 0.0;
    Qf[(n - 2) * n + (n - 3)] = 0.0;
    Qf[(n - 2) * n + (n - 2)] = 0.0;
    Qf[(n - 2) * n + (n - 1)] = 0.0;

    Qf[(n - 1) * n + (n - 7)] = 0.0;
    Qf[(n - 1) * n + (n - 6)] = 0.0;
    Qf[(n - 1) * n + (n - 5)] = 0.0;
    Qf[(n - 1) * n + (n - 4)] = 0.0;
    Qf[(n - 1) * n + (n - 3)] = 0.0;
    Qf[(n - 1) * n + (n - 2)] = 0.0;
    Qf[(n - 1) * n + (n - 1)] = 0.0;
}
bool initKim_Filter_Deriv4(double *RF, const unsigned int n) {
    double *Pf = new double[n * n];
    double *Qf = new double[n * n];
    Kim_Filter_Deriv4_dPf(
        Pf, n);  // define the matrix Pf using the function provided

    // Define the matrix Qf
    Kim_Filter_Deriv4_dQf(Qf, n);

    // Compute the LU decomposition of the matrix Pf
    int *ipiv = new int[n];
    int info;
    int nx = n;  // lapack needs fortran-compatible ints, not const unsigned
    dgetrf_(&nx, &nx, Pf, &nx, ipiv, &info);

    if (info != 0) {
        std::cerr << "LU factorization failed: " << info << std::endl;
        delete[] ipiv;
        return 1;
    }

    // Compute the inverse of the matrix Pf
    double *Pfinv = new double[n * n];
    std::memcpy(Pfinv, Pf, n * n * sizeof(double));
    int lwork = n * n;
    double *work = new double[lwork];
    dgetri_(&nx, Pfinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        std::cerr << "Matrix inversion failed: " << info << std::endl;
        delete[] ipiv;
        delete[] Pfinv;
        delete[] work;
        return 1;
    }

    // Compute the product of the inverted matrix Pinv and matrix Qf
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            RF[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                RF[i * n + j] += Pfinv[i * n + k] * Qf[k * n + j];
            }
        }
    }

    delete[] ipiv;
    delete[] Pfinv;
    delete[] work;
    delete[] Pf;
    delete[] Qf;

    return 0;
}
