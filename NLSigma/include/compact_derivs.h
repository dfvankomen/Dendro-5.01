#pragma once

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
#include <libxsmm.h>
#endif

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

#include "dendro.h"

#define INDEX_3D(i, j, k) ((i) + nx * ((j) + ny * (k)))

#define INDEX_2D(i, j) ((i) + n * (j))

#define INDEX_N2D(i, j, n) ((i) + (n) * (j))

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
}

namespace dendro_cfd {

// enum DerType { CFD_P1_O4 = 0, CFD_P1_O6, CFD_Q1_O6_ETA1 };

enum DerType {
    // NO CFD Initialization
    CFD_NONE = -1,

    // the "main" compact finite difference types
    CFD_P1_O4 = 0,
    CFD_P1_O6,
    CFD_Q1_O6_ETA1,
    // isotropic finite difference types
    CFD_KIM_O4,
    CFD_HAMR_O4,
    CFD_JT_O6,

    // Explicit options using matrix mult
    EXPLCT_FD_O4,
    EXPLCT_FD_O6,
    EXPLCT_FD_O8,

    // additional "helpers" that are mostly for internal/edge building
    CFD_DRCHLT_ORDER_4,
    CFD_DRCHLT_ORDER_6,
    CFD_P1_O4_CLOSE,
    CFD_P1_O6_CLOSE,
    CFD_P1_O4_L4_CLOSE,
    CFD_P1_O6_L6_CLOSE,
    CFD_Q1_O6,
    CFD_Q1_O6_CLOSE,
    CFD_DRCHLT_Q6,
    CFD_DRCHLT_Q6_L6,
    CFD_Q1_O6_ETA1_CLOSE,

};

enum DerType2nd {
    // NO CFD Initialization
    CFD2ND_NONE = -1,

    // the "main" compact finite difference types
    CFD2ND_P2_O4 = 0,
    CFD2ND_P2_O6,
    CFD2ND_Q2_O6_ETA1,
    // isotropic finite difference types
    CFD2ND_KIM_O4,   // FIX: KIM second orders aren't supported yet
    CFD2ND_HAMR_O4,  // FIX: HAMR second order isn't supported yet
    CFD2ND_JT_O6,    // FIX: JT second order isn't supported yet

    // explicit options using matrix mult
    EXPLCT2ND_FD_O4,
    EXPLCT2ND_FD_O6,
    EXPLCT2ND_FD_O8,

    // additional "helpers" that are mostly for internal/edge building
    CFD2ND_DRCHLT_ORDER_4,
    CFD2ND_DRCHLT_ORDER_6,
    CFD2ND_P2_O4_CLOSE,
    CFD2ND_P2_O6_CLOSE,
    CFD2ND_P2_O4_L4_CLOSE,
    CFD2ND_P2_O6_L6_CLOSE,
    CFD2ND_Q2_O6,
    CFD2ND_Q2_O6_CLOSE,
    CFD2ND_DRCHLT_Q6,
    CFD2ND_DRCHLT_Q6_L6,
    CFD2ND_Q2_O6_ETA1_CLOSE,

};

enum FilterType {
    // NO CFD Initialization
    FILT_NONE = -1,

    // standard filters...
    FILT_KO_DISS = 0,

    // isotropic finite difference types
    FILT_KIM_6,
    FILT_JT_6,
    FILT_JT_8,

    // explicit ko diss
    EXPLCT_KO,

};

// NOTE: these are going to be used as global parameters if they're not physical
enum BoundaryType {
    BLOCK_CFD_CLOSURE = 0,  // closure gives better results but 6th requires 4
                            // points, and 4th requires 3 points
    BLOCK_CFD_DIRICHLET,
    BLOCK_CFD_LOPSIDE_CLOSURE,
    BLOCK_PHYS_BOUNDARY
};

class CFDMethod {
   public:
    DerType name;
    uint32_t order;
    int32_t Ld;
    int32_t Rd;
    int32_t Lf;
    int32_t Rf;
    double alpha[16];
    double a[16];

    CFDMethod(DerType dertype) {
        switch (dertype) {
            case CFD_P1_O4:
                set_for_CFD_P1_O4();
                break;

            case CFD_P1_O6:
                set_for_CFD_P1_O6();
                break;

            case CFD_Q1_O6_ETA1:
                set_for_CFD_Q1_O6_ETA1();
                break;

            default:
                throw std::invalid_argument(
                    "Invalid CFD method type of " + std::to_string(dertype) +
                    " for initializing the CFDMethod Object");
                break;
        }
    }

    ~CFDMethod() {}

    void set_for_CFD_P1_O4() {
        name = CFD_P1_O4;
        order = 4;
        Ld = 1;
        Rd = 1;
        Lf = 1;
        Rf = 1;
        alpha[0] = 0.25;
        alpha[1] = 1.0;
        alpha[2] = 0.25;

        a[0] = -0.75;
        a[1] = 0.0;
        a[2] = 0.75;
    }

    void set_for_CFD_P1_O6() {
        name = CFD_P1_O6;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 2;
        Rf = 2;

        alpha[0] = 1.0 / 3.0;
        alpha[1] = 1.0;
        alpha[2] = 1.0 / 3.0;

        const double t1 = 1.0 / 36.0;
        a[0] = -t1;
        a[1] = -28.0 * t1;
        a[2] = 0.0;
        a[3] = 28.0 * t1;
        a[4] = t1;
    }

    void set_for_CFD_Q1_O6_ETA1() {
        name = CFD_Q1_O6_ETA1;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 3;
        Rf = 3;

        alpha[0] = 0.37987923;
        alpha[1] = 1.0;
        alpha[2] = 0.37987923;

        a[0] = 0.0023272948;
        a[1] = -0.052602255;
        a[2] = -0.78165660;
        a[3] = 0.0;
        a[4] = 0.78165660;
        a[5] = 0.052602255;
        a[6] = -0.0023272948;
    }
};

class CFDMethod2nd {
   public:
    DerType2nd name;
    uint32_t order;
    int32_t Ld;
    int32_t Rd;
    int32_t Lf;
    int32_t Rf;
    double alpha[16];
    double a[16];

    CFDMethod2nd(DerType2nd dertype) {
        switch (dertype) {
            case CFD2ND_P2_O4:
                set_for_CFD_P2_O4();
                break;

            case CFD2ND_P2_O6:
                set_for_CFD_P2_O6();
                break;

            case CFD2ND_Q2_O6_ETA1:
                set_for_CFD_Q2_O6_ETA1();
                break;

            default:
                throw std::invalid_argument(
                    "Invalid CFD 2nd order method type of " +
                    std::to_string(dertype) +
                    " for initializing the CFDMethod2nd Object");
                break;
        }
    }

    ~CFDMethod2nd() {}

    void set_for_CFD_P2_O4() {
        name = CFD2ND_P2_O4;
        order = 4;
        Ld = 1;
        Rd = 1;
        Lf = 1;
        Rf = 1;
        alpha[0] = 0.1;
        alpha[1] = 1.0;
        alpha[2] = 0.1;

        a[0] = 6.0 / 5.0;
        a[1] = -12.0 / 5.0;
        a[2] = 6.0 / 5.0;
    }

    void set_for_CFD_P2_O6() {
        name = CFD2ND_P2_O6;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 2;
        Rf = 2;

        alpha[0] = 2.0 / 11.0;
        alpha[1] = 1.0;
        alpha[2] = 2.0 / 11.0;

        const double t1 = 1.0 / 44.0;
        a[0] = 3.0 * t1;
        a[1] = 48.0 * t1;
        a[2] = -102.0 * t1;
        a[3] = 48.0 * t1;
        a[4] = 3.0 * t1;
    }

    void set_for_CFD_Q2_O6_ETA1() {
        name = CFD2ND_Q2_O6_ETA1;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 3;
        Rf = 3;

        alpha[0] = 0.24246603;
        alpha[1] = 1.0;
        alpha[2] = 0.24246603;

        a[0] = -0.0037062571;
        a[1] = 0.14095923;
        a[2] = 0.95445144;
        a[3] = -2.1834088;
        a[4] = 0.95445144;
        a[5] = 0.14095923;
        a[6] = -0.0037062571;
    }
};

void print_square_mat(double *m, const uint32_t n);

DerType getDerTypeForEdges(const DerType derivtype,
                           const BoundaryType boundary);

DerType2nd get2ndDerTypeForEdges(const DerType2nd derivtype,
                                 const BoundaryType boundary);

void buildPandQMatrices(double *P, double *Q, const uint32_t padding,
                        const uint32_t n, const DerType derivtype,
                        const bool is_left_edge = false,
                        const bool is_right_edge = false);

void buildPandQMatrices2ndOrder(double *P, double *Q, const uint32_t padding,
                                const uint32_t n, const DerType2nd derivtype,
                                const bool is_left_edge = false,
                                const bool is_right_edge = false);

void buildPandQFilterMatrices(double *P, double *Q, const uint32_t padding,
                              const uint32_t n, const FilterType derivtype,
                              const bool is_left_edge = false,
                              const bool is_right_edge = false);

void buildMatrixLeft(double *P, double *Q, int *xib, const DerType dtype,
                     const int nghosts, const int n);

void buildMatrixRight(double *P, double *Q, int *xie, const DerType dtype,
                      const int nghosts, const int n);

void buildMatrixLeft2nd(double *P, double *Q, int *xib, const DerType2nd dtype,
                        const int nghosts, const int n);

void buildMatrixRight2nd(double *P, double *Q, int *xie, const DerType2nd dtype,
                         const int nghosts, const int n);

void calculateDerivMatrix(double *D, double *P, double *Q, const int n);

void setArrToZero(double *Mat, const int n);

/*
 Computes
     C := alpha*op( A )*op( B ) + beta*C,
*/
void mulMM(double *C, double *A, double *B, int na, int nb);

enum CompactDerivValueOrder {
    DERIV_NORM = 0,
    DERIV_LEFT,
    DERIV_RIGHT,
    DERIV_LEFTRIGHT,
    DERIV_2ND_NORM,
    DERIV_2ND_LEFT,
    DERIV_2ND_RIGHT,
    DERIV_2ND_LEFTRIGHT,
    FILT_NORM,
    FILT_LEFT,
    FILT_RIGHT,
    FILT_LEFTRIGHT,
    R_MAT_END
};

class CompactFiniteDiff {
   private:
    // STORAGE VARIABLES USED FOR THE DIFFERENT DIMENSIONS
    // Assume that the blocks are all the same size (to start with)

    double *m_RMatrices[CompactDerivValueOrder::R_MAT_END] = {};

    // TODO: we're going to want to store the filter and R variables as hash
    // maps
    // // Storage for the R matrix operator (combined P and Q matrices in CFD)
    // std::map<uint32_t, double *> m_R_storage;
    // // Storage for the RF matrix operator (the filter matrix)
    // std::map<uint32_t, double *> m_RF_storage;

    // Temporary storage for operations in progress
    double *m_u1d = nullptr;
    double *m_u2d = nullptr;
    // Additional temporary storage for operations in progress
    double *m_du1d = nullptr;
    double *m_du2d = nullptr;

    // to check for initialization (not used)
    bool m_initialized_matrices = false;

    // storing the derivative and filter types internally
    // could just be the parameter types
    DerType m_deriv_type = CFD_KIM_O4;
    DerType2nd m_second_deriv_type = CFD2ND_P2_O4;
    FilterType m_filter_type = FILT_NONE;
    unsigned int m_curr_dim_size = 0;
    unsigned int m_padding_size = 0;

    double m_beta_filt = 0.0;

   public:
    CompactFiniteDiff(const unsigned int dim_size,
                      const unsigned int padding_size,
                      const DerType deriv_type = CFD_KIM_O4,
                      const DerType2nd second_deriv_type = CFD2ND_P2_O4,
                      const FilterType filter_type = FILT_NONE);
    ~CompactFiniteDiff();

    void change_dim_size(const unsigned int dim_size);

    void initialize_cfd_storage();
    void initialize_cfd_matrix();
    void initialize_cfd_filter();
    void delete_cfd_matrices();

    void set_filter_type(FilterType filter_type) {
        m_filter_type = filter_type;
    }

    void set_deriv_type(const DerType deriv_type) {
        if (deriv_type != CFD_NONE && deriv_type != CFD_P1_O4 &&
            deriv_type != CFD_P1_O6 && deriv_type != CFD_Q1_O6_ETA1 &&
            deriv_type != CFD_KIM_O4 && deriv_type != CFD_HAMR_O4 &&
            deriv_type != CFD_JT_O6 && deriv_type != EXPLCT_FD_O4 &&
            deriv_type != EXPLCT_FD_O6 && deriv_type != EXPLCT_FD_O8) {
            throw std::invalid_argument(
                "Couldn't change deriv type of CFD object, deriv type was not "
                "a valid "
                "'base' "
                "type: deriv_type = " +
                std::to_string(deriv_type));
        }
        m_deriv_type = deriv_type;
    }

    void set_second_deriv_type(const DerType2nd deriv_type) {
        if (deriv_type != CFD2ND_NONE && deriv_type != CFD2ND_P2_O4 &&
            deriv_type != CFD2ND_P2_O6 && deriv_type != CFD2ND_Q2_O6_ETA1 &&
            deriv_type != CFD2ND_KIM_O4 && deriv_type != CFD2ND_HAMR_O4 &&
            deriv_type != CFD2ND_JT_O6 && deriv_type != EXPLCT2ND_FD_O4 &&
            deriv_type != EXPLCT2ND_FD_O6 && deriv_type != EXPLCT2ND_FD_O8) {
            throw std::invalid_argument(
                "Couldn't change 2nd deriv type of CFD object, deriv type was "
                "not "
                "a valid "
                "'base' "
                "type: deriv_type = " +
                std::to_string(deriv_type));
        }
        m_second_deriv_type = deriv_type;
    }

    /**
     * Sets the padding size. NOTE however that this does *not* attempt to
     * regenerate the matrices, so be sure to call the initialization
     */
    void set_padding_size(const unsigned int padding_size) {
        m_padding_size = padding_size;
    }

    // the actual derivative computation side of things
    void cfd_x(double *const Dxu, const double *const u, const double dx,
               const unsigned int *sz, unsigned bflag);
    void cfd_y(double *const Dyu, const double *const u, const double dy,
               const unsigned int *sz, unsigned bflag);
    void cfd_z(double *const Dzu, const double *const u, const double dz,
               const unsigned int *sz, unsigned bflag);

    void cfd_xx(double *const Dxu, const double *const u, const double dx,
                const unsigned int *sz, unsigned bflag);
    void cfd_yy(double *const Dyu, const double *const u, const double dy,
                const unsigned int *sz, unsigned bflag);
    void cfd_zz(double *const Dzu, const double *const u, const double dz,
                const unsigned int *sz, unsigned bflag);

    // then the actual filters
    void filter_cfd_x(double *const u, double *const filtx_work,
                      const double dx, const unsigned int *sz, unsigned bflag);
    void filter_cfd_y(double *const u, double *const filty_work,
                      const double dy, const unsigned int *sz, unsigned bflag);
    void filter_cfd_z(double *const u, double *const filtz_work,
                      const double dz, const unsigned int *sz, unsigned bflag);
};

extern CompactFiniteDiff cfd;

/**
 * Initialization of various Compact Methods
 *
 * From this point on various compact finite methods can be calculated and
 * derived
 */

/**
 * Initialization of the P and Q matrices for Kim's 4th Order Compact
 * Derivatives
 *
 * P and Q are assumed to **already by zeroed out**.
 *
 * These derivative coefficients come from Tables I and II of :
 *
 * Jae Wook Kim, "Quasi-disjoint pentadiagonal matrix systems for
 * the parallelization of compact finite-difference schemes and
 * filters," Journal of Computational Physics 241 (2013) 168–194.
 */
void initializeKim4PQ(double *P, double *Q, int n);

/**
 * Initialization of the P and Q matrices for Kim's 6th Order Compact Filter
 *
 * P and Q are assumed to **already by zeroed out**.
 *
 * These filter coefficients come from Tables III and IV of :
 *
 * Jae Wook Kim, "Quasi-disjoint pentadiagonal matrix systems for
 * the parallelization of compact finite-difference schemes and
 * filters," Journal of Computational Physics 241 (2013) 168–194.
 */
void initializeKim6FilterPQ(double *P, double *Q, int n);

// KO explicit filters

void buildKOExplicitFilter(double *R, const unsigned int n,
                           const unsigned int padding, const unsigned int order,
                           bool is_left_edge, bool is_right_edge);

void buildKOExplicit6thOrder(double *R, const unsigned int n,
                             const unsigned int padding, bool is_left_edge,
                             bool is_right_edge);
void buildKOExplicit8thOrder(double *R, const unsigned int n,
                             const unsigned int padding, bool is_left_edge,
                             bool is_right_edge);

void buildDerivExplicitRMatrix(double *R, const unsigned int padding,
                               const unsigned int n, const DerType deriv_type,
                               const bool is_left_edge,
                               const bool is_right_edge);

void build2ndDerivExplicitRMatrix(double *R, const unsigned int padding,
                                  const unsigned int n,
                                  const DerType2nd deriv_type,
                                  const bool is_left_edge,
                                  const bool is_right_edge);

// explicit deriv operators
void buildDerivExplicit4thOrder(double *R, const unsigned int n,
                                bool is_left_edge, bool is_right_edge);

void buildDerivExplicit6thOrder(double *R, const unsigned int n,
                                bool is_left_edge, bool is_right_edge);

void buildDerivExplicit8thOrder(double *R, const unsigned int n,
                                bool is_left_edge, bool is_right_edge);

void build2ndDerivExplicit4thOrder(double *R, const unsigned int n,
                                   bool is_left_edge, bool is_right_edge);

void build2ndDerivExplicit6thOrder(double *R, const unsigned int n,
                                   bool is_left_edge, bool is_right_edge);

void build2ndDerivExplicit8thOrder(double *R, const unsigned int n,
                                   bool is_left_edge, bool is_right_edge);

class CFDNotImplemented : public std::exception {
   private:
    std::string message_;

   public:
    explicit CFDNotImplemented(const std::string &msg) : message_(msg) {}
    const char *what() { return message_.c_str(); }
};

}  // namespace dendro_cfd

/**
 * HAMR Derivatives and such Initialization
 */

/**
 * Initializes the "P" Matrix of the HAMR Derivatives
 *
 * @param P Pointer to the output "P" matrix
 * @param n The number of rows/cols of the square matrix
 */
void HAMRDeriv4_dP(double *P, int n);

/**
 * Initializes the "Q" Matrix of the HAMR Derivatives
 *
 * @param Q Pointer to the output "Q" matrix
 * @param n The number of rows/cols of the square matrix
 */
void HAMRDeriv4_dQ(double *Q, int n);

/**
 * Initializes the "R" Matrix of the HAMR Derivatives
 *
 * This is a combination function that can automatically call
 * dP and dQ to just give the R matrix.
 *
 * @param R Pointer to the output "R" matrix
 * @param n The number of rows/cols of the square matrix
 */
bool initHAMRDeriv4(double *R, const unsigned int n);

/**
 * JTP 6 Derivatives Initialization
 */

/**
 * Initializes the "P" Matrix of the JTP Derivatives
 *
 * @param P Pointer to the output "P" matrix
 * @param n The number of rows/cols of the square matrix
 */
void JTPDeriv6_dP(double *P, int n);

/**
 * Initializes the "Q" Matrix of the JTP Derivatives
 *
 * @param Q Pointer to the output "Q" matrix
 * @param n The number of rows/cols of the square matrix
 */
void JTPDeriv6_dQ(double *Q, int n);

/**
 * Initializes the "R" Matrix of the JTP Derivatives
 *
 * This is a combination function that can automatically call
 * dP and dQ to just give the R matrix.
 *
 * @param R Pointer to the output "R" matrix
 * @param n The number of rows/cols of the square matrix
 */
bool initJTPDeriv6(double *R, const unsigned int n);

/**
 * Kim Derivatives Initialization
 */

/**
 * Initializes the "P" Matrix of the Kim Derivatives
 *
 * NOTE: this method is depreciated in favor of initializeKim4PQ
 *
 * @param P Pointer to the output "P" matrix
 * @param n The number of rows/cols of the square matrix
 */
void KimDeriv4_dP(double *P, int n);

/**
 * Initializes the "Q" Matrix of the Kim Derivatives
 *
 * NOTE: this method is depreciated in favor of initializeKim4PQ
 *
 * @param Q Pointer to the output "Q" matrix
 * @param n The number of rows/cols of the square matrix
 */
void KimDeriv4_dQ(double *Q, int n);

/**
 * Initializes the "R" Matrix of the Kim Derivatives
 *
 * NOTE: this method is depreciated in favor of initializeKim4PQ
 *
 * @param R Pointer to the output "R" matrix
 * @param n The number of rows/cols of the square matrix
 */
bool initKimDeriv4(double *R, const unsigned int n);

/**
 * Initializes the "RF" Matrix of the Kim Filter
 *
 * NOTE: this method is depreciated in favor of initializeKim6FilterPQ
 *
 * @param RF Pointer to the output "RF" matrix
 * @param n The number of rows/cols of the square matrix
 */
bool initKim_Filter_Deriv4(double *RF, const unsigned int n);
