#pragma once

#include "derivatives.h"

/**
 * Shorthand for the index of entry (i, j) in an nxn
 *  matrix, when stored in a single array.
 * In this setup, we store the matrix column by column.
 * Indices start at zero.
 */
#define IDXN(i, j, n) ((i) + (n) * (j))

namespace dendroderivs {

/**
 * struct containing the information needed to create
 *  P and Q matrices using CompactDerivs::buildMatrix
 */
struct MatrixDiagonalEntries {
    std::vector<double> PDiagInterior;
    std::vector<std::vector<double>> PDiagBoundary;
    std::vector<double> QDiagInterior;
    std::vector<std::vector<double>> QDiagBoundary;

    void toStdOut() {
        std::cout << "PDiagInterior: ";
        for (auto &a : PDiagInterior) {
            std::cout << a << " ";
        }
        std::cout << std::endl << "PDiagBoundary: ";
        for (auto &a : PDiagBoundary) {
            std::cout << std::endl;
            for (auto &b : a) {
                std::cout << b << " ";
            }
        }
        std::cout << std::endl << "QDiagInterior: ";
        for (auto &a : QDiagInterior) {
            std::cout << a << " ";
        }
        std::cout << std::endl << "QDiagBoundary: ";
        for (auto &a : QDiagBoundary) {
            std::cout << std::endl;
            for (auto &b : a) {
                std::cout << b << " ";
            }
        }
        std::cout << std::endl;
    }
};

/**
 * @brief Check to see if the size of the matrix is sufficient for boundaries
 */
inline bool check_size_and_boundary_terms(
    const std::vector<std::vector<double>> &diag_boundary, const unsigned n) {
    for (const auto &interior : diag_boundary) {
        if (interior.size() > n) return false;
    }

    return true;
}

/**
 * @brief Convert a diagonal boundary and interior into a col-major matrix
 */
std::vector<double> createMatrix(
    const std::vector<std::vector<double>> &diag_boundary,
    const std::vector<double> &diag_interior, const unsigned int n,
    const double parity, const unsigned int boundary_top = 0,
    const unsigned int boundary_bottom = 0);

/**
 * @brief Create just the P matrix from the diagonal object
 */
std::vector<double> create_P_from_diagonals(
    const MatrixDiagonalEntries &matrixDiagonals, const unsigned int n,
    const double parity, const unsigned int boundary_top = 0,
    const unsigned int boundary_bottom = 0);

/**
 * @brief Create just the Q matrix from the diagonal object
 */
std::vector<double> create_Q_from_diagonals(
    const MatrixDiagonalEntries &matrixDiagonals, const unsigned int n,
    const double parity = -1.0, const unsigned int boundary_top = 0,
    const unsigned int boundary_bottom = 0);

// this class works for both first and second order derivatives
class CompactDerivs : public Derivs {
   protected:
    MatrixDiagonalEntries *diagEntries;

    // matrices
    std::vector<double> P_;  ///< P matrix deriv
    std::vector<double> Q_;  ///< Q matrix deriv

    // protected as this class should ONLY be inherited, never instantiated!
    CompactDerivs(unsigned int ele_order) : Derivs{ele_order} {}

    /**
     * we implement a copy constructor just to print when it's called;
     * we would like to avoid accidental shallow copies
     */
    CompactDerivs(const CompactDerivs &obj) : Derivs(obj) {
#ifdef DEBUG
        std::cout << "[copy constructor for CompactDerivs was called!\n"
                  << "this is a mistake as there is no implementation]"
                  << std::endl;
#endif
    };

    // if you want to access this outside of class implementations, consider
    // moving to util.h/util.cpp
    void buildMatrix(double *M, std::vector<double> &diag,
                     std::vector<std::vector<double>> &bound, double parity,
                     unsigned int n);

   public:
    /**
     * nothing to delete, I think
     * (defined outside of class declaration so GCC will make the vtable)
     */
    virtual ~CompactDerivs();

    /**
     * @brief Pure virtual function to calculate the derivative.
     * @param du Pointer to where calculated derivative will be stored.
     * @param u Pointer to the input array.
     * @param dx The step size or grid spacing.
     * @param sz The number of points expected to be calculated in each dim
     * @param bflag The boundary flag
     */
    virtual void do_grad_x(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;
    virtual void do_grad_y(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;
    virtual void do_grad_z(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;

    inline const double *getP() const { return P_.data(); }
    inline const double *getQ() const { return Q_.data(); }
};

}  // namespace dendroderivs
