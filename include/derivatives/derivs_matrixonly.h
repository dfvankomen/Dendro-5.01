
#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"

namespace dendroderivs {

template <unsigned int DerivOrder>
class MatrixCompactDerivs : public CompactDerivs {
   protected:
    std::vector<double> D_;  // D = P^{-1} * Q for derivative
    std::vector<double> workspace_;

    // interior and bounded entries for each P and Q matrix
    MatrixDiagonalEntries *diagEntries = nullptr;

   public:
    MatrixCompactDerivs(unsigned int n, unsigned int pw)
        : CompactDerivs{n, pw} {
        unsigned int nsq = n * n;  // n squared
        D_               = std::vector<double>(n * n, 0.0);
        workspace_       = std::vector<double>(n * n * n, 0.0);
    }
    /**
     * we implement a copy constructor just to print when it's called;
     * we would like to avoid accidental shallow copies
     */
    MatrixCompactDerivs(const MatrixCompactDerivs &obj) : CompactDerivs(obj) {
#ifdef DEBUG
        std::cout << "[copy constructor for MatrixCompactDerivs was called!\n"
                  << "this is a mistake as there is no implementation]"
                  << std::endl;
#endif
    };
    virtual ~MatrixCompactDerivs() {
#ifdef DEBUG
        std::cout << "in MatrixCompactDerivs deconstructor" << std::endl;
#endif
        // if (D_ != nullptr) delete[] D_;
        if (diagEntries != nullptr) delete diagEntries;
    }

    /**
     * @brief Pure virtual function to calculate the derivative.
     * @param du Pointer to where calculated derivative will be stored.
     * @param u Pointer to the input array.
     * @param dx The step size or grid spacing.
     * @param sz The number of points expected to be calculated in each dim
     * @param bflag The boundary flag
     */
    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        if constexpr (DerivOrder == 1) {
            matmul_x_dim(D_.data(), du, u, 1.0 / dx, sz);
        } else {
            matmul_x_dim(D_.data(), du, u, 1.0 / (dx * dx), sz);
        }
    }

    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        if constexpr (DerivOrder == 1) {
            matmul_y_dim(D_.data(), du, u, 1.0 / dx, sz, workspace_.data());
        } else {
            matmul_y_dim(D_.data(), du, u, 1.0 / (dx * dx), sz,
                         workspace_.data());
        }
    }

    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        if constexpr (DerivOrder == 1) {
            matmul_z_dim(D_.data(), du, u, 1.0 / dx, sz, workspace_.data());
        } else {
            matmul_z_dim(D_.data(), du, u, 1.0 / (dx * dx), sz,
                         workspace_.data());
        }
    }

    inline const double *getD() const { return D_.data(); }

    void init();
};

}  // namespace dendroderivs
