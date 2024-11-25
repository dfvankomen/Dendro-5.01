
#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"

namespace dendroderivs {

template <unsigned int DerivOrder>
class MatrixCompactDerivs : public CompactDerivs {
   protected:
    std::vector<double> D_;  // D = P^{-1} * Q for derivative
    DerivMatrixStorage *D_storage = nullptr;
    std::vector<double> workspace_;

    // interior and bounded entries for each P and Q matrix
    MatrixDiagonalEntries *diagEntries = nullptr;

   public:
    MatrixCompactDerivs(unsigned int ele_order) : CompactDerivs{ele_order} {
        unsigned int nsq       = p_n * p_n;  // n squared
        D_                     = std::vector<double>(nsq, 0.0);
        workspace_             = std::vector<double>(nsq * p_n, 0.0);
        D_storage              = new DerivMatrixStorage;

        // then allocate the DerivMatrixStorage
        D_storage->D_original  = std::vector<double>(nsq, 0.0);
        D_storage->D_left      = std::vector<double>(nsq, 0.0);
        D_storage->D_right     = std::vector<double>(nsq, 0.0);
        D_storage->D_leftright = std::vector<double>(nsq, 0.0);
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
        delete diagEntries;
        delete D_storage;
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
        const std::vector<double> *D_use =
            get_deriv_mat_by_bflag_x(D_storage, bflag);

        if constexpr (DerivOrder == 1) {
            matmul_x_dim(D_use->data(), du, u, 1.0 / dx, sz);
        } else {
            matmul_x_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz);
        }
    }

    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        const std::vector<double> *D_use =
            get_deriv_mat_by_bflag_x(D_storage, bflag);

        if constexpr (DerivOrder == 1) {
            matmul_y_dim(D_use->data(), du, u, 1.0 / dx, sz, workspace_.data());
        } else {
            matmul_y_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz,
                         workspace_.data());
        }
    }

    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        const std::vector<double> *D_use =
            get_deriv_mat_by_bflag_x(D_storage, bflag);

        if constexpr (DerivOrder == 1) {
            matmul_z_dim(D_use->data(), du, u, 1.0 / dx, sz, workspace_.data());
        } else {
            matmul_z_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz,
                         workspace_.data());
        }
    }

    inline const double *getD() const { return D_.data(); }

    void init();
};

}  // namespace dendroderivs
