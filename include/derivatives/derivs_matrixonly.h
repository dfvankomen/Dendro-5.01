
#pragma once

#include <unordered_map>

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"

#define DDERIVS_MAX_BLOCKS_INIT 4

namespace dendroderivs {

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage> createMatrixSystemForSingleSize(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries *diagEntries,
    const bool skip_leftright = false);

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage>
createMatrixSystemForSingleSizeAllUniqueDiags(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries *diagEntries,
    const MatrixDiagonalEntries *diagEntriesLeft,
    const MatrixDiagonalEntries *diagEntriesRight,
    const MatrixDiagonalEntries *diagEntriesLeftRight,
    const bool skip_leftright = false);

template <unsigned int DerivOrder>
class MatrixCompactDerivs : public CompactDerivs {
   protected:
    std::vector<double> workspace_;
    unsigned int workspace_dim_;

    std::unordered_map<unsigned int, std::unique_ptr<DerivMatrixStorage>>
        D_storage_map_;

    // interior and bounded entries for each P and Q matrix
    MatrixDiagonalEntries *diagEntries = nullptr;

   public:
    MatrixCompactDerivs(unsigned int ele_order) : CompactDerivs{ele_order} {
        unsigned int nsq = p_n * p_n;  // n squared

        // establish workspace to be as large as our largest
        workspace_       = std::vector<double>(nsq * p_n, 0.0);
        workspace_dim_   = p_n;
    }
    /**
     * Based on the crazy things that are being stored across all of these
     * MatrixCompactDerivs, things need to be properly copied if it's moved!
     */
    MatrixCompactDerivs(const MatrixCompactDerivs &obj) : CompactDerivs(obj) {
#ifdef DEBUG
        std::cout << "[copy constructor for MatrixCompactDerivs was called!\n"
                  << "this is a mistake as there is no implementation]"
                  << std::endl;
#endif
        if (obj.diagEntries) {
            diagEntries = new MatrixDiagonalEntries{
                obj.diagEntries->PDiagInterior, obj.diagEntries->PDiagBoundary,
                obj.diagEntries->QDiagInterior, obj.diagEntries->QDiagBoundary};
        } else {
            diagEntries = nullptr;
        }

        // make sure to copy over workspace
        workspace_ = obj.workspace_;

        // and then make sure to copy over D_storage_map
        for (const auto &pair : obj.D_storage_map_) {
            D_storage_map_[pair.first] =
                pair.second ? std::make_unique<DerivMatrixStorage>(*pair.second)
                            : nullptr;
        }
#ifdef DEBUG
        std::cout << "in MatrixCompactDerivs deconstructor" << std::endl;
#endif
        // if (D_ != nullptr) delete[] D_;
        delete diagEntries;
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
        // get a pointer to the vector we're using
        std::vector<double> *D_use;

        // check to make sure that the data is available
        if (D_storage_map_.find(sz[0]) != D_storage_map_.end()) {
            // size exists
            D_use =
                get_deriv_mat_by_bflag_x(D_storage_map_[sz[0]].get(), bflag);
        } else {
#ifdef DEBUG_MODE
            std::cout << "[matonly:grad_x]: Matrix Size for " << sz[0]
                      << " DOES NOT EXIST for deriv " << this->toString()
                      << ", creating..." << std::endl;
#endif
            // if it isn't, then we just have to create a new one for this size
            D_storage_map_.emplace(sz[0],
                                   createMatrixSystemForSingleSize<DerivOrder>(
                                       p_pw, sz[0], diagEntries, false));

            // workspace_ also needs to be at least as large as this block
            // TODO: consider setting workspace_ to being like 3 * largest_dim
            if (sz[0] > workspace_dim_) {
                workspace_dim_ = sz[0];
                workspace_     = std::vector<double>(
                    workspace_dim_ * workspace_dim_ * workspace_dim_);
            }

            // then get it
            D_use =
                get_deriv_mat_by_bflag_x(D_storage_map_[sz[0]].get(), bflag);
        }

        if constexpr (DerivOrder == 1) {
            matmul_x_dim(D_use->data(), du, u, 1.0 / dx, sz, bflag);
        } else {
            matmul_x_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz, bflag);
        }
    }

    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        // get a pointer to the vector we're using
        std::vector<double> *D_use;

        // check to make sure that the data is available
        if (D_storage_map_.find(sz[1]) != D_storage_map_.end()) {
            // size exists
            D_use =
                get_deriv_mat_by_bflag_y(D_storage_map_[sz[1]].get(), bflag);
        } else {
#ifdef DEBUG_MODE
            std::cout << "[matonly:grad_y]: Matrix Size for " << sz[1]
                      << " DOES NOT EXIST for deriv " << this->toString()
                      << ", creating..." << std::endl;
#endif
            // if it isn't, then we just have to create a new one for this size
            D_storage_map_.emplace(sz[1],
                                   createMatrixSystemForSingleSize<DerivOrder>(
                                       p_pw, sz[1], diagEntries, false));

            if (sz[1] > workspace_dim_) {
                workspace_dim_ = sz[1];
                workspace_     = std::vector<double>(
                    workspace_dim_ * workspace_dim_ * workspace_dim_);
            }

            // then get it
            D_use =
                get_deriv_mat_by_bflag_y(D_storage_map_[sz[1]].get(), bflag);
        }

        if constexpr (DerivOrder == 1) {
            matmul_y_dim(D_use->data(), du, u, 1.0 / dx, sz, workspace_.data(),
                         bflag);
        } else {
            matmul_y_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz,
                         workspace_.data(), bflag);
        }
    }

    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        // get a pointer to the vector we're using
        std::vector<double> *D_use;

        // check to make sure that the data is available
        if (D_storage_map_.find(sz[2]) != D_storage_map_.end()) {
            // size exists
            D_use =
                get_deriv_mat_by_bflag_z(D_storage_map_[sz[2]].get(), bflag);
        } else {
#ifdef DEBUG_MODE
            std::cout << "[matonly:grad_z]: Matrix Size for " << sz[2]
                      << " DOES NOT EXIST for deriv " << this->toString()
                      << ", creating..." << std::endl;
#endif
            // if it isn't, then we just have to create a new one for this size
            D_storage_map_.emplace(sz[2],
                                   createMatrixSystemForSingleSize<DerivOrder>(
                                       p_pw, sz[2], diagEntries, false));

            if (sz[2] > workspace_dim_) {
                workspace_dim_ = sz[1];
                workspace_     = std::vector<double>(
                    workspace_dim_ * workspace_dim_ * workspace_dim_);
            }

            // then get it
            D_use =
                get_deriv_mat_by_bflag_z(D_storage_map_[sz[2]].get(), bflag);
        }

        if constexpr (DerivOrder == 1) {
            matmul_z_dim(D_use->data(), du, u, 1.0 / dx, sz, workspace_.data(),
                         bflag);
        } else {
            matmul_z_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz,
                         workspace_.data(), bflag);
        }
    }

    void init();
};

}  // namespace dendroderivs
