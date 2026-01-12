
#pragma once

#include <stdexcept>
#include <unordered_map>

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"

// in-matrix filter options
#include "derivatives/filt_inmat.h"
#include "derivatives/filt_inmat_byufilter.h"
#include "derivatives/filt_inmat_kim.h"

#define DDERIVS_MAX_BLOCKS_INIT 4

namespace dendroderivs {

inline std::unique_ptr<InMatrixFilter> createInMatrixFilterByType(
    const std::string &in_matrix_filter,
    const std::vector<double> &in_matrix_filter_coeffs) {
    if (in_matrix_filter == "none") {
        return std::make_unique<NoneFilter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "BYUT4") {
        return std::make_unique<BYUT4Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "BYUT6") {
        return std::make_unique<BYUT6Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "BYUT8") {
        return std::make_unique<BYUT8Filter_InMatrix>(in_matrix_filter_coeffs);
    }else if (in_matrix_filter == "KIM") {
        return std::make_unique<KimFilter_InMatrix>(in_matrix_filter_coeffs);
    }  else if (in_matrix_filter == "KIM_1_P6") {
    return std::make_unique<Kim1P6Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_2_P6") {
        return std::make_unique<Kim2P6Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_3_P6") {
        return std::make_unique<Kim3P6Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_4_P6") {
        return std::make_unique<Kim4P6Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_P6") {
        return std::make_unique<KimP6Filter_InMatrix>(in_matrix_filter_coeffs);
    }else if (in_matrix_filter == "A4") {
        return std::make_unique<A4_Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_06_P6") {
        return std::make_unique<Kim_06_P6_Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_075_P6") {
        return std::make_unique<Kim_075_P6_Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_08_P6") {
        return std::make_unique<Kim_08_P6_Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_085_P6") {
        return std::make_unique<Kim_085_P6_Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_09_P6") {
        return std::make_unique<Kim_09_P6_Filter_InMatrix>(in_matrix_filter_coeffs);
    } else if (in_matrix_filter == "KIM_09_P2") {
        return std::make_unique<Kim_09_P2_Filter_InMatrix>(in_matrix_filter_coeffs);
    }else if (in_matrix_filter == "KIM_08_P2") {
        return std::make_unique<Kim_08_P2_Filter_InMatrix>(in_matrix_filter_coeffs);
    }
    else {
        throw std::invalid_argument("Unsupported 'In-Matrix' Filter Type!");
    }
}

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
    unsigned int workspace_tot_;

    std::unordered_map<unsigned int, std::unique_ptr<DerivMatrixStorage>>
        D_storage_map_;

    // interior and bounded entries for each P and Q matrix
    MatrixDiagonalEntries *diagEntries = nullptr;

    std::unique_ptr<InMatrixFilter> in_matrix_filter_;

   public:
    MatrixCompactDerivs(unsigned int ele_order,
                        const std::string &in_matrix_filter = "none",
                        const std::vector<double> &in_matrix_filter_coeffs =
                            std::vector<double>())
        : CompactDerivs{ele_order} {
        // establish workspace to be as large as our largest
        unsigned int workspace_size_calc = p_n * p_n * p_n * 2;
        workspace_        = std::vector<double>(workspace_size_calc, 0.0);
        workspace_tot_    = workspace_size_calc;

        // then call and build up the in_matrix_filter_
        in_matrix_filter_ = createInMatrixFilterByType(in_matrix_filter,
                                                       in_matrix_filter_coeffs);
    }

    ~MatrixCompactDerivs() {
        // make sure diagEntries is properly deleted to avoid memory leak
        delete diagEntries;
    }

    void set_maximum_block_size(size_t block_size) {
        workspace_tot_ = block_size * 2;
        workspace_.resize(workspace_tot_);
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
        // delete diagEntries;
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

            // then get it
            D_use =
                get_deriv_mat_by_bflag_x(D_storage_map_[sz[0]].get(), bflag);
        }

        // NOTE: it is now required that the workspace size be set **properly**
        // by the user by setting derivs' max_block_size

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

            // then get it
            D_use =
                get_deriv_mat_by_bflag_y(D_storage_map_[sz[1]].get(), bflag);
        }

        // NOTE: it is now required that the workspace size be set **properly**
        // by the user by setting derivs' max_block_size

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

            // then get it
            D_use =
                get_deriv_mat_by_bflag_z(D_storage_map_[sz[2]].get(), bflag);
        }

        // NOTE: it is now required that the workspace size be set **properly**
        // by the user by setting derivs' max_block_size

        // std::cout << "workspace_tot is: " << workspace_tot_ << std::endl;
        // std::cout << "sz is: " << sz[0] << ", " << sz[1] << ", " << sz[2]
        //           << std::endl;
        // std::cout << "worksapce_ size is: " << workspace_.size() <<
        // std::endl;

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
