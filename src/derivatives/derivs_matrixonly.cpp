#include "derivatives/derivs_matrixonly.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "derivatives.h"
#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"
#include "lapac.h"
#include "mathUtils.h"

#define _DENDRODERIV_USE_INV_METHOD 0

namespace dendroderivs {

// TODO: implement do_grad_x, y, and z

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage> createMatrixSystemForSingleSize(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries* diagEntries, const bool skip_leftright) {
    const float Q_parity        = DerivOrder == 2 ? 1.0 : -1.0;

    const double nsq            = n * n;

    // create the DerivMatrixStorage object
    auto derivMatrixPtr         = std::make_unique<DerivMatrixStorage>();
    // allocate the values
    derivMatrixPtr->D_original  = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_left      = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_right     = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_leftright = std::vector<double>(nsq, 0.0);

    for (BoundaryType b :
         {BoundaryType::NO_BOUNDARY, BoundaryType::LEFT_BOUNDARY,
          BoundaryType::RIGHT_BOUNDARY, BoundaryType::LEFTRIGHT_BOUNDARY}) {
        // FIXME: this is an interesting problem, we have to assume that Dendro
        // blocks will only potentially have LEFTRIGHT boundary in weird
        // instances where the mesh is super unrefined or we have more than 1
        // block "fused" together. So, there's really only a need to compute
        // LEFTRIGHT for blocks > 1, but it's still so rare. Left-right
        // corresponds to the cases where the boundary on both sides are "hard"
        // boundaries.
        if (b == BoundaryType::LEFTRIGHT_BOUNDARY && skip_leftright) continue;

        // std::cout << "Creating for b=" << b << std::endl;

        unsigned int boundary_top    = 0;
        unsigned int boundary_bottom = 0;

        if (b == BoundaryType::LEFT_BOUNDARY ||
            b == BoundaryType::LEFTRIGHT_BOUNDARY) {
            boundary_top = pw;
        }
        if (b == BoundaryType::RIGHT_BOUNDARY ||
            b == BoundaryType::LEFTRIGHT_BOUNDARY) {
            boundary_bottom = pw;
        }

        // build up P_ and Q_
        std::vector<double> P_temp = create_P_from_diagonals(
            *diagEntries, n, 1.0, boundary_top, boundary_bottom);
        std::vector<double> Q_temp = create_Q_from_diagonals(
            *diagEntries, n, Q_parity, boundary_top, boundary_bottom);

        std::vector<double>* const D_ptr =
            get_deriv_mat_by_boundary(derivMatrixPtr.get(), b);

        // then we solve it
        if constexpr (_DENDRODERIV_USE_INV_METHOD) {
            // ORIGINAL solution
            // we want to solve:
            // P f' = Q f
            // this means that we need to do:
            // P-1 P f' = P-1 Q f
            // which gives us: f' = D f, so D = P-1 Q

            // start by copying P to Pinv
            std::vector<double> Pinv = P_temp;

            // compute the inverse
            lapack::iterative_inverse(P_temp.data(), Pinv.data(), n);

            // perform the norm? could be useful for if we want to make sure
            // we're within -1 and 1 for stability? for (size_t i = 0; i < p_n *
            // p_n; i++) {
            //     Pinv[i] *= norm;
            // }

            // then do matrix multiplication to get D_
            lapack::square_matrix_multiplication(Pinv.data(), Q_temp.data(),
                                                 D_ptr->data(), n);

        } else {
            // ALTERNATE SOLUTION
            // use the "solution" routine of BLAS to solve for D, when we
            // consider that PD = Q Pf' = Qf => D = P-1 Q and P P-1 Q = Q, so ,
            // P (P-1 Q) = Q, so we can solve P D = Q

            int info = 0;
            lapack::lapack_DGESV_T(n, n, P_temp.data(), n, Q_temp.data(),
                                   D_ptr->data(), n, info);
            // this should directly solve for the matrix inverse
        }

        // std::cout << "P is: " << std::endl;
        // printArray_2D_transpose(P_temp.data(), n, n);
        // std::cout << "Q is: " << std::endl;
        // printArray_2D_transpose(Q_temp.data(), n, n);
        // std::cout << "D is: " << std::endl;
        // printArray_2D_transpose(D_ptr->data(), n, n);
    }

    return derivMatrixPtr;
}

template <unsigned int DerivOrder>
void MatrixCompactDerivs<DerivOrder>::init() {
    // so we need to create for a specific number of blocks

    for (unsigned int i = 1; i <= DDERIVS_MAX_BLOCKS_INIT; i++) {
        // calculate the size based on the element order
        const unsigned int n = (i + 1) * p_ele_order + 1;

        // std::cout << "Creating for n blocks: " << i
        //           << " , which is of size: " << n << std::endl;

        // because we're using smart pointers, just store the result immediately
        // std::cout << " n = " << n << " and i==1:" << (i == 1) << std::endl;
        D_storage_map_.emplace(n, createMatrixSystemForSingleSize<DerivOrder>(
                                      p_pw, n, diagEntries, i == 1));
    }
}

template void MatrixCompactDerivs<1>::init();
template void MatrixCompactDerivs<2>::init();

}  // namespace dendroderivs
