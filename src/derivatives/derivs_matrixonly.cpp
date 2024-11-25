#include "derivatives/derivs_matrixonly.h"

#include <cstdint>
#include <stdexcept>

#include "derivatives.h"
#include "derivatives/derivs_utils.h"
#include "lapac.h"
#include "mathUtils.h"
#include "refel.h"

#define _DENDRODERIV_USE_INV_METHOD 0

namespace dendroderivs {

// TODO: implement do_grad_x, y, and z

template <unsigned int DerivOrder>
void MatrixCompactDerivs<DerivOrder>::init() {
    // P and Q should be initialized, now we have to invert Q and then multiply
    // with P

    // build the P and Q matrices

    const float Q_parity =
        this->getDerivOrder() == DerivOrder::D_SECOND_ORDER ? 1.0 : -1.0;

    for (BoundaryType b :
         {BoundaryType::NO_BOUNDARY, BoundaryType::LEFT_BOUNDARY,
          BoundaryType::RIGHT_BOUNDARY, BoundaryType::LEFTRIGHT_BOUNDARY}) {
        unsigned int boundary_top    = 0;
        unsigned int boundary_bottom = 0;

        if (b == BoundaryType::LEFT_BOUNDARY ||
            b == BoundaryType::LEFTRIGHT_BOUNDARY) {
            boundary_top = p_pw;
        }
        if (b == BoundaryType::RIGHT_BOUNDARY ||
            b == BoundaryType::LEFTRIGHT_BOUNDARY) {
            boundary_bottom = p_pw;
        }

        // build up P_ and Q_
        P_ = create_P_from_diagonals(*diagEntries, p_n, 1.0, boundary_top,
                                     boundary_bottom);
        Q_ = create_Q_from_diagonals(*diagEntries, p_n, Q_parity, boundary_top,
                                     boundary_bottom);

        std::vector<double>* const D_ptr =
            get_deriv_mat_by_boundary(D_storage, b);

        // then we solve it
        if constexpr (_DENDRODERIV_USE_INV_METHOD) {
            // ORIGINAL solution
            // we want to solve:
            // P f' = Q f
            // this means that we need to do:
            // P-1 P f' = P-1 Q f
            // which gives us: f' = D f, so D = P-1 Q

            // start by copying P to Pinv
            std::vector<double> Pinv = P_;

            // compute the inverse
            lapack::iterative_inverse(P_.data(), Pinv.data(), p_n);

            // perform the norm? could be useful for if we want to make sure
            // we're within -1 and 1 for stability? for (size_t i = 0; i < p_n *
            // p_n; i++) {
            //     Pinv[i] *= norm;
            // }

            // then do matrix multiplication to get D_
            lapack::square_matrix_multiplication(Pinv.data(), Q_.data(),
                                                 D_ptr->data(), p_n);

        } else {
            // ALTERNATE SOLUTION
            // use the "solution" routine of BLAS to solve for D, when we
            // consider that PD = Q Pf' = Qf => D = P-1 Q and P P-1 Q = Q, so ,
            // P (P-1 Q) = Q, so we can solve P D = Q

            int info = 0;
            lapack::lapack_DGESV_T(p_n, p_n, P_.data(), p_n, Q_.data(),
                                   D_ptr->data(), p_n, info);
            // this should directly solve for the matrix inverse
        }
    }
}

template void MatrixCompactDerivs<1>::init();
template void MatrixCompactDerivs<2>::init();

}  // namespace dendroderivs
