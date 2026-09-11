#include "derivatives/derivs_matrixonly.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "derivatives.h"
#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"
#include "derivatives/filt_inmat.h"
#include "lapac.h"
#include "mathUtils.h"
#include "refel.h"

#define _DENDRODERIV_USE_INV_METHOD 0

namespace dendroderivs {

#ifdef DENDRO_WIDE_PADDING
namespace {
/**
 * DENDRO_WIDE_PADDING: the five "fine face" variants of D. A face that abuts
 * a finer octant has its outermost DENDRO_WIDE_PADDING_EXTRA padding rows
 * unfilled, so D is built with that many rows/cols trimmed at that end
 * (identity there, same mechanism as the physical-boundary trim of pw rows).
 * `solve(top, bottom, side, D)` builds one variant; the caller supplies it so
 * the plain, in-matrix-filter and per-side-diagonal builders each keep their
 * own arithmetic. Any variant that does not fit marks fine_valid = false and
 * the dispatch throws if it is ever requested.
 */
template <typename SolveFn>
void build_fine_face_variants(DerivMatrixStorage& st, const unsigned int pw,
                              const unsigned int n, SolveFn&& solve) {
    const unsigned int ft = DENDRO_WIDE_PADDING_EXTRA;  // trim at a fine face
    const double nsq      = n * n;
    struct V {
        std::vector<double>* D;
        unsigned int top, bottom;
        int side;  // per-side diagonal set for the physical end: 0 none,
                   // 1 left is physical, 2 right is physical
    };
    st.D_fine_left           = std::vector<double>(nsq, 0.0);
    st.D_fine_right          = std::vector<double>(nsq, 0.0);
    st.D_fine_leftright      = std::vector<double>(nsq, 0.0);
    st.D_fine_left_bdy_right = std::vector<double>(nsq, 0.0);
    st.D_bdy_left_fine_right = std::vector<double>(nsq, 0.0);
    const V variants[]       = {{&st.D_fine_left, ft, 0, 0},
                                {&st.D_fine_right, 0, ft, 0},
                                {&st.D_fine_leftright, ft, ft, 0},
                                {&st.D_fine_left_bdy_right, ft, pw, 2},
                                {&st.D_bdy_left_fine_right, pw, ft, 1}};
    st.fine_valid            = true;
    for (const V& v : variants) {
        try {
            solve(v.top, v.bottom, v.side, *v.D);
        } catch (const std::exception&) {
            st.fine_valid = false;
            return;
        }
    }
}
}  // namespace
#endif

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
        // No caller passes true; skipping only ever returned silent zeros.
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

        // build the local P and Q matrices from diagonals
        std::vector<double> P_temp, Q_temp;
        try {
            P_temp = create_P_from_diagonals(*diagEntries, n, 1.0, boundary_top,
                                             boundary_bottom);
            Q_temp = create_Q_from_diagonals(*diagEntries, n, Q_parity,
                                             boundary_top, boundary_bottom);
        } catch (const std::exception&) {
            // Wide closures may not fit once both ends are trimmed.
            if (b != BoundaryType::LEFTRIGHT_BOUNDARY) throw;
            derivMatrixPtr->leftright_valid = false;
            continue;
        }

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
    }

#ifdef DENDRO_WIDE_PADDING
    build_fine_face_variants(
        *derivMatrixPtr, pw, n,
        [&](unsigned int top, unsigned int bottom, int /*side*/,
            std::vector<double>& D) {
            std::vector<double> P_t =
                create_P_from_diagonals(*diagEntries, n, 1.0, top, bottom);
            std::vector<double> Q_t = create_Q_from_diagonals(
                *diagEntries, n, Q_parity, top, bottom);
            int info = 0;
            lapack::lapack_DGESV_T(n, n, P_t.data(), n, Q_t.data(), D.data(),
                                   n, info);
        });
#endif

    return derivMatrixPtr;
}

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage>
createMatrixSystemForSingleSizeInMatrixFilter(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries* diagEntries,
    const MatrixDiagonalEntries* filterEntries, const bool skip_leftright,
    const InMatFilterType filt_type) {
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
        // No caller passes true; skipping only ever returned silent zeros.
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

        // build the local P and Q matrices from diagonals
        std::vector<double> P_temp, Q_temp, R_temp, S_temp;
        try {
            P_temp = create_P_from_diagonals(*diagEntries, n, 1.0, boundary_top,
                                             boundary_bottom);
            Q_temp = create_Q_from_diagonals(*diagEntries, n, Q_parity,
                                             boundary_top, boundary_bottom);
            R_temp = create_P_from_diagonals(*filterEntries, n, 1.0,
                                             boundary_top, boundary_bottom);
            S_temp = create_Q_from_diagonals(*filterEntries, n, 1.0,
                                             boundary_top, boundary_bottom);
        } catch (const std::exception&) {
            // Wide closures may not fit once both ends are trimmed.
            if (b != BoundaryType::LEFTRIGHT_BOUNDARY) throw;
            derivMatrixPtr->leftright_valid = false;
            continue;
        }

        std::vector<double>* const D_ptr =
            get_deriv_mat_by_boundary(derivMatrixPtr.get(), b);
        // start by copying P to Pinv
        std::vector<double> Pinv = P_temp;
        std::vector<double> Rinv = R_temp;
        std::vector<double> R1_S = R_temp;
        std::vector<double> QRS  = R_temp;

        // compute the inverse
        lapack::iterative_inverse(R_temp.data(), Rinv.data(), n);
        lapack::square_matrix_multiplication(Rinv.data(), S_temp.data(),
                                             R1_S.data(), n);
if (filt_type == InMatFilterType::IMFT_KIM ||
    filt_type == InMatFilterType::IMFT_KIM_1_P6 ||
    filt_type == InMatFilterType::IMFT_KIM_2_P6 ||
    filt_type == InMatFilterType::IMFT_KIM_3_P6 ||
    filt_type == InMatFilterType::IMFT_KIM_4_P6 ||
    filt_type == InMatFilterType::IMFT_A4 ||
    filt_type == InMatFilterType::IMFT_KIM_P6 ||
    filt_type == InMatFilterType::IMFT_Kim_06_P6 ||
    filt_type == InMatFilterType::IMFT_Kim_075_P6 ||
    filt_type == InMatFilterType::IMFT_Kim_08_P6 ||
    filt_type == InMatFilterType::IMFT_Kim_085_P6 ||
    filt_type == InMatFilterType::IMFT_Kim_09_P6||
    filt_type == InMatFilterType::IMFT_Kim_09_P2||
filt_type == InMatFilterType::IMFT_Kim_08_P2)
            {
            for (size_t idx = 0; idx < n; ++idx) {
                R1_S[n * idx + idx] += 1.0;
            }
            }

        lapack::square_matrix_multiplication(Q_temp.data(), R1_S.data(),
                                             QRS.data(), n);
        lapack::iterative_inverse(P_temp.data(), Pinv.data(), n);

        lapack::square_matrix_multiplication(Pinv.data(), QRS.data(),
                                             D_ptr->data(), n);
    }

#ifdef DENDRO_WIDE_PADDING
    build_fine_face_variants(
        *derivMatrixPtr, pw, n,
        [&](unsigned int top, unsigned int bottom, int /*side*/,
            std::vector<double>& D) {
            std::vector<double> P_t =
                create_P_from_diagonals(*diagEntries, n, 1.0, top, bottom);
            std::vector<double> Q_t = create_Q_from_diagonals(
                *diagEntries, n, Q_parity, top, bottom);
            std::vector<double> R_t =
                create_P_from_diagonals(*filterEntries, n, 1.0, top, bottom);
            std::vector<double> S_t =
                create_Q_from_diagonals(*filterEntries, n, 1.0, top, bottom);
            std::vector<double> Pinv = P_t, Rinv = R_t, R1_S = R_t, QRS = R_t;
            lapack::iterative_inverse(R_t.data(), Rinv.data(), n);
            lapack::square_matrix_multiplication(Rinv.data(), S_t.data(),
                                                 R1_S.data(), n);
            if (filt_type == InMatFilterType::IMFT_KIM ||
                filt_type == InMatFilterType::IMFT_KIM_1_P6 ||
                filt_type == InMatFilterType::IMFT_KIM_2_P6 ||
                filt_type == InMatFilterType::IMFT_KIM_3_P6 ||
                filt_type == InMatFilterType::IMFT_KIM_4_P6 ||
                filt_type == InMatFilterType::IMFT_A4 ||
                filt_type == InMatFilterType::IMFT_KIM_P6 ||
                filt_type == InMatFilterType::IMFT_Kim_06_P6 ||
                filt_type == InMatFilterType::IMFT_Kim_075_P6 ||
                filt_type == InMatFilterType::IMFT_Kim_08_P6 ||
                filt_type == InMatFilterType::IMFT_Kim_085_P6 ||
                filt_type == InMatFilterType::IMFT_Kim_09_P6 ||
                filt_type == InMatFilterType::IMFT_Kim_09_P2 ||
                filt_type == InMatFilterType::IMFT_Kim_08_P2) {
                for (size_t idx = 0; idx < n; ++idx) R1_S[n * idx + idx] += 1.0;
            }
            lapack::square_matrix_multiplication(Q_t.data(), R1_S.data(),
                                                 QRS.data(), n);
            lapack::iterative_inverse(P_t.data(), Pinv.data(), n);
            lapack::square_matrix_multiplication(Pinv.data(), QRS.data(),
                                                 D.data(), n);
        });
#endif
    return derivMatrixPtr;
}

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage>
createMatrixSystemForSingleSizeAllUniqueDiags(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries* diagEntries,
    const MatrixDiagonalEntries* diagEntriesLeft,
    const MatrixDiagonalEntries* diagEntriesRight,
    const MatrixDiagonalEntries* diagEntriesLeftRight,
    const bool skip_leftright) {
    const float Q_parity        = DerivOrder == 2 ? 1.0 : -1.0;

    const double nsq            = n * n;

    // create the DerivMatrixStorage object
    auto derivMatrixPtr         = std::make_unique<DerivMatrixStorage>();
    // allocate the values
    derivMatrixPtr->D_original  = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_left      = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_right     = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_leftright = std::vector<double>(nsq, 0.0);

    const MatrixDiagonalEntries* tempDiagEntries = nullptr;

    for (BoundaryType b :
         {BoundaryType::NO_BOUNDARY, BoundaryType::LEFT_BOUNDARY,
          BoundaryType::RIGHT_BOUNDARY, BoundaryType::LEFTRIGHT_BOUNDARY}) {
        // No caller passes true; skipping only ever returned silent zeros.
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

        if (b == BoundaryType::LEFT_BOUNDARY) {
            tempDiagEntries = diagEntriesLeft;
        } else if (b == BoundaryType::RIGHT_BOUNDARY) {
            tempDiagEntries = diagEntriesRight;
        } else if (b == BoundaryType::LEFTRIGHT_BOUNDARY) {
            tempDiagEntries = diagEntriesLeftRight;
        } else {
            tempDiagEntries = diagEntries;
        }

        // build the local P and Q matrices from diagonals
        std::vector<double> P_temp, Q_temp;
        try {
            P_temp = create_P_from_diagonals(*tempDiagEntries, n, 1.0,
                                             boundary_top, boundary_bottom);
            Q_temp = create_Q_from_diagonals(*tempDiagEntries, n, Q_parity,
                                             boundary_top, boundary_bottom);
        } catch (const std::exception&) {
            // Wide closures may not fit once both ends are trimmed.
            if (b != BoundaryType::LEFTRIGHT_BOUNDARY) throw;
            derivMatrixPtr->leftright_valid = false;
            continue;
        }

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
    }

#ifdef DENDRO_WIDE_PADDING
    // a fine face uses the plain closures; a physical face on the other end
    // uses that side's own diagonal set, as in the stock loop above
    build_fine_face_variants(
        *derivMatrixPtr, pw, n,
        [&](unsigned int top, unsigned int bottom, int side,
            std::vector<double>& D) {
            const MatrixDiagonalEntries* de =
                (side == 1) ? diagEntriesLeft
                            : (side == 2) ? diagEntriesRight : diagEntries;
            std::vector<double> P_t =
                create_P_from_diagonals(*de, n, 1.0, top, bottom);
            std::vector<double> Q_t =
                create_Q_from_diagonals(*de, n, Q_parity, top, bottom);
            int info = 0;
            lapack::lapack_DGESV_T(n, n, P_t.data(), n, Q_t.data(), D.data(),
                                   n, info);
        });
#endif

    return derivMatrixPtr;
}

template std::unique_ptr<DerivMatrixStorage>
createMatrixSystemForSingleSizeAllUniqueDiags<1>(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries* diagEntries,
    const MatrixDiagonalEntries* diagEntriesLeft,
    const MatrixDiagonalEntries* diagEntriesRight,
    const MatrixDiagonalEntries* diagEntriesLeftRight,
    const bool skip_leftright);

// explicit instantiations: the header's build_storage_for_size calls these
// from other TUs, and the implicit ones inside init() can be inlined away
template std::unique_ptr<DerivMatrixStorage> createMatrixSystemForSingleSize<1>(
    const unsigned int, const unsigned int, const MatrixDiagonalEntries*,
    const bool);
template std::unique_ptr<DerivMatrixStorage> createMatrixSystemForSingleSize<2>(
    const unsigned int, const unsigned int, const MatrixDiagonalEntries*,
    const bool);
template std::unique_ptr<DerivMatrixStorage>
createMatrixSystemForSingleSizeInMatrixFilter<1>(
    const unsigned int, const unsigned int, const MatrixDiagonalEntries*,
    const MatrixDiagonalEntries*, const bool, const InMatFilterType);
template std::unique_ptr<DerivMatrixStorage>
createMatrixSystemForSingleSizeInMatrixFilter<2>(
    const unsigned int, const unsigned int, const MatrixDiagonalEntries*,
    const MatrixDiagonalEntries*, const bool, const InMatFilterType);

template <unsigned int DerivOrder>
void MatrixCompactDerivs<DerivOrder>::init() {
    // so we need to create for a specific number of blocks

    for (unsigned int i = 1; i <= DDERIVS_MAX_BLOCKS_INIT; i++) {
        // calculate the size based on the element order
#ifdef DENDRO_WIDE_PADDING
        // block of i elements plus the (wider) padding on both sides
        const unsigned int n = i * p_ele_order + 1 + 2 * p_pw;
#else
        const unsigned int n = (i + 1) * p_ele_order + 1;
#endif

        // std::cout << "Creating for n blocks: " << i
        //           << " , which is of size: " << n << std::endl;

        // because we're using smart pointers, just store the result immediately
        // the filter/scheme branch lives in build_storage_for_size so that this
        // eager path and the lazy get_storage_for_size path cannot disagree.
        // i == 1 is the single-element block, which most needs LEFTRIGHT.
        D_storage_map_.emplace(n, build_storage_for_size(n, false));
    }
}

template void MatrixCompactDerivs<1>::init();
template void MatrixCompactDerivs<2>::init();

}  // namespace dendroderivs
