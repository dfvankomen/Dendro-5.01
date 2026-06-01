#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"

namespace dendroderivs {

/*
 * BandedCompactDerivs: implements compact FD via per-block LAPACK banded
 * solve (dgbsvx) rather than the pre-inverted GEMM path.
 *
 * Four per-direction matrix variants are pre-factorized (interior /
 * left / right / leftright); bflag selects among them at runtime, in
 * exact parallel to the MatrixCompactDerivs path. Bandwidths kl/ku
 * are auto-detected from the assembled dense matrices, so derived
 * classes do not have to declare them explicitly (declarations may
 * still be passed via kVals but are ignored).
 *
 * Note: P/Q are sized at p_n = 2*eleorder + 1, set at construction.
 * Calls with sz[i] != p_n produce garbage at present. Fixing that is
 * deferred; see findings/04_evidence_gaps.md.
 */
class BandedCompactDerivs : public CompactDerivs {
   public:
    // one per BoundaryType: {NO, LEFT, RIGHT, LEFTRIGHT}. Public so that
    // the free-function build_variant in derivs_banded.cpp can construct it.
    struct Variant {
        int pkl = 0, pku = 0, qkl = 0, qku = 0;        // per-variant bandwidths
        std::vector<double> Pb;                         // banded-stored P
        std::vector<double> Qb;                         // banded-stored Q
        BandedMatrixSolveVars *vars = nullptr;          // owns AFB, IPIV, etc.
    };

   protected:
    std::array<Variant, 4> variants_;

    double Q_parity_ = -1.0;                            // -1 for 1st, +1 for 2nd

    // workspace for matmul + solve (reused across directions, not thread-safe)
    std::vector<double> workspace_;

    // ignored: kept only because old derived classes still pass it in.
    BandedMatrixDiagonalWidths *kVals = nullptr;

   protected:
    // build all four variants. Q_parity must be -1 for first-derivative
    // schemes and +1 for second-derivative schemes; this multiplies the
    // mirrored boundary rows of Q to enforce the correct symmetry.
    void init(MatrixDiagonalEntries *entries, double Q_parity);

    // legacy entry point: ignores kVals (bandwidths are auto-detected).
    // Parity is now required from the caller because the original
    // implementation hardcoded -1 and silently produced sign-flipped
    // boundary rows for every second-derivative scheme.
    void init(BandedMatrixDiagonalWidths *kVals,
              MatrixDiagonalEntries *entries, double Q_parity) {
        init(entries, Q_parity);
    }

    // pick the variant whose factorization matches the boundary state
    // along the named axis
    inline Variant &select_x(unsigned int bflag) {
        const bool L = bflag & (1u << OCT_DIR_LEFT);
        const bool R = bflag & (1u << OCT_DIR_RIGHT);
        BoundaryType bt = BoundaryType::NO_BOUNDARY;
        if (L && !R)      bt = BoundaryType::LEFT_BOUNDARY;
        else if (!L && R) bt = BoundaryType::RIGHT_BOUNDARY;
        else if (L && R)  bt = BoundaryType::LEFTRIGHT_BOUNDARY;
        // fall back to NO_BOUNDARY for any variant that didn't factor
        return variants_[bt].vars ? variants_[bt]
                                  : variants_[BoundaryType::NO_BOUNDARY];
    }
    inline Variant &select_y(unsigned int bflag) {
        const bool L = bflag & (1u << OCT_DIR_DOWN);
        const bool R = bflag & (1u << OCT_DIR_UP);
        BoundaryType bt = BoundaryType::NO_BOUNDARY;
        if (L && !R)      bt = BoundaryType::LEFT_BOUNDARY;
        else if (!L && R) bt = BoundaryType::RIGHT_BOUNDARY;
        else if (L && R)  bt = BoundaryType::LEFTRIGHT_BOUNDARY;
        // fall back to NO_BOUNDARY for any variant that didn't factor
        return variants_[bt].vars ? variants_[bt]
                                  : variants_[BoundaryType::NO_BOUNDARY];
    }
    inline Variant &select_z(unsigned int bflag) {
        const bool L = bflag & (1u << OCT_DIR_BACK);
        const bool R = bflag & (1u << OCT_DIR_FRONT);
        BoundaryType bt = BoundaryType::NO_BOUNDARY;
        if (L && !R)      bt = BoundaryType::LEFT_BOUNDARY;
        else if (!L && R) bt = BoundaryType::RIGHT_BOUNDARY;
        else if (L && R)  bt = BoundaryType::LEFTRIGHT_BOUNDARY;
        // fall back to NO_BOUNDARY for any variant that didn't factor
        return variants_[bt].vars ? variants_[bt]
                                  : variants_[BoundaryType::NO_BOUNDARY];
    }

   public:
    BandedCompactDerivs(unsigned int ele_order) : CompactDerivs{ele_order} {};
    BandedCompactDerivs(const BandedCompactDerivs &obj) : CompactDerivs(obj) {};
    virtual ~BandedCompactDerivs();

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override;
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override;
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override;

    void set_maximum_block_size(size_t block_size) override {};
};
}  // namespace dendroderivs
