#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "derivatives.h"
#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_matrixonly.h"
#include "derivatives/derivs_utils.h"

/**
 * @file derivs_ccd.h
 * @brief Combined Compact Difference (CCD) operators.
 *
 * A CCD scheme solves for f' and f'' out of ONE coupled system rather than two
 * independent P f' = Q f systems. Treating f'' as a second unknown field buys
 * 50% more free coefficients at the same stencil width, which is worth order
 * 6m instead of Pade's 4m at stencil half-width m (a 5-point CCD is 12th order
 * where a 5-point pentadiagonal Pade is 8th).
 *
 * ## Why this drops into the existing engine unchanged
 *
 * We solve in the SCALED unknowns
 *
 *     g = h f'        w = h^2 f''
 *
 * which makes the whole system h-free. The two operators that fall out are then
 * h-free dense n x n matrices -- exactly what MatrixCompactDerivs already
 * caches per block size and applies with 1/h^p as the GEMM scale. So CCD reuses
 * do_grad_x/y/z, the "_last" variants, the batch entries, the per-size storage,
 * the bflag dispatch and all three libxsmm kernels verbatim; only the assembly
 * step is new. Written dimensionally instead (with dx inside the blocks, as the
 * CCD literature does) the operator would depend on h and could no longer be
 * cached per size at all.
 *
 * ## Shape of the system
 *
 * Unknowns are interleaved, v = [g_0, w_0, g_1, w_1, ...], so the LHS A is
 * block-tridiagonal with 2x2 blocks (scalar-banded, kl = ku = 3 for a 3-point
 * scheme) and the RHS B is 2n x n. One solve gives M = A^-1 B (2n x n), and the
 * two derivative operators are alternate rows of M -- D1 = rows 2i, D2 = rows
 * 2i+1. Picking them out is a strided copy, not a matmul.
 *
 * The equivalent "Sengupta" form (group the unknowns as [g_0..g_N, w_0..w_N]
 * and eliminate one by a Schur complement) is the SAME system under a
 * perfect-shuffle permutation, so it is not implemented here: it needs four
 * n x n inversions instead of one, its Schur complements fill in dense from
 * banded inputs, and it additionally requires two intermediate blocks to be
 * invertible on their own. It lives in scripts/ccd_operators.py as an
 * independent oracle instead.
 *
 * Coefficients are supplied per scheme as a CCDDiagonalEntries; see impl_ccd.h.
 */

namespace dendroderivs {

/**
 * @brief One CCD equation's coefficients: the g, w and f families.
 *
 * Each equation reads
 *
 *     sum_k G_k g_{i+k} + sum_k W_k w_{i+k} = sum_k F_k f_{i+k}
 *
 * `*Interior` are the constant-coefficient interior stencils, centered on the
 * diagonal the same way MatrixDiagonalEntries::PDiagInterior is. `*Boundary`
 * holds one row per near-boundary node, each row a dense list indexed from
 * column 0 (so a term the scheme omits is an explicit 0.0, not a gap).
 * `*BoundaryLower` is the bottom/right closure in the SAME index order as the
 * top; createMatrix reverses it and applies the parity when it places it.
 *
 * All three families of one equation must share a stencil width, since they are
 * centered by the same rule.
 */
struct CCDRowSet {
    std::vector<double> GInterior;
    std::vector<double> WInterior;
    std::vector<double> FInterior;

    std::vector<std::vector<double>> GBoundary;
    std::vector<std::vector<double>> WBoundary;
    std::vector<std::vector<double>> FBoundary;

    std::vector<std::vector<double>> GBoundaryLower;
    std::vector<std::vector<double>> WBoundaryLower;
    std::vector<std::vector<double>> FBoundaryLower;
};

/**
 * @brief The two equations of a CCD scheme.
 *
 * `eq1` is normalized on g_i and yields D1 (the first derivative); `eq2` is
 * normalized on w_i and yields D2. The convenience constructor mirrors each
 * equation's lower closure from its upper one, which is what a scheme with
 * symmetric ends wants; a scheme with a genuinely asymmetric closure assigns
 * the `*BoundaryLower` fields directly afterwards.
 *
 * @warning The two boundary rows must not be linearly dependent. Deriving both
 * closures from the same term set at maximal order looks natural and is wrong:
 * the space of one-sided relations at maximal order is one-dimensional, so both
 * equations land on the same relation up to scale, A goes singular, and the
 * solver returns garbage rather than failing. Dropping the w_i term from the
 * first-derivative closure fixes it by construction. createCCDMatrixSystem-
 * ForSingleSize checks the conditioning and throws, but the cheap fix is to not
 * build a degenerate pair in the first place. See impl_ccd.cpp.
 */
struct CCDDiagonalEntries {
    CCDRowSet eq1;
    CCDRowSet eq2;

    CCDDiagonalEntries(CCDRowSet eq1_in, CCDRowSet eq2_in)
        : eq1{std::move(eq1_in)}, eq2{std::move(eq2_in)} {
        // symmetric ends by default; the parity flip happens at placement, so
        // the lower rows are literally the upper ones in the same index order.
        if (eq1.GBoundaryLower.empty()) eq1.GBoundaryLower = eq1.GBoundary;
        if (eq1.WBoundaryLower.empty()) eq1.WBoundaryLower = eq1.WBoundary;
        if (eq1.FBoundaryLower.empty()) eq1.FBoundaryLower = eq1.FBoundary;
        if (eq2.GBoundaryLower.empty()) eq2.GBoundaryLower = eq2.GBoundary;
        if (eq2.WBoundaryLower.empty()) eq2.WBoundaryLower = eq2.WBoundary;
        if (eq2.FBoundaryLower.empty()) eq2.FBoundaryLower = eq2.FBoundary;
    }
};

/**
 * @brief A CCD scheme written in our block notation. THIS is where you add one.
 *
 * A scheme is a stack of 2x2 blocks, one per stencil offset k, acting on the
 * pair v_i = (f'_i, f''_i):
 *
 *     B_k v_{i-k}  +  D v_i  +  A_k v_{i+k}   =   (rhs on f)
 *
 *     A_k = [ -b1_k   -c1_k*dx ]     B_k = [ -b1_k   +c1_k*dx ]
 *           [ -b2_k/dx  -c2_k  ]           [ +b2_k/dx  -c2_k  ]
 *
 *     D   = [ 1  0 ]
 *           [ 0  1 ]
 *
 * so row 0 is the f' equation and row 1 is the f'' equation, and D is the
 * identity because each equation is normalized on its own unknown.
 *
 * ## dx does not appear here, and that is not a simplification
 *
 * Solving in the scaled unknowns g = h f' and w = h^2 f'' cancels every dx in
 * the blocks above *exactly* -- eq1 comes out with an overall 1/h and eq2 with
 * 1/h^2, which is precisely the factor MatrixCompactDerivs already applies per
 * call. The named coefficients survive untouched. So the numbers you type here
 * are the same b1, c1, b2, c2 as on the whiteboard; you just never write the dx.
 * (This is also what lets one operator be cached and reused at every h.)
 *
 * ## What to type
 *
 * One entry per off-diagonal k = 1..m, ordered outward from the diagonal, for a
 * (2m+1)-point scheme. A 3-point scheme has one entry in each list. The RHS f
 * coefficients are `a1` (f' equation) and `a2` (f'' equation), matching the a1
 * in the C1 matrix of the N x N form.
 *
 * The center f coefficients are NOT typed: they are forced by consistency (the
 * operator must annihilate a constant), so ccd_from_blocks() derives them --
 * 0 for eq1 by antisymmetry, -2*sum(a2_k) for eq2. Two fewer numbers to get
 * wrong.
 *
 * Interior symmetry is likewise applied for you, not typed: the g and w rows
 * follow the A_k/B_k sign pattern above, so a scheme is exactly 6m numbers plus
 * its closure rows.
 *
 * @note Sign convention follows the blocks as literally written -- A_k carries
 * a LEADING MINUS on b1. So a scheme whose f'_{i±1} coefficient is +7/16 is
 * entered here as `b1 = {-7.0/16.0}`. If your notes write b1 = +7/16, they are
 * using A_k = [+b1 ...] and every b1/b2 sign here flips.
 */
struct CCDBlocks {
    // eq1 -- the f' equation. b1: g family, c1: w family, a1: f (rhs).
    std::vector<double> b1, c1, a1;
    // eq2 -- the f'' equation. b2: g family, c2: w family, a2: f (rhs).
    std::vector<double> b2, c2, a2;

    // One-sided closure rows, one per near-boundary node, each a dense list
    // indexed from column 0 (an omitted term is an explicit 0.0). These do not
    // use the b/c/a naming: a one-sided row has no symmetry to exploit, so it
    // is written out in full. Suffix 1 = f' equation, 2 = f'' equation.
    std::vector<std::vector<double>> gBoundary1, wBoundary1, fBoundary1;
    std::vector<std::vector<double>> gBoundary2, wBoundary2, fBoundary2;
};

/**
 * @brief Expand the block notation into the interleaved row form the builder
 * assembles from.
 *
 * @throws std::invalid_argument if the families of an equation disagree on m,
 * or if m is 0.
 */
CCDDiagonalEntries *ccd_from_blocks(const CCDBlocks &blk);

/**
 * @brief Build the four bflag variants of a CCD operator for one block size.
 *
 * `DerivOrder` selects which half of the coupled solution is kept: 1 -> D1
 * (rows 2i of A^-1 B), 2 -> D2 (rows 2i+1). Both come from the same
 * CCDDiagonalEntries, which is why a CCD scheme registers the same coefficient
 * function in both the first- and second-order registries.
 *
 * @throws std::runtime_error if the assembled system is numerically singular
 * (almost always a degenerate closure pair -- see CCDDiagonalEntries).
 */
template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage> createCCDMatrixSystemForSingleSize(
    const unsigned int pw, const unsigned int n,
    const CCDDiagonalEntries *ccdEntries, const bool skip_leftright = false);

/**
 * @brief Matrix-form engine for CCD schemes.
 *
 * Everything that applies the operator is inherited unchanged from
 * MatrixCompactDerivs -- this class exists only to swap in the coupled build.
 * The base's `diagEntries` (P/Q) stays null and unused.
 */
template <unsigned int DerivOrder>
class CombinedCompactDerivs : public MatrixCompactDerivs<DerivOrder> {
   protected:
    CCDDiagonalEntries *ccdEntries = nullptr;

    std::unique_ptr<DerivMatrixStorage> build_storage_for_size(
        unsigned int n, bool skip_leftright) override {
        if (!ccdEntries) {
            throw std::runtime_error(
                "CombinedCompactDerivs: ccdEntries was never set — the derived "
                "class must assign it before calling init()");
        }
        return createCCDMatrixSystemForSingleSize<DerivOrder>(
            this->p_pw, n, ccdEntries, skip_leftright);
    }

   public:
    CombinedCompactDerivs(unsigned int ele_order,
                          const std::string &in_matrix_filter = "none",
                          const std::vector<double> &in_matrix_filter_coeffs =
                              std::vector<double>())
        : MatrixCompactDerivs<DerivOrder>{ele_order, in_matrix_filter,
                                          in_matrix_filter_coeffs} {
        // in-matrix filter fusion isn't defined for a coupled operator yet:
        // it's ambiguous whether sigma*F belongs in the f' equation's rhs, the
        // f'' equation's, or both. refuse rather than silently ignore the
        // filter the caller asked for.
        if (this->in_matrix_filter_->get_filter_type() !=
            InMatFilterType::IMFT_NONE) {
            throw std::invalid_argument(
                "CCD schemes do not support in-matrix filters yet (requested: "
                "'" +
                in_matrix_filter +
                "'). Use an explicit filter, or 'none'.");
        }
    }

    CombinedCompactDerivs(const CombinedCompactDerivs &obj)
        : MatrixCompactDerivs<DerivOrder>(obj) {
        // deep copy: CCDDiagonalEntries is all value types, so the implicit
        // copy is a real one. the base copy ctor already deep-copied
        // D_storage_map_, so a clone is usable without rebuilding.
        ccdEntries = obj.ccdEntries ? new CCDDiagonalEntries(*obj.ccdEntries)
                                    : nullptr;
    }

    ~CombinedCompactDerivs() { delete ccdEntries; }
};

// one CCD scheme = one function returning its coupled coefficients
using CCDDiagCreatorFn = CCDDiagonalEntries *(*)();

/**
 * @brief Generic CCD wrapper — the per-scheme boilerplate eliminator.
 *
 * Mirrors GenericMatrixDerivs: a scheme is a coefficient function plus a
 * registry line, with no class of its own.
 */
template <unsigned int DerivOrder>
class GenericCCDDerivs : public CombinedCompactDerivs<DerivOrder> {
    CCDDiagCreatorFn ccd_fn_;
    DerivType dtype_;
    std::string name_;

   public:
    GenericCCDDerivs(CCDDiagCreatorFn fn, DerivType dt, std::string name,
                     unsigned int ele_order, const std::string &filter = "none",
                     const std::vector<double> &fcoeffs = {})
        : CombinedCompactDerivs<DerivOrder>{ele_order, filter, fcoeffs},
          ccd_fn_(fn),
          dtype_(dt),
          name_(std::move(name)) {
        this->ccdEntries = ccd_fn_();
        // safe to call the virtual from here: the object is already a
        // GenericCCDDerivs by the time a ctor body runs, so init() dispatches
        // to CombinedCompactDerivs::build_storage_for_size.
        this->init();
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<GenericCCDDerivs>(*this);
    }
    DerivType getDerivType() const override { return dtype_; }
    enum DerivOrder getDerivOrder() const override {
        return (DerivOrder == 1) ? D_FIRST_ORDER : D_SECOND_ORDER;
    }
    std::string toString() const override { return name_; }
    void set_maximum_block_size(size_t block_size) override {
        MatrixCompactDerivs<DerivOrder>::set_maximum_block_size(block_size);
    }
};

}  // namespace dendroderivs
