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
 * @file derivs_ccfd.h
 * @brief Combined Compact Finite Difference (CCFD) operators.
 *
 * A CCFD scheme solves for f' and f'' out of ONE coupled system rather than two
 * independent P f' = Q f systems. Treating f'' as a second unknown field buys
 * 50% more free coefficients at the same stencil width, which is worth order
 * 6m instead of Pade's 4m at stencil half-width m (a 5-point CCFD is 12th order
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
 * caches per block size and applies with 1/h^p as the GEMM scale. So CCFD reuses
 * do_grad_x/y/z, the "_last" variants, the batch entries, the per-size storage,
 * the bflag dispatch and all three libxsmm kernels verbatim; only the assembly
 * step is new. Written dimensionally instead (with dx inside the blocks, as the
 * CCFD literature does) the operator would depend on h and could no longer be
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
 * The equivalent "Sengupta" solve (group the unknowns as [g_0..g_N, w_0..w_N]
 * and eliminate one by a Schur complement) is the SAME system under a
 * perfect-shuffle permutation, so it is not implemented here: it needs four
 * n x n inversions instead of one, its Schur complements fill in dense from
 * banded inputs, and it additionally requires two intermediate blocks to be
 * invertible on their own. It lives in scripts/ccfd_operators.py as an
 * independent oracle instead.
 *
 * NOTE the distinction: a scheme is *defined* below in Sengupta's six-matrix
 * notation (A1,B1,C1 / A2,B2,C2), because that is the form the collaborators'
 * Mathematica notebook emits and the form written up in
 * findings/CCFD_Matrix_Implementation.pdf. That notation is only the coefficient
 * *layout*; the coupled system it describes is still assembled and solved
 * interleaved, per the paragraph above. Definition format and solve method are
 * separate choices, and only the definition format is Sengupta's.
 */

namespace dendroderivs {

/**
 * @brief A CCFD scheme in Sengupta's six-matrix notation. THIS is where you add
 * one.
 *
 * Sengupta writes the combined compact system as two coupled equations
 *
 *     A1 f' + B1 f'' = C1 f      (normalized on f'_i  -> yields D1)
 *     A2 f' + B2 f'' = C2 f      (normalized on f''_i -> yields D2)
 *
 * so there are six banded matrices. The A matrices hold the f' coefficients
 * (the alpha family), the B matrices the f'' coefficients (beta), and the C
 * matrices the f coefficients (a). See findings/CCFD_Matrix_Implementation.pdf
 * for the full N x N matrix picture; the fields here are the diagonal + closure
 * data those matrices are built from.
 *
 * Each `*Interior` vector is the fully-expanded, CENTERED interior stencil
 * (length 2m+1 for a (2m+1)-point scheme), exactly as the generated header
 * writes it, e.g.
 *
 *     A1Interior = { alpha13, alpha12, alpha11, 1.0, alpha11, alpha12, alpha13 }
 *     A2Interior = { -alpha23, -alpha22, -alpha21, 0.0, alpha21, alpha22, alpha23 }
 *
 * The generated header bakes each matrix's symmetry (A1,B2,C2 symmetric;
 * A2,B1,C1 antisymmetric) and the consistency-forced centers (0 for the
 * antisymmetric ones, s = -2*sum(a2_k) for C2) straight into these vectors, so
 * the builder places them literally -- it does not re-derive any of that.
 *
 * Each `*Boundary` holds one dense row per near-boundary node, indexed from
 * column 0 (a term the scheme omits is an explicit 0.0, not a gap).
 * `*BoundaryLower` is the bottom/right closure in the SAME index order as the
 * top; the builder reverses it and applies the matrix's parity when it places
 * it. The convenience constructor mirrors each upper closure onto its lower one,
 * which is what a scheme with symmetric ends wants; a scheme with a genuinely
 * asymmetric closure assigns the `*BoundaryLower` fields directly afterwards.
 *
 * The constructor argument order matches the generated-header template exactly:
 * the six interior vectors (A1,A2,B1,B2,C1,C2) followed by the six boundary
 * blocks in the same order. A Mathematica-emitted header therefore constructs
 * this struct verbatim.
 *
 * @warning The two boundary rows must not be linearly dependent. Deriving both
 * closures from the same term set at maximal order looks natural and is wrong:
 * the space of one-sided relations at maximal order is one-dimensional, so both
 * equations land on the same relation up to scale, A goes singular, and the
 * solver returns garbage rather than failing. Dropping the f'' term from the
 * first-derivative closure (an explicit 0.0 in B1Boundary's first column) fixes
 * it by construction. createCCFDMatrixSystemForSingleSize checks the
 * conditioning and throws, but the cheap fix is to not build a degenerate pair
 * in the first place. See impl_ccfd.cpp.
 */
struct CCFDDiagonalEntries {
    // eq1 (D1): A1 f' + B1 f'' = C1 f. eq2 (D2): A2 f' + B2 f'' = C2 f.
    // Declared in the generated-header constructor's argument order.
    std::vector<double> A1Interior, A2Interior;
    std::vector<double> B1Interior, B2Interior;
    std::vector<double> C1Interior, C2Interior;

    std::vector<std::vector<double>> A1Boundary, A2Boundary;
    std::vector<std::vector<double>> B1Boundary, B2Boundary;
    std::vector<std::vector<double>> C1Boundary, C2Boundary;

    std::vector<std::vector<double>> A1BoundaryLower, A2BoundaryLower;
    std::vector<std::vector<double>> B1BoundaryLower, B2BoundaryLower;
    std::vector<std::vector<double>> C1BoundaryLower, C2BoundaryLower;

    CCFDDiagonalEntries(
        std::vector<double> A1Interior_, std::vector<double> A2Interior_,
        std::vector<double> B1Interior_, std::vector<double> B2Interior_,
        std::vector<double> C1Interior_, std::vector<double> C2Interior_,
        std::vector<std::vector<double>> A1Boundary_,
        std::vector<std::vector<double>> A2Boundary_,
        std::vector<std::vector<double>> B1Boundary_,
        std::vector<std::vector<double>> B2Boundary_,
        std::vector<std::vector<double>> C1Boundary_,
        std::vector<std::vector<double>> C2Boundary_)
        : A1Interior{std::move(A1Interior_)},
          A2Interior{std::move(A2Interior_)},
          B1Interior{std::move(B1Interior_)},
          B2Interior{std::move(B2Interior_)},
          C1Interior{std::move(C1Interior_)},
          C2Interior{std::move(C2Interior_)},
          A1Boundary{std::move(A1Boundary_)},
          A2Boundary{std::move(A2Boundary_)},
          B1Boundary{std::move(B1Boundary_)},
          B2Boundary{std::move(B2Boundary_)},
          C1Boundary{std::move(C1Boundary_)},
          C2Boundary{std::move(C2Boundary_)} {
        // symmetric ends by default; the parity flip happens at placement, so
        // the lower rows are literally the upper ones in the same index order.
        A1BoundaryLower = A1Boundary;
        A2BoundaryLower = A2Boundary;
        B1BoundaryLower = B1Boundary;
        B2BoundaryLower = B2Boundary;
        C1BoundaryLower = C1Boundary;
        C2BoundaryLower = C2Boundary;
    }
};

/**
 * @brief Build the four bflag variants of a CCFD operator for one block size.
 *
 * `DerivOrder` selects which half of the coupled solution is kept: 1 -> D1
 * (rows 2i of A^-1 B), 2 -> D2 (rows 2i+1). Both come from the same
 * CCFDDiagonalEntries, which is why a CCFD scheme registers the same coefficient
 * function in both the first- and second-order registries.
 *
 * @throws std::runtime_error if the assembled system is numerically singular
 * (almost always a degenerate closure pair -- see CCFDDiagonalEntries).
 */
template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage> createCCFDMatrixSystemForSingleSize(
    const unsigned int pw, const unsigned int n,
    const CCFDDiagonalEntries *ccfdEntries, const bool skip_leftright = false);

/**
 * @brief Matrix-form engine for CCFD schemes.
 *
 * Everything that applies the operator is inherited unchanged from
 * MatrixCompactDerivs -- this class exists only to swap in the coupled build.
 * The base's `diagEntries` (P/Q) stays null and unused.
 */
template <unsigned int DerivOrder>
class CombinedCompactDerivs : public MatrixCompactDerivs<DerivOrder> {
   protected:
    CCFDDiagonalEntries *ccfdEntries = nullptr;

    std::unique_ptr<DerivMatrixStorage> build_storage_for_size(
        unsigned int n, bool skip_leftright) override {
        if (!ccfdEntries) {
            throw std::runtime_error(
                "CombinedCompactDerivs: ccfdEntries was never set — the derived "
                "class must assign it before calling init()");
        }
        return createCCFDMatrixSystemForSingleSize<DerivOrder>(
            this->p_pw, n, ccfdEntries, skip_leftright);
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
                "CCFD schemes do not support in-matrix filters yet (requested: "
                "'" +
                in_matrix_filter +
                "'). Use an explicit filter, or 'none'.");
        }
    }

    CombinedCompactDerivs(const CombinedCompactDerivs &obj)
        : MatrixCompactDerivs<DerivOrder>(obj) {
        // deep copy: CCFDDiagonalEntries is all value types, so the implicit
        // copy is a real one. the base copy ctor already deep-copied
        // D_storage_map_, so a clone is usable without rebuilding.
        ccfdEntries = obj.ccfdEntries
                          ? new CCFDDiagonalEntries(*obj.ccfdEntries)
                          : nullptr;
    }

    ~CombinedCompactDerivs() { delete ccfdEntries; }
};

// one CCFD scheme = one function returning its coupled coefficients
using CCFDDiagCreatorFn = CCFDDiagonalEntries *(*)();

/**
 * @brief Generic CCFD wrapper — the per-scheme boilerplate eliminator.
 *
 * Mirrors GenericMatrixDerivs: a scheme is a coefficient function plus a
 * registry line, with no class of its own.
 */
template <unsigned int DerivOrder>
class GenericCCFDDerivs : public CombinedCompactDerivs<DerivOrder> {
    CCFDDiagCreatorFn ccfd_fn_;
    DerivType dtype_;
    std::string name_;

   public:
    GenericCCFDDerivs(CCFDDiagCreatorFn fn, DerivType dt, std::string name,
                      unsigned int ele_order,
                      const std::string &filter          = "none",
                      const std::vector<double> &fcoeffs = {})
        : CombinedCompactDerivs<DerivOrder>{ele_order, filter, fcoeffs},
          ccfd_fn_(fn),
          dtype_(dt),
          name_(std::move(name)) {
        this->ccfdEntries = ccfd_fn_();
        // safe to call the virtual from here: the object is already a
        // GenericCCFDDerivs by the time a ctor body runs, so init() dispatches
        // to CombinedCompactDerivs::build_storage_for_size.
        this->init();
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<GenericCCFDDerivs>(*this);
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

// A parameterized CCFD scheme: same as CCFDDiagCreatorFn but the coefficient
// function takes a runtime coefficient vector, so one generated function covers
// a whole tunable family. Mirrors DiagCreatorWithCoeffsFn for the P/Q schemes.
using CCFDDiagCreatorWithCoeffsFn =
    CCFDDiagonalEntries *(*)(const std::vector<double> &);

/**
 * @brief Generic wrapper for CCFD schemes that accept user coefficients.
 *
 * The coupled analogue of GenericMatrixDerivsWithCoeffs: a tunable scheme is a
 * coefficient function `create<Name>Diagonals(const std::vector<double>&)` plus
 * a registry line via make_ccfd_coeffs, with no class of its own. The generated
 * header's coefficient section references `D_coeffs[k]` for the free
 * parameters; a scheme with none uses the no-arg CCFDDiagCreatorFn instead.
 *
 * The `coeffs_in` vector is padded/truncated to exactly `n_coeffs` (missing
 * slots default to 0.0), identical to the BYU families, so a caller that passes
 * too few or too many values gets a well-defined operator rather than a crash.
 */
template <unsigned int DerivOrder>
class GenericCCFDDerivsWithCoeffs : public CombinedCompactDerivs<DerivOrder> {
    CCFDDiagCreatorWithCoeffsFn ccfd_fn_;
    DerivType dtype_;
    std::string name_;
    std::vector<double> coeffs_;
    unsigned int n_coeffs_;

   public:
    GenericCCFDDerivsWithCoeffs(CCFDDiagCreatorWithCoeffsFn fn, DerivType dt,
                                std::string name, unsigned int n_coeffs,
                                unsigned int ele_order,
                                const std::string &filter          = "none",
                                const std::vector<double> &fcoeffs = {},
                                const std::vector<double> &coeffs_in = {})
        : CombinedCompactDerivs<DerivOrder>{ele_order, filter, fcoeffs},
          ccfd_fn_(fn),
          dtype_(dt),
          name_(std::move(name)),
          n_coeffs_(n_coeffs) {
        // pad/truncate to the expected count, same as GenericMatrixDerivsWithCoeffs
        coeffs_.resize(n_coeffs_, 0.0);
        for (unsigned int i = 0; i < n_coeffs_ && i < coeffs_in.size(); i++)
            coeffs_[i] = coeffs_in[i];
        this->ccfdEntries = ccfd_fn_(coeffs_);
        // safe to call the virtual from here: the object is already a
        // GenericCCFDDerivsWithCoeffs by the time a ctor body runs.
        this->init();
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<GenericCCFDDerivsWithCoeffs>(*this);
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
