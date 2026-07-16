#include "derivatives/derivs_ccd.h"

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"
#include "lapac.h"

// LAPACK routines we need that lapac.h doesn't declare. We deliberately do the
// factor / estimate-condition / solve sequence by hand instead of calling a
// plain dgesv, for two reasons:
//   1. lapack::lapack_DGESV_T assumes nrhs == n (it memcpy's n*n into X), which
//      is false here -- A is 2n x 2n but B is only 2n x n.
//   2. a degenerate CCD closure pair makes A numerically singular, and dgesv
//      only reports EXACT singularity. It would return plausible-looking
//      garbage instead, so we estimate rcond and refuse.
extern "C" void dgetrs_(char* trans, int* n, int* nrhs, double* a, int* lda,
                        int* ipiv, double* b, int* ldb, int* info);
extern "C" void dgecon_(char* norm, int* n, double* a, int* lda, double* anorm,
                        double* rcond, double* work, int* iwork, int* info);
extern "C" double dlange_(char* norm, int* m, int* n, double* a, int* lda,
                          double* work);

namespace dendroderivs {

namespace {

/**
 * Per-family parity for mirroring a top closure onto the bottom -- the coupled
 * analogue of the scalar Q_parity in createMatrixSystemForSingleSize.
 *
 * Under reflection g = h f' is odd while w = h^2 f'' and f are even. Each
 * equation is then renormalized by its own pinned coefficient's parity so the
 * pinned term stays +1 (eq1 is pinned on g, which is odd; eq2 on w, which is
 * even). That works out to:
 *
 *     eq1:  g +1,  w -1,  f -1
 *     eq2:  g -1,  w +1,  f +1
 *
 * Get these backwards and the operator still converges in the interior -- only
 * the closure rows are wrong -- so it fails quietly. Verified against the
 * independent Python builder (scripts/ccd_operators.py), where mirroring the
 * symmetric interior rows is required to be the identity.
 */
struct FamilyParity {
    double g;
    double w;
    double f;
};
constexpr FamilyParity kEq1Parity{1.0, -1.0, -1.0};
constexpr FamilyParity kEq2Parity{-1.0, 1.0, 1.0};

// Refuse to hand back operators from a system this ill-conditioned. A healthy
// CCD system sits around rcond ~ 1e-6 and, usefully, is flat in block size; a
// degenerate closure pair collapses to ~1e-18, which is what this catches.
constexpr double kRcondFloor = 1.0e3 * std::numeric_limits<double>::epsilon();

const char* boundary_name(BoundaryType b) {
    switch (b) {
        case BoundaryType::NO_BOUNDARY: return "NO_BOUNDARY";
        case BoundaryType::LEFT_BOUNDARY: return "LEFT_BOUNDARY";
        case BoundaryType::RIGHT_BOUNDARY: return "RIGHT_BOUNDARY";
        case BoundaryType::LEFTRIGHT_BOUNDARY: return "LEFTRIGHT_BOUNDARY";
    }
    return "UNKNOWN";
}

// Build one equation's three family matrices on the active (n_fill) grid.
// createMatrix already does the reversal + parity for the lower closure and
// the interior centering, so we reuse it per family rather than re-deriving
// that placement logic. boundary_top/bottom are passed as 0 here: the identity
// embed for a CCD block has to happen on the 2x2-block level, not per family,
// so we do it ourselves below.
void build_family_matrices(const CCDRowSet& rs, unsigned int n_fill,
                           const FamilyParity& par, std::vector<double>& G,
                           std::vector<double>& W, std::vector<double>& F) {
    G = createMatrix(rs.GBoundary, rs.GBoundaryLower, rs.GInterior, n_fill,
                     par.g, 0, 0);
    W = createMatrix(rs.WBoundary, rs.WBoundaryLower, rs.WInterior, n_fill,
                     par.w, 0, 0);
    F = createMatrix(rs.FBoundary, rs.FBoundaryLower, rs.FInterior, n_fill,
                     par.f, 0, 0);
}

}  // namespace

namespace {

// build one interior row: coefficients for offsets -m..+m, centered.
// `sym` is the parity of the outward entries about the center: the +k entry is
// `sign * v[k-1]` and the -k entry is `sym * sign * v[k-1]`.
std::vector<double> expand_row(const std::vector<double>& v, double sign,
                               double sym, double center) {
    const size_t m = v.size();
    std::vector<double> row(2 * m + 1, 0.0);
    row[m] = center;
    for (size_t k = 1; k <= m; ++k) {
        row[m + k] = sign * v[k - 1];
        row[m - k] = sym * sign * v[k - 1];
    }
    return row;
}

void require_same_width(const std::vector<double>& a,
                        const std::vector<double>& b,
                        const std::vector<double>& c, const char* eq) {
    if (a.size() != b.size() || a.size() != c.size()) {
        std::ostringstream os;
        os << "ccd_from_blocks: the three families of " << eq
           << " must all have one entry per off-diagonal, but got sizes "
           << a.size() << ", " << b.size() << " and " << c.size()
           << ". A (2m+1)-point scheme needs exactly m in each.";
        throw std::invalid_argument(os.str());
    }
    if (a.empty()) {
        throw std::invalid_argument(
            "ccd_from_blocks: a CCD scheme needs at least one off-diagonal "
            "(m >= 1); the lists for the interior stencil are empty.");
    }
}

}  // namespace

CCDDiagonalEntries* ccd_from_blocks(const CCDBlocks& blk) {
    require_same_width(blk.b1, blk.c1, blk.a1, "eq1 (b1, c1, a1)");
    require_same_width(blk.b2, blk.c2, blk.a2, "eq2 (b2, c2, a2)");

    // eq1, the f' equation. Reading the k-th blocks
    //     A_k = [-b1_k  -c1_k*dx]   B_k = [-b1_k  +c1_k*dx]
    // row 0 gives, in the scaled unknowns (dx cancels):
    //     g:  -b1_k on BOTH sides            -> sign -1, even
    //     w:  -c1_k at +k, +c1_k at -k       -> sign -1, odd
    //     f:  +a1_k at +k, -a1_k at -k       -> sign +1, odd
    // the pinned g_i is 1 and there is no w_i term; eq1's f center is 0 by
    // antisymmetry.
    CCDRowSet eq1;
    eq1.GInterior = expand_row(blk.b1, -1.0, +1.0, 1.0);
    eq1.WInterior = expand_row(blk.c1, -1.0, -1.0, 0.0);
    eq1.FInterior = expand_row(blk.a1, +1.0, -1.0, 0.0);

    // eq2, the f'' equation. Row 1 of the same blocks:
    //     A_k = [-b2_k/dx  -c2_k]   B_k = [+b2_k/dx  -c2_k]
    //     g:  -b2_k at +k, +b2_k at -k       -> sign -1, odd
    //     w:  -c2_k on BOTH sides            -> sign -1, even
    //     f:  +a2_k on BOTH sides            -> sign +1, even
    // the pinned w_i is 1 and there is no g_i term. eq2's f center is NOT free:
    // the row must annihilate a constant, so it is -2*sum(a2_k).
    double a2_sum = 0.0;
    for (double v : blk.a2) a2_sum += v;

    CCDRowSet eq2;
    eq2.GInterior = expand_row(blk.b2, -1.0, -1.0, 0.0);
    eq2.WInterior = expand_row(blk.c2, -1.0, +1.0, 1.0);
    eq2.FInterior = expand_row(blk.a2, +1.0, +1.0, -2.0 * a2_sum);

    eq1.GBoundary  = blk.gBoundary1;
    eq1.WBoundary  = blk.wBoundary1;
    eq1.FBoundary  = blk.fBoundary1;
    eq2.GBoundary  = blk.gBoundary2;
    eq2.WBoundary  = blk.wBoundary2;
    eq2.FBoundary  = blk.fBoundary2;

    return new CCDDiagonalEntries(eq1, eq2);
}

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage> createCCDMatrixSystemForSingleSize(
    const unsigned int pw, const unsigned int n,
    const CCDDiagonalEntries* ccdEntries, const bool skip_leftright) {
    static_assert(DerivOrder == 1 || DerivOrder == 2,
                  "CCD operators are only defined for the 1st and 2nd "
                  "derivative");

    // which half of the coupled solution we keep: rows 2i are g (-> D1),
    // rows 2i+1 are w (-> D2). This is the S1/S2 row selection, as a strided
    // copy rather than a matmul.
    constexpr unsigned int row_off = (DerivOrder == 1) ? 0u : 1u;

    const size_t nsq               = static_cast<size_t>(n) * n;

    auto derivMatrixPtr            = std::make_unique<DerivMatrixStorage>();
    derivMatrixPtr->D_original     = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_left         = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_right        = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->D_leftright    = std::vector<double>(nsq, 0.0);
    derivMatrixPtr->dim_size       = n;

    for (BoundaryType b :
         {BoundaryType::NO_BOUNDARY, BoundaryType::LEFT_BOUNDARY,
          BoundaryType::RIGHT_BOUNDARY, BoundaryType::LEFTRIGHT_BOUNDARY}) {
        // same rationale as the scalar builder: LEFTRIGHT only matters for
        // fused/unrefined blocks, so the smallest size skips it.
        if (b == BoundaryType::LEFTRIGHT_BOUNDARY && skip_leftright) continue;

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

        if (static_cast<int>(n) - static_cast<int>(boundary_top) -
                static_cast<int>(boundary_bottom) <=
            0) {
            throw std::invalid_argument(
                "CCD: boundary padding leaves no active points at n = " +
                std::to_string(n));
        }
        const unsigned int n_fill = n - boundary_top - boundary_bottom;

        // --- the six family matrices, on the active grid -------------------
        std::vector<double> G1, W1, F1, G2, W2, F2;
        build_family_matrices(ccdEntries->eq1, n_fill, kEq1Parity, G1, W1, F1);
        build_family_matrices(ccdEntries->eq2, n_fill, kEq2Parity, G2, W2, F2);

        // --- assemble the coupled system, interleaved ----------------------
        // v = [g_0, w_0, g_1, w_1, ...]; row 2i is eq1 at node i, row 2i+1 is
        // eq2. Column-major throughout (IDXN), matching the scalar path.
        // N2 is non-const because the LAPACK Fortran interface takes int*.
        int N2            = static_cast<int>(2 * n);
        const size_t ld_A = static_cast<size_t>(N2);

        std::vector<double> A(static_cast<size_t>(N2) * N2, 0.0);
        std::vector<double> B(static_cast<size_t>(N2) * n, 0.0);

        for (unsigned int i = 0; i < n_fill; i++) {
            const unsigned int ii = i + boundary_top;  // node in full space
            for (unsigned int j = 0; j < n_fill; j++) {
                const unsigned int jj = j + boundary_top;
                const size_t s        = IDXN(i, j, n_fill);

                A[IDXN(2 * ii, 2 * jj, ld_A)]         = G1[s];
                A[IDXN(2 * ii, 2 * jj + 1, ld_A)]     = W1[s];
                A[IDXN(2 * ii + 1, 2 * jj, ld_A)]     = G2[s];
                A[IDXN(2 * ii + 1, 2 * jj + 1, ld_A)] = W2[s];

                B[IDXN(2 * ii, jj, ld_A)]             = F1[s];
                B[IDXN(2 * ii + 1, jj, ld_A)]         = F2[s];
            }
        }

        // --- identity embed for the padding, at the 2x2-block level --------
        // When a face is a real boundary its ghost cells hold no valid data, so
        // the operator acts only on the active points and the padding passes
        // through untouched. The scalar path gets this from createMatrix
        // writing identity rows into both P and Q (so D's padding row is e_i);
        // for CCD the padding node needs an identity 2x2 block in A *and* a 1
        // in both of its B rows, which reproduces the same passthrough
        // (D1[p,p] = D2[p,p] = 1). Anything else here only shows up in the
        // bflag != 0 variants, so test all four.
        auto pad_node = [&](unsigned int p) {
            A[IDXN(2 * p, 2 * p, ld_A)]         = 1.0;
            A[IDXN(2 * p + 1, 2 * p + 1, ld_A)] = 1.0;
            B[IDXN(2 * p, p, ld_A)]             = 1.0;
            B[IDXN(2 * p + 1, p, ld_A)]         = 1.0;
        };
        for (unsigned int p = 0; p < boundary_top; p++) pad_node(p);
        for (unsigned int p = 0; p < boundary_bottom; p++) pad_node(n - p - 1);

        // --- solve A M = B, guarding against a singular assembly ------------
        char norm_one = '1';
        char trans_n  = 'N';
        int nrhs      = static_cast<int>(n);
        int lda       = N2;
        int ldb       = N2;
        int info      = 0;

        std::vector<double> work(4 * static_cast<size_t>(N2), 0.0);
        std::vector<int> iwork(static_cast<size_t>(N2), 0);
        std::vector<int> ipiv(static_cast<size_t>(N2), 0);

        // 1-norm must come off the ORIGINAL A, before dgetrf overwrites it
        const double anorm_val =
            dlange_(&norm_one, &N2, &N2, A.data(), &lda, work.data());
        double anorm = anorm_val;

        dgetrf_(&N2, &N2, A.data(), &lda, ipiv.data(), &info);
        if (info != 0) {
            std::ostringstream oss;
            oss << "CCD: LU factorization failed at n = " << n << ", variant "
                << boundary_name(b) << " (dgetrf info = " << info << "). "
                << "info > 0 means an exactly singular U -- almost always a "
                   "linearly dependent boundary closure pair.";
            throw std::runtime_error(oss.str());
        }

        double rcond = 0.0;
        dgecon_(&norm_one, &N2, A.data(), &lda, &anorm, &rcond, work.data(),
                iwork.data(), &info);
        if (info != 0 || !(rcond > kRcondFloor)) {
            std::ostringstream oss;
            oss << "CCD: system is numerically singular at n = " << n
                << ", variant " << boundary_name(b) << " (rcond = " << rcond
                << ", floor = " << kRcondFloor << "). The solve would return "
                << "plausible-looking garbage rather than fail, so it is "
                << "refused here. Most likely the two boundary closure rows "
                << "are linearly dependent -- see the warning on "
                << "CCDDiagonalEntries. A too-small block for the closure "
                << "stencil does this too.";
            throw std::runtime_error(oss.str());
        }

        dgetrs_(&trans_n, &N2, &nrhs, A.data(), &lda, ipiv.data(), B.data(),
                &ldb, &info);
        if (info != 0) {
            throw std::runtime_error(
                "CCD: triangular solve failed at n = " + std::to_string(n) +
                " (dgetrs info = " + std::to_string(info) + ")");
        }

        // --- slice out our half: D = S1 M (or S2 M) ------------------------
        std::vector<double>* const D_ptr =
            get_deriv_mat_by_boundary(derivMatrixPtr.get(), b);
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < n; j++) {
                (*D_ptr)[IDXN(i, j, n)] = B[IDXN(2 * i + row_off, j, ld_A)];
            }
        }
    }

    return derivMatrixPtr;
}

template std::unique_ptr<DerivMatrixStorage> createCCDMatrixSystemForSingleSize<
    1>(const unsigned int pw, const unsigned int n,
       const CCDDiagonalEntries* ccdEntries, const bool skip_leftright);

template std::unique_ptr<DerivMatrixStorage> createCCDMatrixSystemForSingleSize<
    2>(const unsigned int pw, const unsigned int n,
       const CCDDiagonalEntries* ccdEntries, const bool skip_leftright);

}  // namespace dendroderivs
