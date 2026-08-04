#include "derivatives/derivs_ccfd.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
//   2. a degenerate CCFD closure pair makes A numerically singular, and dgesv
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
 * Per-matrix parity for mirroring a top closure onto the bottom -- the coupled
 * analogue of the scalar Q_parity in createMatrixSystemForSingleSize.
 *
 * Under reflection g = h f' is odd while w = h^2 f'' and f are even. Each
 * equation is then renormalized by its own pinned coefficient's parity so the
 * pinned term stays +1 (eq1 is pinned on g, which is odd; eq2 on w, which is
 * even). That works out, per Sengupta matrix, to:
 *
 *     eq1:  A1 (g) +1,  B1 (w) -1,  C1 (f) -1     (A1 symmetric, B1/C1 antisym)
 *     eq2:  A2 (g) -1,  B2 (w) +1,  C2 (f) +1     (A2 antisym, B2/C2 symmetric)
 *
 * which are exactly the six matrices' documented symmetries. Get these
 * backwards and the operator still converges in the interior -- only the
 * closure rows are wrong -- so it fails quietly. Verified against the
 * independent Python builder (scripts/ccfd_operators.py), where mirroring the
 * symmetric interior rows is required to be the identity.
 */
constexpr double kA1Parity = 1.0;   // eq1, g family
constexpr double kB1Parity = -1.0;  // eq1, w family
constexpr double kC1Parity = -1.0;  // eq1, f family
constexpr double kA2Parity = -1.0;  // eq2, g family
constexpr double kB2Parity = 1.0;   // eq2, w family
constexpr double kC2Parity = 1.0;   // eq2, f family

// Refuse to hand back operators from a system this ill-conditioned. A healthy
// CCFD system sits around rcond ~ 1e-6 and, usefully, is flat in block size; a
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

// Build one Sengupta matrix (a single g/w/f family of one equation) on the
// active (n_fill) grid. createMatrix already does the reversal + parity for the
// lower closure and the interior centering, so we reuse it per family rather
// than re-deriving that placement logic. boundary_top/bottom are passed as 0
// here: the identity embed for a CCFD block has to happen on the 2x2-block
// level, not per family, so we do it ourselves below.
std::vector<double> build_family(const std::vector<double>& interior,
                                 const std::vector<std::vector<double>>& bnd,
                                 const std::vector<std::vector<double>>& bndLow,
                                 unsigned int n_fill, double parity) {
    return createMatrix(bnd, bndLow, interior, n_fill, parity, 0, 0);
}

// ---------------------------------------------------------------------------
// Canonicalization: undo the generator's septadiagonal padding.
//
// The collaborators' Mathematica header generator is written once, generically,
// for a scheme that is septadiagonal (7-point) on both sides, and narrower
// schemes are expressed by setting coefficients to zero. That works for the
// INTERIOR vectors -- a zero coefficient contributes nothing -- but it does not
// work for the closure rows, because the number of closure rows is structural,
// not a coefficient. createMatrix treats rows [0, diag_boundary.size()) as
// closure rows and starts the interior stencil only at row diag_boundary.size().
//
// So a pentadiagonal scheme emitted through the septadiagonal template carries
// three closure rows per end, and its unused third row comes out as the
// template's filler: a 1 on the pinned unknown's own node and zeros everywhere
// else, in all three of that equation's matrices. That row does not say "use the
// interior stencil here" -- it says
//
//     g_2 = 0        (eq1)          w_2 = 0        (eq2)
//
// i.e. it pins the derivative at node 2 (and, mirrored, at node n-3) to exactly
// zero. Because D = A^-1 B is dense, that wrong constraint does not stay local:
// rows 0..4 all reference g_2/w_2, so the contamination spreads across the whole
// block and the operator loses convergence entirely.
//
// It is invisible to the rcond guard below, by construction -- a row of the
// identity is perfectly well conditioned -- so it has to be caught structurally.
// Detect the filler pattern exactly and drop those rows, which lets createMatrix
// treat those nodes as interior, as the scheme intends.
// ---------------------------------------------------------------------------

// Largest offset from center carrying a nonzero coefficient, over ALL six
// interior vectors jointly. It has to be joint: createMatrix derives ku
// per-family from that family's own vector length, and two families of the same
// equation placing their interior rows at different offsets would assemble a
// row that mixes stencil widths.
unsigned int effective_half_width(const CCFDDiagonalEntries* e) {
    const std::vector<double>* const vecs[6] = {
        &e->A1Interior, &e->B1Interior, &e->C1Interior,
        &e->A2Interior, &e->B2Interior, &e->C2Interior};
    unsigned int m = 0;
    for (const auto* v : vecs) {
        if (v->empty()) continue;
        const int c = (static_cast<int>(v->size()) - 1) / 2;
        for (int j = 0; j < static_cast<int>(v->size()); j++) {
            if ((*v)[j] != 0.0) m = std::max(m, static_cast<unsigned int>(
                                                    std::abs(j - c)));
        }
    }
    return m;
}

// Is `row` of `eq` the template's filler -- the pinned unknown set to 1 on its
// own node with every other coefficient in that equation identically zero?
// eq1 pins g (A-family), eq2 pins w (B-family). Nothing a real scheme wants
// looks like this: it is an equation with no data on its right-hand side.
bool is_vacuous_pin(const CCFDDiagonalEntries* e, int eq, size_t row) {
    const auto& A = (eq == 1) ? e->A1Boundary : e->A2Boundary;
    const auto& B = (eq == 1) ? e->B1Boundary : e->B2Boundary;
    const auto& C = (eq == 1) ? e->C1Boundary : e->C2Boundary;
    if (row >= A.size() || row >= B.size() || row >= C.size()) return false;

    const auto& pinned = (eq == 1) ? A[row] : B[row];
    const auto& other  = (eq == 1) ? B[row] : A[row];

    for (double v : C[row])
        if (v != 0.0) return false;
    for (double v : other)
        if (v != 0.0) return false;
    for (size_t j = 0; j < pinned.size(); j++) {
        if (j == row) {
            if (pinned[j] != 1.0) return false;
        } else if (pinned[j] != 0.0) {
            return false;
        }
    }
    return true;
}

void trim_interior(std::vector<double>& v, unsigned int m, const char* name) {
    const size_t want = 2u * m + 1u;
    if (v.size() <= want) return;
    const size_t c = (v.size() - 1) / 2;
    for (size_t j = 0; j < v.size(); j++) {
        const size_t off = (j > c) ? (j - c) : (c - j);
        if (off > m && v[j] != 0.0) {
            throw std::runtime_error(
                std::string("CCFD: ") + name +
                " has a nonzero coefficient outside the scheme's effective "
                "half-width -- canonicalization would drop real data. This is "
                "a bug in the coefficient header, not in the trim.");
        }
    }
    v = std::vector<double>(v.begin() + (c - m), v.begin() + (c + m + 1));
}

void trim_boundary(std::vector<std::vector<double>>& b, size_t keep) {
    if (b.size() > keep) b.resize(keep);
}

// Returns a copy with the padding removed. A scheme emitted at its true width
// (CCFD6, say) is returned unchanged, so this is a no-op for anything that was
// already correct.
CCFDDiagonalEntries canonicalize(const CCFDDiagonalEntries* in) {
    CCFDDiagonalEntries e = *in;
    const unsigned int m  = effective_half_width(&e);

    // eq1 and eq2 are separate rows of the coupled matrix (2i and 2i+1), so
    // they may legitimately need different closure-row counts; trim each
    // independently. Never trim below m, or the first interior row would reach
    // outside the matrix (createMatrix now also guards this).
    for (int eq = 1; eq <= 2; eq++) {
        auto& A     = (eq == 1) ? e.A1Boundary : e.A2Boundary;
        auto& B     = (eq == 1) ? e.B1Boundary : e.B2Boundary;
        auto& C     = (eq == 1) ? e.C1Boundary : e.C2Boundary;
        auto& AL    = (eq == 1) ? e.A1BoundaryLower : e.A2BoundaryLower;
        auto& BL    = (eq == 1) ? e.B1BoundaryLower : e.B2BoundaryLower;
        auto& CL    = (eq == 1) ? e.C1BoundaryLower : e.C2BoundaryLower;

        size_t keep = A.size();
        while (keep > 0 && is_vacuous_pin(in, eq, keep - 1)) keep--;
        keep = std::max<size_t>(keep, m);
        // only contiguous trailing filler is removed; a vacuous row wedged
        // between two real closures means the header is malformed, and the
        // consistency check below will say so.
        trim_boundary(A, keep);
        trim_boundary(B, keep);
        trim_boundary(C, keep);
        trim_boundary(AL, keep);
        trim_boundary(BL, keep);
        trim_boundary(CL, keep);
    }

    trim_interior(e.A1Interior, m, "A1Interior");
    trim_interior(e.B1Interior, m, "B1Interior");
    trim_interior(e.C1Interior, m, "C1Interior");
    trim_interior(e.A2Interior, m, "A2Interior");
    trim_interior(e.B2Interior, m, "B2Interior");
    trim_interior(e.C2Interior, m, "C2Interior");
    return e;
}

// ---------------------------------------------------------------------------
// Build-time Taylor consistency check.
//
// Every row of the coupled system -- interior or closure -- is a linear relation
//
//     sum_j A_j g_j + sum_j B_j w_j - sum_j C_j f_j = 0
//
// which, at unit spacing with g = f' and w = f'', must annihilate f = x^p for
// p = 0,1,2,... up to the row's design order. Failing at p <= 2 is not a
// low-order closure, it is an INCONSISTENT one: the row cannot reproduce a
// linear (eq1) or quadratic (eq2) function, so the operator does not converge at
// any resolution. That is a property of the coefficients alone, so it is cheap
// to check here and impossible to check later -- by the time it reaches
// convergence testing it looks like a mysterious accuracy loss.
//
// The tolerance is relative to the row's coefficient magnitude and sits well
// above double roundoff (a correct row lands at ~1e-15), so this fires on real
// coefficient error, not on formatting.
constexpr double kConsistencyTol = 1.0e-10;

// residual of one row on f = x^p, evaluated about node 0 with x_j = j
double row_residual(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, int p) {
    auto pw_ = [](double base, int ex) -> double {
        if (ex < 0) return 0.0;
        double r = 1.0;
        for (int k = 0; k < ex; k++) r *= base;
        return r;
    };
    double acc = 0.0;
    for (size_t j = 0; j < A.size(); j++)
        acc += A[j] * (p >= 1 ? p * pw_(static_cast<double>(j), p - 1) : 0.0);
    for (size_t j = 0; j < B.size(); j++)
        acc += B[j] *
               (p >= 2 ? p * (p - 1) * pw_(static_cast<double>(j), p - 2) : 0.0);
    for (size_t j = 0; j < C.size(); j++)
        acc -= C[j] * pw_(static_cast<double>(j), p);
    return acc;
}

double row_scale(const std::vector<double>& A, const std::vector<double>& B,
                 const std::vector<double>& C) {
    double s = 0.0;
    for (double v : A) s += std::abs(v);
    for (double v : B) s += std::abs(v);
    for (double v : C) s += std::abs(v);
    return (s > 0.0) ? s : 1.0;
}

void check_row_consistency(const std::vector<double>& A,
                           const std::vector<double>& B,
                           const std::vector<double>& C, int eq,
                           const std::string& what) {
    const double scale = row_scale(A, B, C);
    const int pmin     = (eq == 1) ? 1 : 2;  // f' must get x^1, f'' must get x^2
    for (int p = 0; p <= pmin; p++) {
        const double r = std::abs(row_residual(A, B, C, p)) / scale;
        if (r > kConsistencyTol) {
            std::ostringstream oss;
            oss << "CCFD: " << what << " is INCONSISTENT -- it does not "
                << "annihilate f = x^" << p << " (relative residual " << r
                << ", tolerance " << kConsistencyTol << "). An equation that "
                << "cannot reproduce a "
                << (eq == 1 ? "linear" : "quadratic")
                << " function has order 0 and the operator will not converge "
                << "at any resolution. Because D = A^-1 B is dense, one bad "
                << "closure row degrades every point in the block, not just "
                << "the boundary. Re-derive this row's coefficients.";
            throw std::runtime_error(oss.str());
        }
    }
}

// centered interior vectors, re-indexed to the 0..2m absolute positions the
// residual helper expects (the shift is exact: a relation that annihilates x^p
// about one origin annihilates it about any other).
void check_entries_consistency(const CCFDDiagonalEntries& e) {
    check_row_consistency(e.A1Interior, e.B1Interior, e.C1Interior, 1,
                          "interior row of equation 1 (f')");
    check_row_consistency(e.A2Interior, e.B2Interior, e.C2Interior, 2,
                          "interior row of equation 2 (f'')");
    for (size_t r = 0; r < e.A1Boundary.size(); r++)
        check_row_consistency(e.A1Boundary[r], e.B1Boundary[r], e.C1Boundary[r],
                              1, "closure row " + std::to_string(r) +
                                     " of equation 1 (f')");
    for (size_t r = 0; r < e.A2Boundary.size(); r++)
        check_row_consistency(e.A2Boundary[r], e.B2Boundary[r], e.C2Boundary[r],
                              2, "closure row " + std::to_string(r) +
                                     " of equation 2 (f'')");
    for (size_t r = 0; r < e.A1BoundaryLower.size(); r++)
        check_row_consistency(e.A1BoundaryLower[r], e.B1BoundaryLower[r],
                              e.C1BoundaryLower[r], 1,
                              "lower closure row " + std::to_string(r) +
                                  " of equation 1 (f')");
    for (size_t r = 0; r < e.A2BoundaryLower.size(); r++)
        check_row_consistency(e.A2BoundaryLower[r], e.B2BoundaryLower[r],
                              e.C2BoundaryLower[r], 2,
                              "lower closure row " + std::to_string(r) +
                                  " of equation 2 (f'')");
}

}  // namespace

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage> createCCFDMatrixSystemForSingleSize(
    const unsigned int pw, const unsigned int n,
    const CCFDDiagonalEntries* ccfdEntries, const bool skip_leftright) {
    static_assert(DerivOrder == 1 || DerivOrder == 2,
                  "CCFD operators are only defined for the 1st and 2nd "
                  "derivative");

    // which half of the coupled solution we keep: rows 2i are g (-> D1),
    // rows 2i+1 are w (-> D2). This is the S1/S2 row selection, as a strided
    // copy rather than a matmul.
    constexpr unsigned int row_off = (DerivOrder == 1) ? 0u : 1u;

    const size_t nsq               = static_cast<size_t>(n) * n;

    // Strip the generator's septadiagonal padding before anything looks at the
    // coefficients: drop the filler closure rows a narrower scheme leaves
    // behind (which would otherwise pin g/w to zero at those nodes) and narrow
    // the interior vectors to match. A scheme emitted at its true width comes
    // back unchanged. Then verify every row is at least CONSISTENT -- the rcond
    // guard further down cannot see either failure, since both leave a
    // perfectly well-conditioned system.
    const CCFDDiagonalEntries canonical = canonicalize(ccfdEntries);
    check_entries_consistency(canonical);
    ccfdEntries                    = &canonical;

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
                "CCFD: boundary padding leaves no active points at n = " +
                std::to_string(n));
        }
        const unsigned int n_fill = n - boundary_top - boundary_bottom;

        // --- the six Sengupta matrices, on the active grid ------------------
        // eq1: A1 f' + B1 f'' = C1 f, mapped to the (g, w, f) families G1,W1,F1.
        // eq2: A2 f' + B2 f'' = C2 f -> G2,W2,F2. The A/B/C names are the
        // definition notation; the coupled solve below is interleaved.
        const CCFDDiagonalEntries* e = ccfdEntries;
        std::vector<double> G1 = build_family(e->A1Interior, e->A1Boundary,
                                              e->A1BoundaryLower, n_fill,
                                              kA1Parity);
        std::vector<double> W1 = build_family(e->B1Interior, e->B1Boundary,
                                              e->B1BoundaryLower, n_fill,
                                              kB1Parity);
        std::vector<double> F1 = build_family(e->C1Interior, e->C1Boundary,
                                              e->C1BoundaryLower, n_fill,
                                              kC1Parity);
        std::vector<double> G2 = build_family(e->A2Interior, e->A2Boundary,
                                              e->A2BoundaryLower, n_fill,
                                              kA2Parity);
        std::vector<double> W2 = build_family(e->B2Interior, e->B2Boundary,
                                              e->B2BoundaryLower, n_fill,
                                              kB2Parity);
        std::vector<double> F2 = build_family(e->C2Interior, e->C2Boundary,
                                              e->C2BoundaryLower, n_fill,
                                              kC2Parity);

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
        // for CCFD the padding node needs an identity 2x2 block in A *and* a 1
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
            oss << "CCFD: LU factorization failed at n = " << n << ", variant "
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
            oss << "CCFD: system is numerically singular at n = " << n
                << ", variant " << boundary_name(b) << " (rcond = " << rcond
                << ", floor = " << kRcondFloor << "). The solve would return "
                << "plausible-looking garbage rather than fail, so it is "
                << "refused here. Most likely the two boundary closure rows "
                << "are linearly dependent -- see the warning on "
                << "CCFDDiagonalEntries. A too-small block for the closure "
                << "stencil does this too.";
            throw std::runtime_error(oss.str());
        }

        dgetrs_(&trans_n, &N2, &nrhs, A.data(), &lda, ipiv.data(), B.data(),
                &ldb, &info);
        if (info != 0) {
            throw std::runtime_error(
                "CCFD: triangular solve failed at n = " + std::to_string(n) +
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

template std::unique_ptr<DerivMatrixStorage>
createCCFDMatrixSystemForSingleSize<1>(const unsigned int pw,
                                       const unsigned int n,
                                       const CCFDDiagonalEntries* ccfdEntries,
                                       const bool skip_leftright);

template std::unique_ptr<DerivMatrixStorage>
createCCFDMatrixSystemForSingleSize<2>(const unsigned int pw,
                                       const unsigned int n,
                                       const CCFDDiagonalEntries* ccfdEntries,
                                       const bool skip_leftright);

}  // namespace dendroderivs
