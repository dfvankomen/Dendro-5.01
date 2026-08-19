/**
 * @file testProlongationOrder.cpp
 * @brief h-sweep order tests for parent->child prolongation across a 2:1
 *        refinement interface.
 *
 * Mesh-free 1D reduction of the unzip pad reconstruction. Two measurements:
 *
 *   1. The order of the prolongation operator itself at a hanging (half-point)
 *      pad node.
 *   2. The order of a 6th-order centred 2nd derivative applied to fine-block
 *      data whose pad values had to be prolongated across the jump.
 *
 * Geometry (interface at X0, fine block occupies x >= X0, H = 2h):
 *
 *   coarse nodes ... X0-3H X0-2H X0-H  X0 | X0+h  X0+2h  X0+3h ... fine block
 *   fine pad index          -3    -2   -1   0
 *
 * Pad -2 lands on the coarse node X0-H and every scheme reproduces it exactly;
 * only pads -1 and -3 are genuinely interpolated. That alternation is why the
 * interface error decays over exactly three nodes.
 *
 * The coarse element that today's operator uses spans [X0-6H, X0] and its
 * seven nodes are exactly X0, X0-H, ... X0-6H. So the element-local degree-6
 * operator IS the 7-point one-sided coarse stencil, and the wide candidates
 * differ from it only in point count and centring. TEST_CASE
 * "RefElement ip_1D_* matches exact Lagrange weights" pins the library matrix
 * to that statement; the order study then runs in exact arithmetic so the
 * measurement is not limited by the double-precision coefficients.
 *
 * Arithmetic is __float128: a degree-9 stencil needs a small h to be
 * asymptotic, and at that h its error is already below long-double roundoff,
 * so there is no usable window in double or long double.
 *
 * The 2nd-derivative coefficients are hardcoded rather than taken from
 * src/derivatives, to keep this test independent of that directory.
 *
 * CSV path comes from the PROLONG_CSV environment variable
 * (default: prolongation_results.csv).
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <quadmath.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <mpi.h>

#include "mesh.h"
#include "meshUtils.h"
#include <functional>
#include <string>

#include "octUtils.h"
#include "refel.h"
#include "wideprolong.h"

namespace {

using real = __float128;

constexpr unsigned ELE_ORDER = 6;

/** Interface location. Kept away from any zero of the test function. */
constexpr real X0 = 0.3Q;

/**
 * Test function exp(A x). Deliberately not trigonometric: the sampled node
 * for bin `dist` sits at X0 + (dist-1)*h and so drifts as h changes. With a
 * sinusoid the high derivative that sets the error constant swings through
 * its zeros as the node drifts, which corrupts the fitted order. exp has
 * strictly positive, slowly varying derivatives of every order.
 *
 * A is small enough that a 10-point stencil (span 20h) is still asymptotic at
 * the coarse end of the sweep.
 */
constexpr real A_EXP = 2.0Q;

real f(real x) { return expq(A_EXP * x); }
real d2f(real x) { return A_EXP * A_EXP * expq(A_EXP * x); }

/** 6th-order centred 2nd-derivative stencil, offsets -3..3, times 1/h^2. */
const real D2_C[7] = {1.0Q / 90, -3.0Q / 20, 3.0Q / 2, -49.0Q / 18,
                      3.0Q / 2,  -3.0Q / 20, 1.0Q / 90};
constexpr int D2_R = 3;

/** Below this a measurement is quad roundoff rather than truncation. */
constexpr real ERR_FLOOR = 1e-30Q;

double dbl(real x) { return (double)x; }

/** Lagrange weights interpolating nodes xs to target xt. */
void lagrange_weights(const std::vector<real> &xs, real xt,
                      std::vector<real> &w) {
    const size_t n = xs.size();
    w.assign(n, 0.0Q);
    for (size_t i = 0; i < n; i++) {
        real num = 1.0Q;
        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;
            num *= (xt - xs[j]) / (xs[i] - xs[j]);
        }
        w[i] = num;
    }
}

/**
 * Least-squares slope of log(err) vs log(h) over the points that are above
 * the roundoff floor. n_used reports how many survived.
 */
double fit_order(const std::vector<real> &h, const std::vector<real> &err,
                 unsigned &n_used) {
    real sx = 0, sy = 0, sxx = 0, sxy = 0;
    n_used = 0;
    for (size_t i = 0; i < h.size(); i++) {
        if (!(err[i] > ERR_FLOOR)) continue;
        const real lx = logq(h[i]);
        const real ly = logq(err[i]);
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
        n_used++;
    }
    if (n_used < 3) return 0.0 / 0.0;
    const real denom = (real)n_used * sxx - sx * sx;
    if (denom == 0.0Q) return 0.0 / 0.0;
    return dbl(((real)n_used * sxy - sx * sy) / denom);
}

/**
 * How the three pad values are reconstructed.
 *
 * n_coarse : coarse nodes X0, X0-H, X0-2H, ... fed to a Lagrange fit.
 *            n_coarse = 7 reproduces today's element-local operator.
 * n_fine   : fine-block nodes X0+h, X0+2h, ... also fed in. These are real
 *            DOFs and are the only way to centre a stencil on a pad node, but
 *            using them makes the pad depend on block interior data, which is
 *            a design question and not just an accuracy knob.
 */
struct PadScheme {
    std::string name;
    int n_coarse;
    int n_fine;
    /**
     * Explicit node offsets in units of the FINE spacing h, negative into the
     * coarse side. Empty means "derive from n_coarse/n_fine", i.e. uniform
     * coarse spacing H = 2h. Non-empty is how the mixed-spacing case is
     * expressed: extending into a coarser neighbour puts its nodes at 4h, so
     * the same point count spans further and the error constant grows.
     */
    std::vector<double> offs;
};

const std::vector<PadScheme> &schemes() {
    // element nodes sit at 0,-2,-4,...,-12 in units of h (coarse spacing 2h)
    static const std::vector<double> ELEM = {0, -2, -4, -6, -8, -10, -12};
    auto with = [](std::vector<double> extra) {
        std::vector<double> v = ELEM;
        for (double e : extra) v.push_back(e);
        return v;
    };

    static const std::vector<PadScheme> s = {
        {"narrow-elemlocal-7pt", 7, 0, {}},  // today's ip_1D_* path
        {"wide-coarse-8pt", 8, 0, {}},
        {"wide-coarse-9pt", 9, 0, {}},
        {"wide-coarse-10pt", 10, 0, {}},
        {"straddle-7c+2f-9pt", 7, 2, {}},
        {"straddle-7c+3f-10pt", 7, 3, {}},
        // cross-check that the explicit-offset form reproduces the derived one
        {"unif-10pt-explicit", 0, 0, with({-14, -16, -18})},
        // extension into a COARSER neighbour: its nodes are at 4h spacing, so
        // the same point count spans 24h instead of 18h
        {"mixed-8pt-coarsenbr", 0, 0, with({-16})},
        {"mixed-9pt-coarsenbr", 0, 0, with({-16, -20})},
        {"mixed-10pt-coarsenbr", 0, 0, with({-16, -20, -24})},
    };
    return s;
}

/** Fill pad[0..2] = reconstructed u at X0-h, X0-2h, X0-3h. */
void reconstruct_pad(const PadScheme &s, real h, real pad[3]) {
    const real H = 2 * h;

    std::vector<real> xs;
    if (!s.offs.empty()) {
        xs.reserve(s.offs.size());
        for (double o : s.offs) xs.push_back(X0 + (real)o * h);
    } else {
        xs.reserve(s.n_coarse + s.n_fine);
        for (int i = 0; i < s.n_coarse; i++) xs.push_back(X0 - (real)i * H);
        for (int i = 1; i <= s.n_fine; i++) xs.push_back(X0 + (real)i * h);
    }

    std::vector<real> u(xs.size());
    for (size_t i = 0; i < xs.size(); i++) u[i] = f(xs[i]);

    const real targets[3] = {X0 - h, X0 - 2 * h, X0 - 3 * h};
    std::vector<real> w;
    for (int t = 0; t < 3; t++) {
        lagrange_weights(xs, targets[t], w);
        real acc = 0.0Q;
        for (size_t i = 0; i < xs.size(); i++) acc += w[i] * u[i];
        pad[t] = acc;
    }
}

const std::vector<real> &h_sweep() {
    static const std::vector<real> h = {
        1.0Q / 128,  1.0Q / 192,  1.0Q / 288,  1.0Q / 432,
        1.0Q / 648,  1.0Q / 972,  1.0Q / 1458, 1.0Q / 2187};
    return h;
}

/* ------------------------------------------------------------------ */
/* CSV accumulation                                                    */
/* ------------------------------------------------------------------ */

struct Row {
    std::string scheme, quantity;
    double h, error;
};
std::vector<Row> g_rows;

void add_row(const std::string &s, const std::string &q, real h, real e) {
    g_rows.push_back({s, q, dbl(h), dbl(e)});
}

void write_csv() {
    // Silent no-op when no test actually ran: doctest_discover_tests parses
    // this binary's stdout to enumerate cases, and any stray line here would
    // be registered as a phantom test.
    if (g_rows.empty()) return;

    const char *env = std::getenv("PROLONG_CSV");
    const std::string path = env ? env : "prolongation_results.csv";
    FILE *fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::printf("[warn] could not open %s for writing\n", path.c_str());
        return;
    }
    std::fprintf(fp, "scheme,quantity,h,error\n");
    for (const Row &r : g_rows)
        std::fprintf(fp, "%s,%s,%.17g,%.17g\n", r.scheme.c_str(),
                     r.quantity.c_str(), r.h, r.error);
    std::fclose(fp);
    std::printf("\nwrote %zu rows to %s\n", g_rows.size(), path.c_str());
}

}  // namespace

/* ------------------------------------------------------------------ */
/* The library operator is what we think it is                         */
/* ------------------------------------------------------------------ */

TEST_CASE("RefElement ip_1D_* matches exact Lagrange weights") {
    RefElement refel(3, ELE_ORDER);
    const unsigned nrp = ELE_ORDER + 1;

    // NOTE: these accessors return ipT_1D_*, the TRANSPOSE of ip_1D_*, so
    // ipT[i*nrp + j] = l_j(child point i).
    const double *ip0 = refel.getIMTChild0();
    const double *ip1 = refel.getIMTChild1();
    REQUIRE(ip0 != nullptr);
    REQUIRE(ip1 != nullptr);

    // Reference element nodes, uniform on [-1,1]; child sample points are
    // u_0 = (u-1)/2 and u_1 = (u+1)/2.
    std::vector<real> u(nrp);
    for (unsigned k = 0; k < nrp; k++)
        u[k] = -1.0Q + 2.0Q * (real)k / (real)ELE_ORDER;

    real max0 = 0, max1 = 0;
    std::vector<real> w;
    for (unsigned i = 0; i < nrp; i++) {
        lagrange_weights(u, 0.5Q * (u[i] - 1.0Q), w);
        for (unsigned j = 0; j < nrp; j++) {
            const real d = fabsq(w[j] - (real)ip0[i * nrp + j]);
            if (d > max0) max0 = d;
        }
        lagrange_weights(u, 0.5Q * (u[i] + 1.0Q), w);
        for (unsigned j = 0; j < nrp; j++) {
            const real d = fabsq(w[j] - (real)ip1[i * nrp + j]);
            if (d > max1) max1 = d;
        }
    }

    std::printf(
        "\n=== RefElement ip_1D_* vs exact degree-%u Lagrange weights ===\n"
        "max|ip_1D_0 - exact| = %.3e\nmax|ip_1D_1 - exact| = %.3e\n",
        ELE_ORDER, dbl(max0), dbl(max1));

    CHECK(dbl(max0) < 1e-14);
    CHECK(dbl(max1) < 1e-14);
}

/* ------------------------------------------------------------------ */
/* The wide operator                                                   */
/* ------------------------------------------------------------------ */

TEST_CASE("wide operator degenerates to the narrow one with no extension") {
    RefElement refel(3, ELE_ORDER);
    const unsigned nrp = ELE_ORDER + 1;
    const double *ipT[2] = {refel.getIMTChild0(), refel.getIMTChild1()};

    double worst = 0.0;
    for (unsigned c = 0; c < 2; c++) {
        std::vector<double> op;
        unsigned n_in = 0;
        // width is requested wide but clamps to the 7 nodes that exist, so
        // this must land exactly on today's element-local operator.
        dendro::wideprolong::build_1d(
            ELE_ORDER, c, 0, 0, dendro::wideprolong::stencil_width(ELE_ORDER),
            op, n_in);
        REQUIRE(n_in == nrp);

        for (unsigned i = 0; i < nrp; i++)
            for (unsigned j = 0; j < nrp; j++) {
                const double d =
                    std::fabs(op[i * n_in + j] - ipT[c][i * nrp + j]);
                if (d > worst) worst = d;
            }
    }
    std::printf(
        "\n=== wide operator with ext=0 vs RefElement ip_1D_* ===\n"
        "max|build_1d - ip_1D| = %.3e\n", worst);
    CHECK(worst < 1e-14);
}

TEST_CASE("wide operator is exact to its stencil degree") {
    // Polynomial exactness is the definition of the interpolation order and
    // does not depend on how the window is chosen, so this does not just
    // restate the implementation.
    struct Cfg { unsigned ext_lo, ext_hi; };
    const std::vector<Cfg> cfgs = {{0, 0}, {3, 0}, {0, 3}, {6, 0}, {3, 3}};
    const unsigned width_req = dendro::wideprolong::stencil_width(ELE_ORDER);

    std::printf("\n=== wide operator polynomial exactness ===\n");
    std::printf("%6s %6s %6s %6s %14s %14s\n", "ext_lo", "ext_hi", "n_in",
                "width", "max err deg w-1", "err deg w");

    for (const Cfg &cfg : cfgs) {
        for (unsigned c = 0; c < 2; c++) {
            std::vector<double> op;
            unsigned n_in = 0;
            dendro::wideprolong::build_1d(ELE_ORDER, c, cfg.ext_lo, cfg.ext_hi,
                                          width_req, op, n_in);
            const unsigned width = (width_req < n_in) ? width_req : n_in;

            std::vector<real> xs(n_in);
            for (unsigned j = 0; j < n_in; j++)
                xs[j] = ((real)j - (real)cfg.ext_lo) / (real)ELE_ORDER;

            real worst_ok = 0, err_over = 0;
            for (unsigned d = 0; d <= width; d++) {
                real worst = 0;
                for (unsigned i = 0; i <= ELE_ORDER; i++) {
                    const real xt = 0.5Q * (real)c +
                                    (real)i / (real)(2 * ELE_ORDER);
                    real acc = 0;
                    for (unsigned j = 0; j < n_in; j++)
                        acc += (real)op[i * n_in + j] * powq(xs[j], (real)d);
                    const real e = fabsq(acc - powq(xt, (real)d));
                    if (e > worst) worst = e;
                }
                if (d < width) {
                    if (worst > worst_ok) worst_ok = worst;
                } else {
                    err_over = worst;
                }
            }

            if (c == 0)
                std::printf("%6u %6u %6u %6u %14.3e %14.3e\n", cfg.ext_lo,
                            cfg.ext_hi, n_in, width, dbl(worst_ok),
                            dbl(err_over));

            CAPTURE(cfg.ext_lo);
            CAPTURE(cfg.ext_hi);
            CAPTURE(c);
            // exact through degree width-1 ...
            CHECK(dbl(worst_ok) < 1e-12);
            // ... and genuinely not beyond, so `width` means what it says.
            CHECK(dbl(err_over) > 1e-9);
        }
    }
}

TEST_CASE("apply_3d with no extension reproduces I3D_Parent2Child") {
    // The flag-off acceptance bar, checked at the 3D operator rather than
    // only on the 1D matrices: all 8 children, against the live library call.
    RefElement refel(3, ELE_ORDER);
    const unsigned nrp = ELE_ORDER + 1;
    const unsigned np  = nrp * nrp * nrp;

    std::vector<double> in(np);
    for (unsigned i = 0; i < np; i++)
        in[i] = std::sin(0.7 * (double)i) + 0.3 * std::cos(0.11 * (double)i);

    std::vector<double> ref(np), got(np);
    std::vector<double> im1(np), im2(np);

    double worst = 0.0;
    for (unsigned cnum = 0; cnum < 8; cnum++) {
        refel.I3D_Parent2Child(in.data(), ref.data(), cnum, im1.data(),
                               im2.data());

        std::vector<double> opx, opy, opz;
        unsigned nx_in = 0, ny_in = 0, nz_in = 0;
        const unsigned w = dendro::wideprolong::stencil_width(ELE_ORDER);
        dendro::wideprolong::build_1d(ELE_ORDER, (cnum >> 0) & 1u, 0, 0, w,
                                      opx, nx_in);
        dendro::wideprolong::build_1d(ELE_ORDER, (cnum >> 1) & 1u, 0, 0, w,
                                      opy, ny_in);
        dendro::wideprolong::build_1d(ELE_ORDER, (cnum >> 2) & 1u, 0, 0, w,
                                      opz, nz_in);

        const size_t ss =
            dendro::wideprolong::scratch_size(ELE_ORDER, nx_in, ny_in, nz_in);
        std::vector<double> s1(ss), s2(ss);

        dendro::wideprolong::apply_3d(ELE_ORDER, opx.data(), nx_in, opy.data(),
                                      ny_in, opz.data(), nz_in, in.data(),
                                      got.data(), s1.data(), s2.data());

        for (unsigned i = 0; i < np; i++) {
            const double d = std::fabs(got[i] - ref[i]);
            if (d > worst) worst = d;
        }
    }

    std::printf(
        "\n=== apply_3d(ext=0) vs RefElement::I3D_Parent2Child ===\n"
        "max abs diff over all 8 children = %.3e\n", worst);
    CHECK(worst < 1e-13);
}

TEST_CASE("apply_3d is exact on tensor polynomials to its stencil degree") {
    const unsigned nrp   = ELE_ORDER + 1;
    const unsigned width = dendro::wideprolong::stencil_width(ELE_ORDER);
    const unsigned ext   = 3;

    std::vector<double> opx, opy, opz;
    unsigned nx_in = 0, ny_in = 0, nz_in = 0;
    // extension away from the interface on each axis, as a coarse element
    // abutting a fine block actually sees it.
    dendro::wideprolong::build_1d(ELE_ORDER, 1, ext, 0, width, opx, nx_in);
    dendro::wideprolong::build_1d(ELE_ORDER, 1, ext, 0, width, opy, ny_in);
    dendro::wideprolong::build_1d(ELE_ORDER, 1, ext, 0, width, opz, nz_in);

    std::vector<double> xs(nx_in);
    for (unsigned j = 0; j < nx_in; j++)
        xs[j] = ((double)j - (double)ext) / (double)ELE_ORDER;

    // degrees chosen to sit just under the stencil width on every axis
    const unsigned da = width - 1, db = 2, dc = 1;

    std::vector<double> in((size_t)nx_in * ny_in * nz_in);
    for (unsigned k = 0; k < nz_in; k++)
        for (unsigned j = 0; j < ny_in; j++)
            for (unsigned i = 0; i < nx_in; i++)
                in[(size_t)(k * ny_in + j) * nx_in + i] =
                    std::pow(xs[i], (double)da) * std::pow(xs[j], (double)db) *
                    std::pow(xs[k], (double)dc);

    const size_t ss =
        dendro::wideprolong::scratch_size(ELE_ORDER, nx_in, ny_in, nz_in);
    std::vector<double> s1(ss), s2(ss), out((size_t)nrp * nrp * nrp);

    dendro::wideprolong::apply_3d(ELE_ORDER, opx.data(), nx_in, opy.data(),
                                  ny_in, opz.data(), nz_in, in.data(),
                                  out.data(), s1.data(), s2.data());

    double worst = 0.0;
    for (unsigned k = 0; k < nrp; k++)
        for (unsigned j = 0; j < nrp; j++)
            for (unsigned i = 0; i < nrp; i++) {
                const double xt = 0.5 + (double)i / (double)(2 * ELE_ORDER);
                const double yt = 0.5 + (double)j / (double)(2 * ELE_ORDER);
                const double zt = 0.5 + (double)k / (double)(2 * ELE_ORDER);
                const double want = std::pow(xt, (double)da) *
                                    std::pow(yt, (double)db) *
                                    std::pow(zt, (double)dc);
                const double d = std::fabs(
                    out[(size_t)(k * nrp + j) * nrp + i] - want);
                if (d > worst) worst = d;
            }

    std::printf(
        "=== apply_3d tensor-polynomial exactness (deg %u,%u,%u) ===\n"
        "max abs err = %.3e\n", da, db, dc, worst);
    CHECK(worst < 1e-11);
}

TEST_CASE("build_1d_at handles a graded (mixed-spacing) extension") {
    // A coarser neighbour contributes nodes at twice the element's spacing,
    // so the extended array is graded. Lagrange does not care, but the
    // window selection has to be coordinate-based rather than index-based;
    // this pins that.
    const unsigned int p     = ELE_ORDER;
    const unsigned int nrp   = p + 1;
    const unsigned int width = dendro::wideprolong::stencil_width(p);

    // parent spans [0,1] with nodes at j/p; a coarser low-side neighbour adds
    // nodes at -2/p, -4/p, -6/p
    std::vector<double> xs;
    for (int e = 3; e >= 1; e--) xs.push_back(-2.0 * (double)e / (double)p);
    for (unsigned int j = 0; j < nrp; j++) xs.push_back((double)j / (double)p);

    for (unsigned int c = 0; c < 2; c++) {
        std::vector<double> op;
        dendro::wideprolong::build_1d_at(p, c, xs, width, op);
        const unsigned int n_in = (unsigned int)xs.size();
        REQUIRE(op.size() == (size_t)nrp * n_in);

        real worst_ok = 0, err_over = 0;
        for (unsigned int d = 0; d <= width; d++) {
            real worst = 0;
            for (unsigned int i = 0; i < nrp; i++) {
                const real xt =
                    0.5Q * (real)c + (real)i / (real)(2 * p);
                real acc = 0;
                for (unsigned int j = 0; j < n_in; j++)
                    acc += (real)op[i * n_in + j] *
                           powq((real)xs[j], (real)d);
                const real e = fabsq(acc - powq(xt, (real)d));
                if (e > worst) worst = e;
            }
            if (d < width) {
                if (worst > worst_ok) worst_ok = worst;
            } else {
                err_over = worst;
            }
        }

        if (c == 0)
            std::printf(
                "\n=== build_1d_at, graded extension (coarser neighbour) "
                "===\nn_in=%u width=%u  max err deg<=%u = %.3e  err deg %u = "
                "%.3e\n",
                n_in, width, width - 1, dbl(worst_ok), width, dbl(err_over));

        CAPTURE(c);
        CHECK(dbl(worst_ok) < 1e-11);
        CHECK(dbl(err_over) > 1e-9);
    }
}

/* ------------------------------------------------------------------ */
/* Test 1 -- prolongation order at the hanging node                    */
/* ------------------------------------------------------------------ */

TEST_CASE("prolongation order at a 2:1 hanging node") {
    const std::vector<real> &hs = h_sweep();

    std::printf(
        "\n=== Test 1: prolongation error at hanging pad node X0-h ===\n");
    std::printf("%-22s %8s %13s %13s %8s %6s %9s\n", "scheme", "npts",
                "err(h=1/128)", "err(h=1/2187)", "order", "nfit", "sum|w|");

    std::vector<double> orders;
    for (const PadScheme &s : schemes()) {
        std::vector<real> errs(hs.size());
        for (size_t i = 0; i < hs.size(); i++) {
            real pad[3];
            reconstruct_pad(s, hs[i], pad);
            errs[i] = fabsq(pad[0] - f(X0 - hs[i]));
            add_row(s.name, "interp_pad_m1", hs[i], errs[i]);
        }
        unsigned n_used = 0;
        const double p = fit_order(hs, errs, n_used);
        orders.push_back(p);

        // Lebesgue constant at the target: how much the stencil amplifies
        // noise or roundoff in its inputs.
        real lam = 0;
        {
            const real h0 = 1.0Q / 512, H0 = 2 * h0;
            std::vector<real> xs;
            if (!s.offs.empty())
                for (double o : s.offs) xs.push_back(X0 + (real)o * h0);
            else {
                for (int i = 0; i < s.n_coarse; i++)
                    xs.push_back(X0 - (real)i * H0);
                for (int i = 1; i <= s.n_fine; i++)
                    xs.push_back(X0 + (real)i * h0);
            }
            std::vector<real> w;
            lagrange_weights(xs, X0 - h0, w);
            for (size_t i = 0; i < w.size(); i++) lam += fabsq(w[i]);
        }

        const int npts = s.offs.empty() ? (s.n_coarse + s.n_fine)
                                        : (int)s.offs.size();
        std::printf("%-22s %8d %13.4e %13.4e %8.3f %6u %9.3f\n",
                    s.name.c_str(), npts, dbl(errs.front()), dbl(errs.back()),
                    p, n_used, dbl(lam));
    }

    // A stencil of n points is exact for degree n-1, so its interpolation
    // error must go like h^n.
    for (size_t i = 0; i < schemes().size(); i++) {
        const PadScheme &s = schemes()[i];
        const double expected =
            s.offs.empty() ? (double)(s.n_coarse + s.n_fine)
                           : (double)s.offs.size();
        CAPTURE(s.name);
        CAPTURE(expected);
        CHECK(orders[i] == doctest::Approx(expected).epsilon(0.03));
    }

    // Pad -2 sits on a coarse node, so every scheme must reproduce it exactly.
    for (const PadScheme &s : schemes()) {
        real pad[3];
        const real h = 1.0Q / 512;
        reconstruct_pad(s, h, pad);
        CAPTURE(s.name);
        CHECK(dbl(fabsq(pad[1] - f(X0 - 2 * h))) < 1e-25);
    }
}

/* ------------------------------------------------------------------ */
/* Test 2 -- 2nd derivative across the interface                       */
/* ------------------------------------------------------------------ */

TEST_CASE("2nd-derivative order across a 2:1 interface") {
    const std::vector<real> &hs = h_sweep();

    // dist 1,2,3 are the fine nodes whose D2 stencil reaches a pad node;
    // dist 6 is a control whose stencil lies entirely inside the block and
    // therefore measures the bare 6th-order scheme.
    const int dists[4] = {1, 2, 3, 6};

    std::printf(
        "\n=== Test 2: 6th-order D2 at fine node dist-1, across the jump ===\n");
    std::printf("%-22s %5s %13s %13s %8s %6s\n", "scheme", "dist",
                "err(h=1/128)", "err(h=1/2187)", "order", "nfit");

    double narrow_d1 = 0.0, control = 0.0;

    for (const PadScheme &s : schemes()) {
        for (int d = 0; d < 4; d++) {
            const int centre = dists[d] - 1;
            std::vector<real> errs(hs.size());

            for (size_t i = 0; i < hs.size(); i++) {
                const real h = hs[i];
                real pad[3];
                reconstruct_pad(s, h, pad);

                real acc = 0.0Q;
                for (int o = -D2_R; o <= D2_R; o++) {
                    const int idx = centre + o;
                    const real val =
                        (idx >= 0) ? f(X0 + (real)idx * h) : pad[-idx - 1];
                    acc += D2_C[o + D2_R] * val;
                }
                acc /= (h * h);
                errs[i] = fabsq(acc - d2f(X0 + (real)centre * h));
                add_row(s.name, "d2_dist" + std::to_string(dists[d]), h,
                        errs[i]);
            }

            unsigned n_used = 0;
            const double p = fit_order(hs, errs, n_used);
            std::printf("%-22s %5d %13.4e %13.4e %8.3f %6u\n", s.name.c_str(),
                        dists[d], dbl(errs.front()), dbl(errs.back()), p,
                        n_used);

            if (s.name == "narrow-elemlocal-7pt" && dists[d] == 1)
                narrow_d1 = p;
            if (s.name == "narrow-elemlocal-7pt" && dists[d] == 6) control = p;
        }
    }

    // Methodology gate: the control never touches a pad value, so it must
    // return the bare scheme order. If this is not 6, nothing else in this
    // table means anything.
    MESSAGE("control (dist=6, no prolongation) order: " << control);
    CHECK(control == doctest::Approx(6.0).epsilon(0.02));

    // Today's path: one order short of the interior scheme.
    MESSAGE("narrow dist=1 D2 order: " << narrow_d1);
    CHECK(narrow_d1 == doctest::Approx(5.0).epsilon(0.03));
}

/* ------------------------------------------------------------------ */
/* Gather -- the extended coarse cube off a real mesh                  */
/* ------------------------------------------------------------------ */

TEST_CASE("extended coarse gather reproduces the node values it spans") {
    // Uniform single-level octree: every neighbour is same level, so the
    // probe should grant full extension on interior elements and the gathered
    // cube must equal the analytic field at the coarse nodes it covers. This
    // is what catches the element-offset and local-index arithmetic; the 2:1
    // dispatch rides on top of it.
    int npes = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    std::vector<ot::TreeNode> oct;
    createRegularOctree(oct, 3, 3, m_uiMaxDepth, MPI_COMM_WORLD);
    ot::Mesh *mesh = ot::createMesh(oct.data(), oct.size(), ELE_ORDER,
                                    MPI_COMM_WORLD, 0);
    REQUIRE(mesh != nullptr);

    // NB: inactive ranks must not return early -- the MPI_Allreduce below is
    // a collective over MPI_COMM_WORLD, so skipping it deadlocks.
    const bool active       = mesh->isActive();

    const unsigned int p    = ELE_ORDER;
    const unsigned int nrp  = p + 1;
    const unsigned int nPe  = mesh->getNumNodesPerElement();
    const unsigned int want = 3;

    // DG vector holding a trilinear-in-index field that is unique per node,
    // so any mis-indexed fetch shows up immediately.
    const std::vector<ot::TreeNode> &elems = mesh->getAllElements();
    std::vector<double> dg((size_t)mesh->getAllElements().size() * nPe, 0.0);

    auto node_key = [&](unsigned int e, unsigned int li, unsigned int lj,
                        unsigned int lk) {
        const double sz = (double)(1u << (m_uiMaxDepth - elems[e].getLevel()));
        const double x  = (double)elems[e].getX() + sz * (double)li / (double)p;
        const double y  = (double)elems[e].getY() + sz * (double)lj / (double)p;
        const double z  = (double)elems[e].getZ() + sz * (double)lk / (double)p;
        return 1.0 * x + 1024.0 * y + 1048576.0 * z;
    };

    for (unsigned int e = 0; e < elems.size(); e++)
        for (unsigned int k = 0; k < nrp; k++)
            for (unsigned int j = 0; j < nrp; j++)
                for (unsigned int i = 0; i < nrp; i++)
                    dg[(size_t)e * nPe + (k * nrp + j) * nrp + i] =
                        node_key(e, i, j, k);

    unsigned int full = 0, clipped = 0, ghost_clipped = 0;
    double worst = 0.0;

    const unsigned int e_begin = active ? mesh->getElementLocalBegin() : 0;
    const unsigned int e_end   = active ? mesh->getElementLocalEnd() : 0;

    for (unsigned int e = e_begin; e < e_end; e++) {
        unsigned int ext[6];
        const unsigned int st = mesh->probeCoarseExtension(e, want, ext);
        if (st & ot::Mesh::WPX_CLIPPED_GHOST) ghost_clipped++;

        const unsigned int nx = nrp + ext[0] + ext[1];
        const unsigned int ny = nrp + ext[2] + ext[3];
        const unsigned int nz = nrp + ext[4] + ext[5];
        if (nx == nrp + 2 * want && ny == nrp + 2 * want &&
            nz == nrp + 2 * want)
            full++;
        else
            clipped++;

        std::vector<double> cube((size_t)nx * ny * nz, 0.0);
        mesh->gatherExtendedCoarseNodes(dg.data(), e, ext, cube.data());

        // every gathered value must be the field at the node that slot
        // represents, measured in the centre element's own coordinates
        const double sz = (double)(1u << (m_uiMaxDepth - elems[e].getLevel()));
        for (unsigned int k = 0; k < nz; k++)
            for (unsigned int j = 0; j < ny; j++)
                for (unsigned int i = 0; i < nx; i++) {
                    const double x = (double)elems[e].getX() +
                                     sz * ((double)i - (double)ext[0]) /
                                         (double)p;
                    const double y = (double)elems[e].getY() +
                                     sz * ((double)j - (double)ext[2]) /
                                         (double)p;
                    const double z = (double)elems[e].getZ() +
                                     sz * ((double)k - (double)ext[4]) /
                                         (double)p;
                    const double want_v =
                        1.0 * x + 1024.0 * y + 1048576.0 * z;
                    const double d = std::fabs(
                        cube[(size_t)(k * ny + j) * nx + i] - want_v);
                    if (d > worst) worst = d;
                }
    }

    std::printf(
        "\n=== extended coarse gather on a uniform level-3 mesh ===\n"
        "npes=%d  elements with full ext=%u, clipped=%u, ghost-clipped=%u\n"
        "max |gathered - analytic| = %.3e\n",
        npes, full, clipped, ghost_clipped, worst);

    // Node keys run to ~3e7, so double roundoff alone is ~3e-9. Distinct
    // nodes differ by at least sz/p in x and 1024x / 1048576x that in y,z,
    // so any mis-indexed fetch lands >= 0.6 off -- six orders above this
    // bound. The tolerance is loose against roundoff and razor sharp against
    // the bug it exists to catch.
    if (active) CHECK(worst < 1e-6);

    // A uniform 8^3 grid has exactly 6^3 interior elements, and every one of
    // them must get full extension no matter how the domain was split. This
    // summed form is the rank-independence check: if the probe ever had to
    // fall back because a neighbour landed in a round-2 ghost, this total
    // would drop below 216 for some rank count and the wide operator would
    // silently become a function of the partition.
    // The CG-sourced wrapper is what the unzip call sites use, so check it
    // agrees bit-for-bit with the DG-sourced one on the same data. The
    // pattern is index-based; only the plumbing is under test here.
    double cg_worst = 0.0;
    if (active) {
        const unsigned int cgSz = mesh->getDegOfFreedom();
        std::vector<double> cg(cgSz);
        for (unsigned int i = 0; i < cgSz; i++)
            cg[i] = 1.0 + std::sin(0.37 * (double)i);

        std::vector<double> dg2((size_t)elems.size() * nPe, 0.0);
        std::vector<double> im1(nPe), im2(nPe), scratch(nPe);
        for (unsigned int e = 0; e < elems.size(); e++)
            mesh->getElementNodalValues(cg.data(), dg2.data() + (size_t)e * nPe,
                                        e, false, im1.data(), im2.data());

        for (unsigned int e = e_begin; e < e_end; e++) {
            unsigned int ext2[6];
            mesh->probeCoarseExtension(e, want, ext2);
            const unsigned int mx = nrp + ext2[0] + ext2[1];
            const unsigned int my = nrp + ext2[2] + ext2[3];
            const unsigned int mz = nrp + ext2[4] + ext2[5];

            std::vector<double> a((size_t)mx * my * mz, 0.0);
            std::vector<double> b((size_t)mx * my * mz, 0.0);
            mesh->gatherExtendedCoarseNodes(dg2.data(), e, ext2, a.data());
            mesh->gatherExtendedCoarseNodesCG(cg.data(), e, ext2, b.data(),
                                              scratch.data(), im1.data(),
                                              im2.data());
            for (size_t t = 0; t < a.size(); t++) {
                const double d = std::fabs(a[t] - b[t]);
                if (d > cg_worst) cg_worst = d;
            }
        }
        std::printf("CG-sourced vs DG-sourced gather: max diff = %.3e\n",
                    cg_worst);
    }
    if (active) CHECK(cg_worst == 0.0);

    unsigned int full_global = 0;
    MPI_Allreduce(&full, &full_global, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    CHECK(full_global == 216);
    if (active) CHECK(ghost_clipped == 0);
    // On one rank there is no round-2 ghost anywhere, so any ghost clipping
    // would mean the probe is misclassifying a legitimate boundary.
    if (npes == 1) CHECK(ghost_clipped == 0);

    delete mesh;
}

/* ------------------------------------------------------------------ */
/* Which axes actually have to be widened                              */
/* ------------------------------------------------------------------ */

TEST_CASE("only the differentiated axis needs widening for a pure 2nd deriv") {
    // The solver takes pure and mixed second derivatives, so no stencil reads
    // the pad's 3D corner region. That raises a structural question: for a
    // pure d2/dx2 reading the x pad slab, do the y and z interpolation errors
    // matter at all?
    //
    // They should not. Along an x-line, y and z are fixed, so the transverse
    // interpolation error is a smooth function of x on the scale of the
    // solution, and the x-difference annihilates it. Only the x error
    // oscillates node to node and gets amplified by 1/h^2.
    //
    // If that holds, widening the normal axis alone restores the order, no
    // corner or edge elements are needed, and the whole corner-consistency
    // problem disappears for pure second derivatives.
    //
    // Deliberately non-separable, so transverse errors cannot cancel by
    // construction.
    const real A = 2.0Q, B = 1.5Q, C = 1.2Q, D = 0.7Q, E = 0.5Q;
    auto F = [&](real x, real y, real z) {
        return expq(A * x + B * y + C * z + D * x * y + E * y * z);
    };
    auto Fxx = [&](real x, real y, real z) {
        const real ax = A + D * y;
        return ax * ax * F(x, y, z);
    };

    const real Y0 = 0.21Q, Z0 = 0.17Q;

    // width per axis: 7 = today's element-local, 10 = wide
    auto pad_value = [&](real h, int ox, int m, int n, unsigned wx,
                         unsigned wy, unsigned wz) {
        const real H = 2 * h;
        std::vector<real> xs(wx), ys(wy), zs(wz), wxv, wyv, wzv;

        // normal axis: one-sided into the coarse side, as at a real interface
        for (unsigned i = 0; i < wx; i++) xs[i] = X0 - (real)i * H;
        // transverse axes: centred, which is what a tangential neighbour gives
        for (unsigned j = 0; j < wy; j++)
            ys[j] = Y0 + ((real)j - (real)(wy / 2)) * H;
        for (unsigned k = 0; k < wz; k++)
            zs[k] = Z0 + ((real)k - (real)(wz / 2)) * H;

        lagrange_weights(xs, X0 + (real)ox * h, wxv);
        lagrange_weights(ys, Y0 + (real)m * h, wyv);
        lagrange_weights(zs, Z0 + (real)n * h, wzv);

        real acc = 0;
        for (unsigned k = 0; k < wz; k++)
            for (unsigned j = 0; j < wy; j++)
                for (unsigned i = 0; i < wx; i++)
                    acc += wxv[i] * wyv[j] * wzv[k] * F(xs[i], ys[j], zs[k]);
        return acc;
    };

    struct Cfg { const char *name; unsigned wx, wy, wz; };
    const std::vector<Cfg> cfgs = {
        {"all narrow (7,7,7)", 7, 7, 7},
        {"normal only (10,7,7)", 10, 7, 7},
        {"normal+1 tang (10,10,7)", 10, 10, 7},
        {"all wide (10,10,10)", 10, 10, 10},
    };

    const std::vector<real> &hs = h_sweep();
    const int m = 1, n = 1;  // transverse half-points: the hardest case

    std::printf(
        "\n=== d2/dx2 at the first fine node past the jump, per-axis width "
        "===\n");
    std::printf("%-26s %13s %13s %8s\n", "widened axes", "err(h=1/128)",
                "err(h=1/2187)", "order");

    std::vector<double> ord;
    for (const Cfg &c : cfgs) {
        std::vector<real> errs(hs.size());
        for (size_t t = 0; t < hs.size(); t++) {
            const real h = hs[t];
            real acc = 0;
            for (int o = -D2_R; o <= D2_R; o++) {
                const real v =
                    (o >= 0) ? F(X0 + (real)o * h, Y0 + (real)m * h,
                                 Z0 + (real)n * h)
                             : pad_value(h, o, m, n, c.wx, c.wy, c.wz);
                acc += D2_C[o + D2_R] * v;
            }
            acc /= (h * h);
            errs[t] = fabsq(acc - Fxx(X0, Y0 + (real)m * h, Z0 + (real)n * h));
            add_row(c.name, "d2dx2_axis_study", h, errs[t]);
        }
        unsigned nu = 0;
        const double p = fit_order(hs, errs, nu);
        ord.push_back(p);
        std::printf("%-26s %13.4e %13.4e %8.3f\n", c.name, dbl(errs.front()),
                    dbl(errs.back()), p);
    }

    // MEASURED, and it refutes the tempting simplification above: widening
    // the differentiated axis alone changes nothing, and neither does two of
    // three. All three axes are required.
    //
    // The reason the transverse errors are not annihilated: they are present
    // only on the pad points (o < 0) and absent from the exact interior
    // points (o >= 0). A one-sided patch of smooth O(h^7) error is not in the
    // null space of the D2 stencil, so it survives division by h^2 and lands
    // at O(h^5) just like the normal-axis error.
    //
    // Consequence: corner and edge elements really are needed by the gather,
    // and the corner-consistency problem cannot be sidestepped by the fact
    // that no solver stencil reads the pad's 3D corner region.
    CHECK(ord[0] == doctest::Approx(5.0).epsilon(0.05));
    CHECK(ord[1] == doctest::Approx(5.0).epsilon(0.05));
    CHECK(ord[2] == doctest::Approx(5.0).epsilon(0.05));
    CHECK(ord[3] > 6.0);
}

TEST_CASE("coarser-neighbour extension is limited by its transverse lattice") {
    // The 1D study said a graded extension into a coarser neighbour costs
    // only ~2x on the error constant. That study had no transverse dimension.
    //
    // In 3D the coarser neighbour has spacing 2H on EVERY axis, so it carries
    // nodes at only every other transverse position of the element. The
    // extension slab therefore cannot be filled at the element's transverse
    // resolution from that neighbour alone: those values have to come from a
    // 2H transverse fit, whose error is 2^(p+1) larger than the element's.
    //
    // This measures whether that transverse penalty survives the second
    // derivative, i.e. whether the coarser-neighbour route is usable at all.
    const real A = 2.0Q, B = 1.5Q, C = 1.2Q, D = 0.7Q, E = 0.5Q;
    auto F = [&](real x, real y, real z) {
        return expq(A * x + B * y + C * z + D * x * y + E * y * z);
    };
    auto Fxx = [&](real x, real y, real z) {
        const real ax = A + D * y;
        return ax * ax * F(x, y, z);
    };
    const real Y0 = 0.21Q, Z0 = 0.17Q;
    const int m = 1, n = 1;

    // transverse_h: spacing of the transverse lattice available at this
    // x-node, in units of H. 1 = the element's own, 2 = a coarser neighbour's.
    auto plane_val = [&](real xnode, real h, real tspace, unsigned wt) {
        const real H = 2 * h;
        const real ts = tspace * H;
        std::vector<real> ys(wt), zs(wt), wy, wz;
        for (unsigned j = 0; j < wt; j++)
            ys[j] = Y0 + ((real)j - (real)(wt / 2)) * ts;
        for (unsigned k = 0; k < wt; k++)
            zs[k] = Z0 + ((real)k - (real)(wt / 2)) * ts;
        lagrange_weights(ys, Y0 + (real)m * h, wy);
        lagrange_weights(zs, Z0 + (real)n * h, wz);
        real acc = 0;
        for (unsigned k = 0; k < wt; k++)
            for (unsigned j = 0; j < wt; j++)
                acc += wy[j] * wz[k] * F(xnode, ys[j], zs[k]);
        return acc;
    };

    // ext_tspace: transverse spacing available on the extension slab.
    // 1 means we pretend the coarser neighbour had a fine transverse lattice
    // (the 1D study's implicit assumption); 2 is the truth.
    auto pad_value = [&](real h, int ox, double ext_tspace, unsigned wt) {
        const real H = 2 * h;
        std::vector<real> xs;
        for (int i = 0; i <= 6; i++) xs.push_back(X0 - (real)i * H);
        // coarser neighbour: high face shared at -6H, its own nodes at 2H
        for (int e = 1; e <= 3; e++)
            xs.push_back(X0 - (real)(6 + 2 * e) * H);

        std::vector<real> wx;
        lagrange_weights(xs, X0 + (real)ox * h, wx);

        real acc = 0;
        for (size_t i = 0; i < xs.size(); i++) {
            const double ts = (i <= 6) ? 1.0 : ext_tspace;
            acc += wx[i] * plane_val(xs[i], h, ts, wt);
        }
        return acc;
    };

    struct Cfg { const char *name; double ets; unsigned wt; };
    const std::vector<Cfg> cfgs = {
        {"coarser nbr, ideal transverse (H)", 1.0, 10},
        {"coarser nbr, real transverse (2H)", 2.0, 10},
        {"coarser nbr, real transverse, narrow", 2.0, 7},
    };

    const std::vector<real> &hs = h_sweep();
    std::printf(
        "\n=== coarser-neighbour extension: transverse lattice penalty ===\n");
    std::printf("%-38s %13s %8s\n", "configuration", "err(h=1/128)", "order");

    std::vector<double> ord;
    for (const Cfg &c : cfgs) {
        std::vector<real> errs(hs.size());
        for (size_t t = 0; t < hs.size(); t++) {
            const real h = hs[t];
            real acc = 0;
            for (int o = -D2_R; o <= D2_R; o++) {
                const real v =
                    (o >= 0) ? F(X0 + (real)o * h, Y0 + (real)m * h,
                                 Z0 + (real)n * h)
                             : pad_value(h, o, c.ets, c.wt);
                acc += D2_C[o + D2_R] * v;
            }
            acc /= (h * h);
            errs[t] = fabsq(acc - Fxx(X0, Y0 + (real)m * h, Z0 + (real)n * h));
            add_row(c.name, "coarser_nbr_transverse", h, errs[t]);
        }
        unsigned nu = 0;
        const double p = fit_order(hs, errs, nu);
        ord.push_back(p);
        std::printf("%-38s %13.4e %8.3f\n", c.name, dbl(errs.front()), p);
    }

    // MEASURED: the coarser neighbour's 2H transverse spacing costs almost
    // nothing. A width-10 transverse stencil at 2H still gives O(2^10 H^10),
    // which after division by h^2 is O(H^8) -- far below the O(H^6) target.
    // So the geometric worry about the transverse lattice is real but
    // quantitatively irrelevant.
    CHECK(ord[0] > 6.0);
    CHECK(ord[1] > 6.0);

    // What does break it is transverse WIDTH. Dropping the transverse stencil
    // to the element-local 7 points returns the whole thing to O(h^5),
    // regardless of the x extension. So the extension slab must itself be
    // transversally wide, which means the gather has to reach around the
    // coarser neighbour too -- and that is what breaks the clean rectangular
    // cube the tensor apply currently assumes.
    CHECK(ord[2] == doctest::Approx(5.0).epsilon(0.05));
}

TEST_CASE("gather decimates a finer neighbour onto the coarse lattice") {
    // The uniform-mesh gather test never sees a level jump, so it cannot
    // catch the decimation index map. This one runs on a refined mesh and
    // checks every gathered value against the analytic field, which pins the
    // sub-element selection and the p-2i / 2i local index mapping.
    //
    // A linear field is used deliberately: it is represented exactly on every
    // level, so any mismatch is an indexing error rather than interpolation.
    const double L = (double)(1u << m_uiMaxDepth);
    std::function<double(double, double, double)> fr =
        [L](double x, double y, double z) {
            const double dx = (x - 0.42 * L) / (0.06 * L);
            const double dy = (y - 0.55 * L) / (0.06 * L);
            const double dz = (z - 0.47 * L) / (0.06 * L);
            return std::exp(-(dx * dx + dy * dy + dz * dz));
        };

    std::vector<ot::TreeNode> tmp;
    function2Octree(fr, tmp, m_uiMaxDepth, 1e-3, ELE_ORDER, MPI_COMM_WORLD);
    ot::Mesh *mesh = ot::createMesh(tmp.data(), tmp.size(), ELE_ORDER,
                                    MPI_COMM_WORLD, 0);
    REQUIRE(mesh != nullptr);

    if (!mesh->isActive()) {
        delete mesh;
        return;
    }

    const unsigned int p   = ELE_ORDER;
    const unsigned int nrp = p + 1;
    const unsigned int nPe = mesh->getNumNodesPerElement();
    const unsigned int want =
        dendro::wideprolong::stencil_width(ELE_ORDER) - nrp;

    auto lin = [](double x, double y, double z) {
        return 1.0 + 3.0 * x - 2.0 * y + 5.0 * z;
    };

    const std::vector<ot::TreeNode> &elems = mesh->getAllElements();
    std::vector<double> dg((size_t)elems.size() * nPe, 0.0);
    for (unsigned int e = 0; e < elems.size(); e++) {
        const double sz =
            (double)(1u << (m_uiMaxDepth - elems[e].getLevel()));
        for (unsigned int k = 0; k < nrp; k++)
            for (unsigned int j = 0; j < nrp; j++)
                for (unsigned int i = 0; i < nrp; i++)
                    dg[(size_t)e * nPe + (k * nrp + j) * nrp + i] = lin(
                        (double)elems[e].getX() + sz * (double)i / (double)p,
                        (double)elems[e].getY() + sz * (double)j / (double)p,
                        (double)elems[e].getZ() + sz * (double)k / (double)p);
    }

    double worst = 0.0;
    long with_finer = 0, checked = 0;

    for (unsigned int e = mesh->getElementLocalBegin();
         e < mesh->getElementLocalEnd(); e++) {
        unsigned int ext[6];
        mesh->probeCoarseExtension(e, want, ext,
                                   ot::Mesh::WPX_LVL_SAME |
                                       ot::Mesh::WPX_LVL_FINER);

        // did any direction land on a finer neighbour?
        bool uses_finer = false;
        const std::vector<unsigned int> &e2e = mesh->getE2EMapping();
        const unsigned int nd = mesh->getNumDirections();
        for (unsigned int d = 0; d < 6; d++) {
            if (!ext[d]) continue;
            const unsigned int nb = e2e[e * nd + d];
            if (nb != LOOK_UP_TABLE_DEFAULT && nb < elems.size() &&
                elems[nb].getLevel() == elems[e].getLevel() + 1)
                uses_finer = true;
        }
        if (uses_finer) with_finer++;

        const unsigned int nx = nrp + ext[0] + ext[1];
        const unsigned int ny = nrp + ext[2] + ext[3];
        const unsigned int nz = nrp + ext[4] + ext[5];
        std::vector<double> cube((size_t)nx * ny * nz, 0.0);
        mesh->gatherExtendedCoarseNodes(dg.data(), e, ext, cube.data());

        const double sz =
            (double)(1u << (m_uiMaxDepth - elems[e].getLevel()));
        const double H = sz / (double)p;
        for (unsigned int k = 0; k < nz; k++)
            for (unsigned int j = 0; j < ny; j++)
                for (unsigned int i = 0; i < nx; i++) {
                    const double want_v =
                        lin((double)elems[e].getX() +
                                H * ((double)i - (double)ext[0]),
                            (double)elems[e].getY() +
                                H * ((double)j - (double)ext[2]),
                            (double)elems[e].getZ() +
                                H * ((double)k - (double)ext[4]));
                    const double d =
                        std::fabs(cube[(size_t)(k * ny + j) * nx + i] - want_v);
                    if (d > worst) worst = d;
                    checked++;
                }
    }

    std::printf(
        "\n=== gather on a refined mesh (decimation path) ===\n"
        "elements using a finer neighbour: %ld, values checked %ld\n"
        "max |gathered - analytic| = %.3e\n",
        with_finer, checked, worst);

    // Same check again, but mode-aware: with a straddle the extension nodes
    // sit at H/2 rather than H, so the expected coordinates differ. Without
    // this the straddle index map is entirely untested.
    double str_worst = 0.0;
    long str_dirs = 0;
    {
        for (unsigned int e = mesh->getElementLocalBegin();
             e < mesh->getElementLocalEnd(); e++) {
            unsigned int ext[6];
            unsigned char md[6];
            mesh->probeCoarseExtension(e, want, ext,
                                       ot::Mesh::WPX_LVL_SAME |
                                           ot::Mesh::WPX_LVL_FINER,
                                       md);
            for (int d = 0; d < 6; d++)
                if (ext[d] && md[d] == ot::Mesh::WPX_EXT_STRADDLE) str_dirs++;

            const unsigned int mx = nrp + ext[0] + ext[1];
            const unsigned int my = nrp + ext[2] + ext[3];
            const unsigned int mz = nrp + ext[4] + ext[5];
            std::vector<double> cb((size_t)mx * my * mz, 0.0);
            mesh->gatherExtendedCoarseNodes(dg.data(), e, ext, cb.data(), md);

            const double szz =
                (double)(1u << (m_uiMaxDepth - elems[e].getLevel()));
            const double H = szz / (double)p;

            // coordinate of cube index i on one axis, honouring the mode
            auto axpos = [&](unsigned int i, unsigned int lo, unsigned int hi,
                             unsigned char mlo, unsigned char mhi, double c0) {
                if (i < lo) {
                    const double d =
                        (mlo == ot::Mesh::WPX_EXT_STRADDLE) ? 0.5 : 1.0;
                    return c0 - d * H * (double)(lo - i);
                }
                if (i <= lo + p) return c0 + H * (double)(i - lo);
                const double d =
                    (mhi == ot::Mesh::WPX_EXT_STRADDLE) ? 0.5 : 1.0;
                return c0 + szz + d * H * (double)(i - lo - p);
            };

            for (unsigned int k = 0; k < mz; k++)
                for (unsigned int j = 0; j < my; j++)
                    for (unsigned int i = 0; i < mx; i++) {
                        const double wv = lin(
                            axpos(i, ext[0], ext[1], md[0], md[1],
                                  (double)elems[e].getX()),
                            axpos(j, ext[2], ext[3], md[2], md[3],
                                  (double)elems[e].getY()),
                            axpos(k, ext[4], ext[5], md[4], md[5],
                                  (double)elems[e].getZ()));
                        const double dd = std::fabs(
                            cb[(size_t)(k * my + j) * mx + i] - wv);
                        if (dd > str_worst) str_worst = dd;
                    }
        }
    }
    std::printf(
        "straddle-mode gather: %ld straddle directions, max err = %.3e\n",
        str_dirs, str_worst);
    CHECK(str_dirs > 0);
    CHECK(str_worst < 1e-6);

    // The check above fills the DG array analytically, so it validates the
    // index map only. The real path sources from CG through
    // getElementNodalValues, where a finer neighbour's own hanging faces are
    // filled by the NARROW operator. With a field that no low-degree
    // interpolant reproduces, any such value shows up as a mismatch.
    // Measured twice: with the inner fetch narrow, and with it widened. The
    // gap between them says how much of the contamination the face widening
    // already removes, and therefore whether the remainder is edges.
    double cg_worst = 0.0, cg_worst_wide = 0.0;
    for (int pass = 0; pass < 2; pass++) {
        auto tf = [](double x, double y, double z) {
            return std::exp(0.013 * x) * std::cos(0.011 * y) +
                   0.5 * std::sin(0.009 * z);
        };
        std::function<double(double, double, double)> tfn = tf;
        std::vector<double> cg;
        mesh->createVector(cg, tfn);
        std::vector<double> im1(nPe), im2(nPe),
            scratch((size_t)8 * nPe);

        for (unsigned int e = mesh->getElementLocalBegin();
             e < mesh->getElementLocalEnd(); e++) {
            unsigned int ext[6];
            mesh->probeCoarseExtension(e, want, ext,
                                       ot::Mesh::WPX_LVL_SAME |
                                           ot::Mesh::WPX_LVL_FINER);
            const unsigned int mx = nrp + ext[0] + ext[1];
            const unsigned int my = nrp + ext[2] + ext[3];
            const unsigned int mz = nrp + ext[4] + ext[5];
            std::vector<double> cube((size_t)mx * my * mz, 0.0);
            mesh->gatherExtendedCoarseNodesCG(cg.data(), e, ext, cube.data(),
                                              scratch.data(), im1.data(),
                                              im2.data(), pass == 1);
            const double szz =
                (double)(1u << (m_uiMaxDepth - elems[e].getLevel()));
            const double HH = szz / (double)p;
            for (unsigned int k = 0; k < mz; k++)
                for (unsigned int j = 0; j < my; j++)
                    for (unsigned int i = 0; i < mx; i++) {
                        const double wv = tf(
                            (double)elems[e].getX() +
                                HH * ((double)i - (double)ext[0]),
                            (double)elems[e].getY() +
                                HH * ((double)j - (double)ext[2]),
                            (double)elems[e].getZ() +
                                HH * ((double)k - (double)ext[4]));
                        const double d = std::fabs(
                            cube[(size_t)(k * my + j) * mx + i] - wv);
                        double &tgt = (pass == 0) ? cg_worst : cg_worst_wide;
                        if (d > tgt) tgt = d;
                    }
        }
    }
    std::printf(
        "CG-sourced gather vs analytic (exposes interpolated inputs):\n"
        "  inner fetch narrow      max = %.3e\n"
        "  inner fetch widened     max = %.3e   (faces already widened)\n",
        cg_worst, cg_worst_wide);

    // The index map is exact; the CG path is not. That gap is narrow-operator
    // hanging-node values inside the finer neighbours, and it is why
    // decimation is off by default -- see Mesh::WPX_LVL_DEFAULT.
    CHECK(cg_worst > 1e-10);

    // the decimation path must actually be exercised, or this proves nothing
    CHECK(with_finer > 0);
    // linear field is exact on every level, so this is pure index checking
    CHECK(worst < 1e-6);

    delete mesh;
}

TEST_CASE("the wide stencil is input-limited, and centring is what fixes it") {
    // On a puncture mesh the values fed to the stencil are not exact: they
    // come through getElementNodalValues, whose hanging nodes are themselves
    // interpolated on elements that are clipped. Measured there:
    //   narrow inner fetch  5.41e-06   (= the flag-OFF pad error)
    //   widened inner fetch 1.86e-07
    //
    // So widening improves the INPUTS ~29x. What it gives back is
    // amplification: a stencil reproduces its input error scaled by its
    // Lebesgue constant, and the one-sided width-10 coarse stencil has
    // Lebesgue 15.2 against the narrow operator's 4.3.
    //
    // This perturbs the inputs by a known epsilon and checks that the output
    // error tracks Lebesgue * epsilon. If it does, chasing more coarse-side
    // reach is chasing the wrong variable -- the lever is the constant, and
    // centring the stencil is what lowers it.
    const real eps = 1e-9Q;

    struct S { const char *name; int nc, nf; };
    const std::vector<S> ss = {{"narrow-elemlocal-7pt", 7, 0},
                               {"wide-coarse-10pt", 10, 0},
                               {"straddle-7c+3f-10pt", 7, 3}};

    std::printf(
        "\n=== response to a %.0e input perturbation ===\n", dbl(eps));
    std::printf("%-24s %10s %13s %10s\n", "scheme", "sum|w|", "out err",
                "ratio");

    const real h = 1.0Q / 512, H = 2 * h;
    for (const S &sc : ss) {
        std::vector<real> xs;
        for (int i = 0; i < sc.nc; i++) xs.push_back(X0 - (real)i * H);
        for (int i = 1; i <= sc.nf; i++) xs.push_back(X0 + (real)i * h);

        std::vector<real> w;
        lagrange_weights(xs, X0 - h, w);

        real lam = 0;
        for (size_t i = 0; i < w.size(); i++) lam += fabsq(w[i]);

        // worst-case perturbation: sign-aligned with the weights
        real out = 0;
        for (size_t i = 0; i < w.size(); i++)
            out += w[i] * (w[i] >= 0 ? eps : -eps);

        std::printf("%-24s %10.3f %13.4e %10.3f\n", sc.name, dbl(lam),
                    dbl(fabsq(out)), dbl(fabsq(out) / eps));

        // the response is exactly the Lebesgue constant, by construction
        CHECK(dbl(fabsq(out) / eps) == doctest::Approx(dbl(lam)).epsilon(1e-6));
    }

    // Sharper still: in a straddle stencil only the COARSE nodes carry the
    // hanging-node contamination. The fine-side nodes are block-interior real
    // DOFs, which are exact. So the quantity that actually multiplies the
    // input error is the weight on the coarse subset alone, not the total.
    std::printf("\n=== weight carried by the CONTAMINATED (coarse) nodes ===\n");
    std::printf("%-24s %10s %12s %12s\n", "scheme", "sum|w|", "coarse-only",
                "vs narrow");
    {
        const real h2 = 1.0Q / 512, H2 = 2 * h2;
        struct S2 { const char *name; int nc, nf; };
        const std::vector<S2> zz = {{"narrow-elemlocal-7pt", 7, 0},
                                    {"wide-coarse-10pt", 10, 0},
                                    {"straddle-7c+2f-9pt", 7, 2},
                                    {"straddle-7c+3f-10pt", 7, 3}};
        double narrow_c = 0.0;
        for (const S2 &z : zz) {
            std::vector<real> xz, wz;
            for (int i = 0; i < z.nc; i++) xz.push_back(X0 - (real)i * H2);
            for (int i = 1; i <= z.nf; i++) xz.push_back(X0 + (real)i * h2);
            lagrange_weights(xz, X0 - h2, wz);
            real tot = 0, coarse = 0;
            for (size_t i = 0; i < wz.size(); i++) {
                tot += fabsq(wz[i]);
                if ((int)i < z.nc) coarse += fabsq(wz[i]);
            }
            if (narrow_c == 0.0) narrow_c = dbl(coarse);
            std::printf("%-24s %10.3f %12.3f %12.2fx\n", z.name, dbl(tot),
                        dbl(coarse), narrow_c / dbl(coarse));
        }
    }

    // The centred straddle stencil amplifies LESS than today's narrow
    // operator while also interpolating far more accurately. That is the
    // combination the coarse-side-only schemes cannot reach.
    std::vector<real> xn, xs2, wn, ws2;
    for (int i = 0; i < 7; i++) xn.push_back(X0 - (real)i * H);
    for (int i = 0; i < 7; i++) xs2.push_back(X0 - (real)i * H);
    for (int i = 1; i <= 3; i++) xs2.push_back(X0 + (real)i * h);
    lagrange_weights(xn, X0 - h, wn);
    lagrange_weights(xs2, X0 - h, ws2);
    real ln = 0, ls = 0;
    for (size_t i = 0; i < wn.size(); i++) ln += fabsq(wn[i]);
    for (size_t i = 0; i < ws2.size(); i++) ls += fabsq(ws2[i]);
    MESSAGE("Lebesgue: narrow " << dbl(ln) << " straddle " << dbl(ls));
    CHECK(dbl(ls) < dbl(ln));
}

TEST_CASE("straddle prototype: D2 across the jump with contaminated inputs") {
    // Prototype of the proposed change, under the conditions actually
    // measured on the puncture mesh rather than idealised ones.
    //
    // The coarse nodes reach the stencil through getElementNodalValues and
    // carry hanging-node error; on the puncture mesh that was 1.86e-07 with
    // the inner fetch widened. The fine-side nodes are block-interior real
    // DOFs and are exact. So the perturbation is applied to the coarse
    // subset only, which is what makes straddle behave differently from a
    // coarse-side stencil of the same width.
    const real EPS = 1.862e-07Q;  // measured, puncture mesh, widened fetch

    struct S { const char *name; int nc, nf; };
    const std::vector<S> ss = {{"narrow-elemlocal-7pt", 7, 0},
                               {"wide-coarse-10pt", 10, 0},
                               {"straddle-7c+2f-9pt", 7, 2},
                               {"straddle-7c+3f-10pt", 7, 3}};

    const std::vector<real> &hs = h_sweep();

    std::printf(
        "\n=== D2 at dist=1 with coarse inputs perturbed by %.3e ===\n",
        dbl(EPS));
    std::printf("%-24s %13s %13s %10s\n", "scheme", "err(h=1/128)",
                "err(h=1/2187)", "vs narrow");

    double narrow_err = 0.0;
    for (const S &sc : ss) {
        std::vector<real> errs(hs.size());
        for (size_t t = 0; t < hs.size(); t++) {
            const real h = hs[t], H = 2 * h;

            std::vector<real> xs;
            for (int i = 0; i < sc.nc; i++) xs.push_back(X0 - (real)i * H);
            for (int i = 1; i <= sc.nf; i++) xs.push_back(X0 + (real)i * h);

            // pad values at the three positions the D2 stencil reaches
            real pad[3];
            std::vector<real> w;
            for (int q = 0; q < 3; q++) {
                lagrange_weights(xs, X0 - (real)(q + 1) * h, w);
                real acc = 0;
                for (size_t i = 0; i < xs.size(); i++) {
                    real v = f(xs[i]);
                    // coarse subset is dirty, fine subset is exact
                    if ((int)i < sc.nc)
                        v += (w[i] >= 0 ? EPS : -EPS);
                    acc += w[i] * v;
                }
                pad[q] = acc;
            }

            real acc = 0;
            for (int o = -D2_R; o <= D2_R; o++)
                acc += D2_C[o + D2_R] *
                       ((o >= 0) ? f(X0 + (real)o * h) : pad[-o - 1]);
            acc /= (h * h);
            errs[t] = fabsq(acc - d2f(X0));
            add_row(sc.name, "d2_contaminated", h, errs[t]);
        }

        if (narrow_err == 0.0) narrow_err = dbl(errs.front());
        std::printf("%-24s %13.4e %13.4e %10.2fx\n", sc.name,
                    dbl(errs.front()), dbl(errs.back()),
                    narrow_err / dbl(errs.front()));
    }

    // Under contamination the coarse-side wide stencil is WORSE than the
    // narrow operator it replaces -- which is exactly the decimation
    // regression -- while straddle is better than both.
    std::printf(
        "(coarse-side widening loses to narrow here; straddle does not)\n");
}

/* ------------------------------------------------------------------ */
/* End to end -- unzip pad accuracy across real 2:1 interfaces         */
/* ------------------------------------------------------------------ */

TEST_CASE("unzip pad error across 2:1 interfaces") {
    // Measures what the wired call sites actually buy. The CG values are the
    // analytic field, so same-level copies and injections are exact by
    // construction and every non-zero here comes from prolongation.
    //
    // Compiled one flag setting at a time, so the OFF/ON comparison is made
    // across two builds; the number printed below is the result.
    const double L = (double)(1u << m_uiMaxDepth);

    // wavelength of a few elements, so the prolongation error is well clear
    // of roundoff without the field being unrepresentable
    const double kw = 2.0 * M_PI / L;
    const char *fp_env = std::getenv("PROLONG_FIELD");
    const bool poly    = (fp_env && std::string(fp_env) == "poly");

    // a bump drives the wavelet refinement, which is what creates the jumps
    const char *bp   = std::getenv("PROLONG_BUMP");
    const char *wp   = std::getenv("PROLONG_WTOL");
    const char *mp   = std::getenv("PROLONG_MESH");
    const double bw  = bp ? std::atof(bp) : 0.03;
    const double wt  = wp ? std::atof(wp) : 1e-4;
    const bool punc  = (mp && std::string(mp) == "puncture");

    // Schwarzschild-like option: Brill-Lindquist conformal factor on a
    // domain-200 box, which is what drives refinement in the real runs. The
    // refined region sits deep inside the domain, so unlike the bump case
    // most 2:1 interfaces are far from a domain boundary -- that is exactly
    // the difference being measured here.
    const double DOM = 200.0;
    auto phys = [L, DOM](double c) { return (c / L) * DOM - 0.5 * DOM; };
    auto chi  = [phys](double x, double y, double z) {
        const double px = phys(x), py = phys(y), pz = phys(z);
        double r = std::sqrt(px * px + py * py + pz * pz);
        if (r < 1e-2) r = 1e-2;  // the puncture itself is singular
        const double psi = 1.0 + 0.5 / r;
        return std::pow(psi, -4.0);
    };

    // The polynomial option is the sharp diagnostic: a width-10 stencil is
    // exact to degree 9, so on a degree-7 field the wide path must reach
    // roundoff wherever its input is exact. Anything above roundoff means the
    // stencil is being fed values that are themselves interpolated.
    std::function<double(double, double, double)> fn =
        [kw, L, poly, punc, chi](double x, double y, double z) {
            if (punc && !poly) return chi(x, y, z);
            if (poly) {
                const double u = x / L - 0.5;
                return u * u * u * u * u * u * u;
            }
            return std::sin(kw * x) * std::cos(kw * y) * std::sin(kw * z);
        };

    std::function<double(double, double, double)> fr =
        [L, bw, punc, chi](double x, double y, double z) {
            if (punc) return chi(x, y, z);
            const double dx = (x - 0.42 * L) / (bw * L);
            const double dy = (y - 0.55 * L) / (bw * L);
            const double dz = (z - 0.47 * L) / (bw * L);
            return std::exp(-(dx * dx + dy * dy + dz * dz));
        };

    std::vector<ot::TreeNode> tmp;
    function2Octree(fr, tmp, m_uiMaxDepth, wt, ELE_ORDER, MPI_COMM_WORLD);
    ot::Mesh *mesh = ot::createMesh(tmp.data(), tmp.size(), ELE_ORDER,
                                    MPI_COMM_WORLD, 0);
    REQUIRE(mesh != nullptr);

    const bool active = mesh->isActive();
    double worst = 0.0, sum2 = 0.0;
    long count = 0, nblk = 0;
    unsigned int lmin = 0, lmax = 0;

    long ext_full = 0, ext_part = 0, ext_none = 0;
    double contam_narrow = 0.0, contam_wide = 0.0;
    long width_hist[4] = {0, 0, 0, 0};
    long refuse[4]     = {0, 0, 0, 0};

    if (active) {
        mesh->computeMinMaxLevel(lmin, lmax);

        // How often can the wide stencil actually reach its full width? This
        // is a property of the mesh alone and is what decides whether the
        // remaining widening work is worth doing.
        {
            const unsigned int want =
                dendro::wideprolong::stencil_width(ELE_ORDER) - (ELE_ORDER + 1);
            const std::vector<unsigned int> &e2e = mesh->getE2EMapping();
            const std::vector<ot::TreeNode> &allE = mesh->getAllElements();
            const unsigned int ndir = mesh->getNumDirections();
            for (unsigned int e = mesh->getElementLocalBegin();
                 e < mesh->getElementLocalEnd(); e++) {
                unsigned int ex[6];
                mesh->probeCoarseExtension(e, want, ex);
                bool full = true, any = false;
                for (int a = 0; a < 3; a++) {
                    const unsigned int t = ex[2 * a] + ex[2 * a + 1];
                    if (t < want) full = false;
                    if (t > 0) any = true;
                }
                if (full) ext_full++;
                else if (any) ext_part++;
                else ext_none++;

                // achieved 1D width is (p+1) + min over axes of the total
                // extension, and the 1D study showed 9 vs 10 points is worth
                // ~25x, so the distribution matters more than the mean
                unsigned int mn = want;
                for (int a = 0; a < 3; a++) {
                    const unsigned int t = ex[2 * a] + ex[2 * a + 1];
                    if (t < mn) mn = t;
                }
                if (mn <= 3) width_hist[mn]++;

                // Why was a direction refused? This decides which extension
                // mechanism is worth building: decimating a finer neighbour,
                // or a non-uniform fit into a coarser one.
                for (unsigned int d = 0; d < 6; d++) {
                    if (ex[d]) continue;
                    const unsigned int nb =
                        e2e[e * ndir + d];
                    if (nb == LOOK_UP_TABLE_DEFAULT ||
                        nb >= allE.size()) {
                        refuse[0]++;  // no neighbour: domain boundary
                    } else if (allE[nb].getLevel() > allE[e].getLevel()) {
                        refuse[1]++;  // neighbour finer  -> decimation
                    } else if (allE[nb].getLevel() < allE[e].getLevel()) {
                        refuse[2]++;  // neighbour coarser -> non-uniform
                    } else {
                        refuse[3]++;  // same level: dropped by the corner rule
                    }
                }
            }
        }

        std::vector<double> cg;
        mesh->createVector(cg, fn);
        std::vector<double> uz(mesh->getDegOfFreedomUnZip(), 0.0);

        // Quality of the values the wide stencil is fed on THIS mesh, probing
        // with finer neighbours allowed so the decimation inputs are included.
        {
            const unsigned int nrp2 = ELE_ORDER + 1;
            const unsigned int nPe2 = mesh->getNumNodesPerElement();
            const unsigned int want2 =
                dendro::wideprolong::stencil_width(ELE_ORDER) - nrp2;
            const std::vector<ot::TreeNode> &el2 = mesh->getAllElements();
            std::vector<double> im1(nPe2), im2(nPe2),
                scr((size_t)8 * nPe2);
            for (int pass = 0; pass < 2; pass++) {
                double w = 0.0;
                for (unsigned int e = mesh->getElementLocalBegin();
                     e < mesh->getElementLocalEnd(); e++) {
                    unsigned int ex[6];
                    mesh->probeCoarseExtension(e, want2, ex,
                                               ot::Mesh::WPX_LVL_SAME |
                                                   ot::Mesh::WPX_LVL_FINER);
                    const unsigned int ax = nrp2 + ex[0] + ex[1];
                    const unsigned int ay = nrp2 + ex[2] + ex[3];
                    const unsigned int az = nrp2 + ex[4] + ex[5];
                    std::vector<double> cb((size_t)ax * ay * az, 0.0);
                    mesh->gatherExtendedCoarseNodesCG(cg.data(), e, ex,
                                                      cb.data(), scr.data(),
                                                      im1.data(), im2.data(),
                                                      pass == 1);
                    const double szz =
                        (double)(1u << (m_uiMaxDepth - el2[e].getLevel()));
                    const double HH = szz / (double)ELE_ORDER;
                    for (unsigned int k = 0; k < az; k++)
                        for (unsigned int j = 0; j < ay; j++)
                            for (unsigned int i = 0; i < ax; i++) {
                                const double wv =
                                    fn((double)el2[e].getX() +
                                           HH * ((double)i - (double)ex[0]),
                                       (double)el2[e].getY() +
                                           HH * ((double)j - (double)ex[2]),
                                       (double)el2[e].getZ() +
                                           HH * ((double)k - (double)ex[4]));
                                const double dd = std::fabs(
                                    cb[(size_t)(k * ay + j) * ax + i] - wv);
                                if (dd > w) w = dd;
                            }
                }
                if (pass == 0) contam_narrow = w; else contam_wide = w;
            }
        }
        std::vector<double> errmap(mesh->getDegOfFreedomUnZip(), -1.0);

        mesh->performGhostExchange(cg);
        mesh->unzip(cg.data(), uz.data(), 1);



        const std::vector<ot::Block> &blks = mesh->getLocalBlockList();
        for (size_t b = 0; b < blks.size(); b++) {
            // skip domain-boundary blocks: their pad is not a 2:1 prolongation
            if (blks[b].getBlkNodeFlag()) continue;
            nblk++;

            const ot::TreeNode bn = blks[b].getBlockNode();
            const unsigned int pW = blks[b].get1DPadWidth();
            const unsigned int lx = blks[b].getAllocationSzX();
            const unsigned int ly = blks[b].getAllocationSzY();
            const unsigned int lz = blks[b].getAllocationSzZ();
            const unsigned int of = blks[b].getOffset();
            const double hx       = blks[b].computeGridDx();
            const double hy       = blks[b].computeGridDy();
            const double hz       = blks[b].computeGridDz();

            const double x0 = (double)bn.minX() - pW * hx;
            const double y0 = (double)bn.minY() - pW * hy;
            const double z0 = (double)bn.minZ() - pW * hz;

            for (unsigned int k = 0; k < lz; k++)
                for (unsigned int j = 0; j < ly; j++)
                    for (unsigned int i = 0; i < lx; i++) {
                        // pad only: a block's own interior is a straight copy
                        const bool pad = (i < pW || i >= lx - pW ||
                                          j < pW || j >= ly - pW ||
                                          k < pW || k >= lz - pW);
                        if (!pad) continue;
                        const size_t uidx = of + (size_t)(k * ly + j) * lx + i;
                        const double e = std::fabs(
                            uz[uidx] -
                            fn(x0 + i * hx, y0 + j * hy, z0 + k * hz));
                        errmap[uidx] = e;
                        if (e > worst) worst = e;
                        sum2 += e * e;
                        count++;
                    }
        }

        if (const char *dp = std::getenv("PROLONG_DUMP")) {
            FILE *fp = std::fopen(dp, "wb");
            if (fp) {
                std::fwrite(uz.data(), sizeof(double), uz.size(), fp);
                std::fwrite(errmap.data(), sizeof(double), errmap.size(), fp);
                std::fclose(fp);
            }
        }
    }

    const double rms = (count > 0) ? std::sqrt(sum2 / (double)count) : 0.0;

    std::printf(
        "\n=== unzip vs analytic over interior blocks ===\n"
        "DENDRO_WIDE_PROLONGATION: %s\n"
        "mesh=%s  levels %u..%u, interior blocks %ld, pad points %ld\n"
        "stencil reach over local elements: full %ld (%.1f%%), partial %ld, "
        "none %ld\n"
        "max err = %.6e   rms err = %.6e\n",
#ifdef DENDRO_WIDE_PROLONGATION
        "ON",
#else
        "OFF",
#endif
        punc ? "puncture" : "bump",
        lmin, lmax, nblk, count,
        ext_full,
        100.0 * (double)ext_full /
            (double)((ext_full + ext_part + ext_none) > 0
                         ? (ext_full + ext_part + ext_none)
                         : 1),
        ext_part, ext_none, worst, rms);

    // How much work is even on the edge path? If hanging edges are rare, or
    // their owners unreachable, widening them cannot move anything.
    if (active) {
        const unsigned int EDG[12] = {
            OCT_DIR_LEFT_DOWN,  OCT_DIR_LEFT_UP,    OCT_DIR_LEFT_BACK,
            OCT_DIR_LEFT_FRONT, OCT_DIR_RIGHT_DOWN, OCT_DIR_RIGHT_UP,
            OCT_DIR_RIGHT_BACK, OCT_DIR_RIGHT_FRONT, OCT_DIR_DOWN_BACK,
            OCT_DIR_DOWN_FRONT, OCT_DIR_UP_BACK,    OCT_DIR_UP_FRONT};
        const unsigned int FAC[6] = {OCT_DIR_LEFT, OCT_DIR_RIGHT,
                                     OCT_DIR_DOWN, OCT_DIR_UP,
                                     OCT_DIR_BACK, OCT_DIR_FRONT};
        const unsigned int PR[12][2] = {
            {OCT_DIR_LEFT, OCT_DIR_DOWN},  {OCT_DIR_LEFT, OCT_DIR_UP},
            {OCT_DIR_LEFT, OCT_DIR_BACK},  {OCT_DIR_LEFT, OCT_DIR_FRONT},
            {OCT_DIR_RIGHT, OCT_DIR_DOWN}, {OCT_DIR_RIGHT, OCT_DIR_UP},
            {OCT_DIR_RIGHT, OCT_DIR_BACK}, {OCT_DIR_RIGHT, OCT_DIR_FRONT},
            {OCT_DIR_DOWN, OCT_DIR_BACK},  {OCT_DIR_DOWN, OCT_DIR_FRONT},
            {OCT_DIR_UP, OCT_DIR_BACK},    {OCT_DIR_UP, OCT_DIR_FRONT}};

        long hf = 0, he = 0, he_owner = 0;
        for (unsigned int e = mesh->getElementLocalBegin();
             e < mesh->getElementLocalEnd(); e++) {
            unsigned int cn;
            for (int d = 0; d < 6; d++)
                if (mesh->isFaceHanging(e, FAC[d], cn)) hf++;
            for (int d = 0; d < 12; d++)
                if (mesh->isEdgeHanging(e, EDG[d], cn)) {
                    he++;
                    if (mesh->wpxEdgeOwner(e, PR[d][0], PR[d][1]) !=
                        LOOK_UP_TABLE_DEFAULT)
                        he_owner++;
                }
        }
        std::printf(
            "hanging counts over local elements: faces %ld, edges %ld "
            "(owner resolved %ld)\n",
            hf, he, he_owner);
        std::printf("wide edge path: %ld calls, %ld succeeded\n",
                    ot::wpxEdgeCalls().load(), ot::wpxEdgeWins().load());
    }

    // What would each extension mechanism actually unblock? Re-probing with
    // a wider level mask answers that including the corner rule, which a
    // tally of per-direction refusal reasons cannot.
    if (active) {
        const unsigned int want =
            dendro::wideprolong::stencil_width(ELE_ORDER) - (ELE_ORDER + 1);
        struct M { const char *name; unsigned int mask; };
        const M masks[4] = {
            {"same-level only (built today)", ot::Mesh::WPX_LVL_SAME},
            {"+ decimate into finer", ot::Mesh::WPX_LVL_SAME |
                                          ot::Mesh::WPX_LVL_FINER},
            {"+ graded into coarser", ot::Mesh::WPX_LVL_SAME |
                                          ot::Mesh::WPX_LVL_COARSER},
            {"+ both", ot::Mesh::WPX_LVL_SAME | ot::Mesh::WPX_LVL_FINER |
                           ot::Mesh::WPX_LVL_COARSER}};

        std::printf("what each extension mechanism would unblock:\n");
        for (const M &mm : masks) {
            long full = 0, tot = 0;
            for (unsigned int e = mesh->getElementLocalBegin();
                 e < mesh->getElementLocalEnd(); e++) {
                unsigned int ex[6];
                mesh->probeCoarseExtension(e, want, ex, mm.mask);
                bool f = true;
                for (int a = 0; a < 3; a++)
                    if (ex[2 * a] + ex[2 * a + 1] < want) f = false;
                if (f) full++;
                tot++;
            }
            std::printf("  %-32s full reach %5ld / %5ld  (%.1f%%)\n", mm.name,
                        full, tot, 100.0 * (double)full / (double)(tot ? tot : 1));
        }
    }

    std::printf(
        "stencil input quality (max |gathered - analytic|): "
        "narrow inner fetch %.3e, widened %.3e\n",
        contam_narrow, contam_wide);
    std::printf(
        "achieved 1D stencil width over local elements: "
        "%u pts %ld, %u pts %ld, %u pts %ld, %u pts %ld\n",
        ELE_ORDER + 1 + 0, width_hist[0], ELE_ORDER + 1 + 1, width_hist[1],
        ELE_ORDER + 1 + 2, width_hist[2], ELE_ORDER + 1 + 3, width_hist[3]);

    const long rtot = refuse[0] + refuse[1] + refuse[2] + refuse[3];
    std::printf(
        "refused directions (%ld total): domain-bdy %ld (%.1f%%), "
        "nbr-finer %ld (%.1f%%), nbr-coarser %ld (%.1f%%), "
        "same-level-dropped-by-corner-rule %ld (%.1f%%)\n",
        rtot, refuse[0], 100.0 * refuse[0] / (rtot ? rtot : 1), refuse[1],
        100.0 * refuse[1] / (rtot ? rtot : 1), refuse[2],
        100.0 * refuse[2] / (rtot ? rtot : 1), refuse[3],
        100.0 * refuse[3] / (rtot ? rtot : 1));

    if (active) {
        CHECK(count > 0);
        CHECK(std::isfinite(worst));
    }

    delete mesh;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // octree globals: nothing under ot:: is usable before these are set
    {
        const char *md = std::getenv("PROLONG_MAXDEPTH");
        m_uiMaxDepth   = md ? (unsigned)std::atoi(md) : 8u;
    }
    _InitializeHcurve(m_uiDim);

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    const int res = ctx.run();
    write_csv();
    MPI_Finalize();
    return res;
}
