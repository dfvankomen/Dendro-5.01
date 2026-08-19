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

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
};

const std::vector<PadScheme> &schemes() {
    static const std::vector<PadScheme> s = {
        {"narrow-elemlocal-7pt", 7, 0},   // today's ip_1D_* path
        {"wide-coarse-8pt", 8, 0},
        {"wide-coarse-9pt", 9, 0},
        {"wide-coarse-10pt", 10, 0},
        {"straddle-7c+2f-9pt", 7, 2},
        {"straddle-7c+3f-10pt", 7, 3},
    };
    return s;
}

/** Fill pad[0..2] = reconstructed u at X0-h, X0-2h, X0-3h. */
void reconstruct_pad(const PadScheme &s, real h, real pad[3]) {
    const real H = 2 * h;

    std::vector<real> xs;
    xs.reserve(s.n_coarse + s.n_fine);
    for (int i = 0; i < s.n_coarse; i++) xs.push_back(X0 - (real)i * H);
    for (int i = 1; i <= s.n_fine; i++) xs.push_back(X0 + (real)i * h);

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

/* ------------------------------------------------------------------ */
/* Test 1 -- prolongation order at the hanging node                    */
/* ------------------------------------------------------------------ */

TEST_CASE("prolongation order at a 2:1 hanging node") {
    const std::vector<real> &hs = h_sweep();

    std::printf(
        "\n=== Test 1: prolongation error at hanging pad node X0-h ===\n");
    std::printf("%-22s %8s %13s %13s %8s %6s\n", "scheme", "npts",
                "err(h=1/128)", "err(h=1/2187)", "order", "nfit");

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

        std::printf("%-22s %8d %13.4e %13.4e %8.3f %6u\n", s.name.c_str(),
                    s.n_coarse + s.n_fine, dbl(errs.front()), dbl(errs.back()),
                    p, n_used);
    }

    // A stencil of n points is exact for degree n-1, so its interpolation
    // error must go like h^n.
    for (size_t i = 0; i < schemes().size(); i++) {
        const PadScheme &s = schemes()[i];
        const double expected = s.n_coarse + s.n_fine;
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

int main(int argc, char **argv) {
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    const int res = ctx.run();
    write_csv();
    return res;
}
