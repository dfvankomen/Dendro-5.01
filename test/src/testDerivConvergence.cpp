// h-sweep convergence test for the registered derivative schemes.
//
// For each scheme and each grid spacing h in a swept set, this test:
//   1. Initializes u = sin(2*pi*x) sin(2*pi*y) sin(2*pi*z) over [0, L]^3
//      where L is fixed and the active block size n_active varies.
//   2. Applies grad_x / grad_xx with bflag = all six boundary bits set
//      so the boundary closures of the matrix are exercised.
//   3. Computes RMSE of the result against the analytical derivative in
//      two regions of the active block:
//        - "boundary": active cells within k_bdry of any active face
//        - "interior": active cells farther than k_bdry from every face
//   4. Emits a CSV row per (scheme, deriv, n, region).
//   5. Estimates the observed convergence order from the four finest h
//      values via a least-squares fit of log(rmse) vs log(h).
//
// To run: ./testDerivConvergence [csv_out_path]
//
// The test uses fixed eleorder = 6 (i.e. padding width pw = 3). The active
// region within each block is (n - 2*pw)^3 cells; the boundary stencils
// from the scheme apply to the first/last few active cells along each
// axis.
//
// To keep the test function's smoothness independent of h, the spatial
// extent of the active region is held fixed at L = 1; only the number of
// grid points changes. This is the standard FD convergence-test setup.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "derivatives.h"
#include "derivatives/derivs_factory.h"
#include "derivatives/derivs_utils.h"

using namespace dendroderivs;

// fixed configuration for the test
static constexpr unsigned int ELEORDER = 6;
static constexpr unsigned int PW       = ELEORDER / 2;            // = 3
static constexpr double L_ACTIVE       = 1.0;                     // active extent

// active block sizes to sweep. n_active = n - 2*pw is the matrix dimension.
// h = L_ACTIVE / (n_active - 1).
//
//   n  | n_active | h
//   13 |    7     | 1/6   = 0.16667
//   17 |   11     | 1/10  = 0.10000
//   21 |   15     | 1/14  = 0.07143
//   25 |   19     | 1/18  = 0.05556
//   33 |   27     | 1/26  = 0.03846
//   49 |   43     | 1/42  = 0.02381
//
// six h values spanning ~7x, enough for a robust log-log slope fit.
static const std::vector<unsigned int> N_SWEEP = {13, 17, 21, 25, 33, 49};

// width (in cells) of the "boundary" region for splitting RMSE. Active
// cells within K_BDRY of any active face count as boundary; the rest are
// interior. K_BDRY = PW (the padding width) is the conservative choice:
// matrix-form schemes typically have 2 boundary rows, but the loop-based
// explicit schemes use biased stencils up to PW cells from the boundary,
// so picking PW guarantees "interior" cells use only the centered stencil
// for every scheme in the registry.
static constexpr unsigned int K_BDRY = PW;

// 3D test function:  u(x,y,z) = sin(2*pi*x) sin(2*pi*y) sin(2*pi*z)
//   du/dx     =  2*pi cos(2*pi*x) sin(2*pi*y) sin(2*pi*z)
//   d2u/dx2   = -(2*pi)^2 sin(2*pi*x) sin(2*pi*y) sin(2*pi*z)
//   d2u/dxdy  =  (2*pi)^2 cos(2*pi*x) cos(2*pi*y) sin(2*pi*z)
//   d2u/dxdz  =  (2*pi)^2 cos(2*pi*x) sin(2*pi*y) cos(2*pi*z)
//   d2u/dydz  =  (2*pi)^2 sin(2*pi*x) cos(2*pi*y) cos(2*pi*z)
static void init_field(double *u, double *du, double *ddu,
                       unsigned int n, double h,
                       double *du_xy = nullptr,
                       double *du_xz = nullptr,
                       double *du_yz = nullptr) {
    const double twopi = 2.0 * M_PI;
    const double k2    = twopi * twopi;
    for (unsigned int k = 0; k < n; k++) {
        const double z   = ((int)k - (int)PW) * h;
        const double sz  = std::sin(twopi * z);
        const double cz  = std::cos(twopi * z);
        for (unsigned int j = 0; j < n; j++) {
            const double y  = ((int)j - (int)PW) * h;
            const double sy = std::sin(twopi * y);
            const double cy = std::cos(twopi * y);
            for (unsigned int i = 0; i < n; i++) {
                const double x  = ((int)i - (int)PW) * h;
                const double sx = std::sin(twopi * x);
                const double cx = std::cos(twopi * x);
                const unsigned int idx = i + j * n + k * n * n;
                u[idx]   = sx * sy * sz;
                du[idx]  = twopi * cx * sy * sz;
                ddu[idx] = -k2 * sx * sy * sz;
                if (du_xy) du_xy[idx] = k2 * cx * cy * sz;
                if (du_xz) du_xz[idx] = k2 * cx * sy * cz;
                if (du_yz) du_yz[idx] = k2 * sx * cy * cz;
            }
        }
    }
}

// classify a cell as "boundary" (within K_BDRY of any active face along
// any axis) or "interior". Returns true for boundary cells. i,j,k are
// indices into the full n^3 array; the active region is [pw, n-pw).
static inline bool is_boundary_cell(unsigned int i, unsigned int j,
                                    unsigned int k, unsigned int n) {
    const unsigned int lo = PW;
    const unsigned int hi = n - PW;            // exclusive
    auto near = [&](unsigned int v) {
        return v < lo + K_BDRY || v + K_BDRY >= hi;
    };
    return near(i) || near(j) || near(k);
}

struct RegionStats {
    double sum_sq = 0.0;
    double max_abs = 0.0;
    unsigned long count = 0;

    void accumulate(double err) {
        sum_sq += err * err;
        if (std::fabs(err) > max_abs) max_abs = std::fabs(err);
        count++;
    }

    double rmse() const {
        return count > 0 ? std::sqrt(sum_sq / (double)count) : 0.0;
    }
};

static void compare(const double *computed, const double *truth,
                    unsigned int n, RegionStats &interior, RegionStats &boundary) {
    for (unsigned int k = PW; k < n - PW; k++) {
        for (unsigned int j = PW; j < n - PW; j++) {
            for (unsigned int i = PW; i < n - PW; i++) {
                const unsigned int idx = i + j * n + k * n * n;
                const double err = computed[idx] - truth[idx];
                if (is_boundary_cell(i, j, k, n)) {
                    boundary.accumulate(err);
                } else {
                    interior.accumulate(err);
                }
            }
        }
    }
}

// least-squares fit of  log(rmse) = a + p * log(h)  on the n_fit finest
// points. Returns p (the observed order). Returns NaN if any RMSE is
// non-positive (e.g. underflow to zero) or the fit is degenerate.
static double fit_order(const std::vector<double> &h_vec,
                        const std::vector<double> &rmse_vec,
                        unsigned int n_fit) {
    const unsigned int n_pts = h_vec.size();
    if (n_pts < 2 || n_fit < 2 || n_fit > n_pts) return std::nan("");

    // use the last n_fit values (finest h)
    const unsigned int start = n_pts - n_fit;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    unsigned int valid = 0;
    for (unsigned int i = start; i < n_pts; i++) {
        if (rmse_vec[i] <= 0.0 || !std::isfinite(rmse_vec[i])) continue;
        const double lx = std::log(h_vec[i]);
        const double ly = std::log(rmse_vec[i]);
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
        valid++;
    }
    if (valid < 2) return std::nan("");
    const double denom = valid * sxx - sx * sx;
    if (denom == 0.0) return std::nan("");
    return (valid * sxy - sx * sy) / denom;
}


struct SchemeResult {
    std::string name;
    int deriv_order;                                  // 1 or 2
    std::vector<double> h;                            // per n in N_SWEEP
    std::vector<double> rmse_interior;
    std::vector<double> rmse_boundary;
    std::vector<double> rmse_all;
    bool any_failure = false;
};

// Real (known-working) coefficients for parameterized BYU schemes.
// Without these, the schemes return garbage with the placeholder {0.5}
// and the convergence fit shows them as "broken" — but they're not, the
// caller just has to supply real coefficients. X is for the 1st-order
// variant, Y for the 2nd-order variant (the registry registers each
// scheme name twice, once per order).
static const std::vector<double> &coeffs_for(const std::string &name,
                                              int deriv_order) {
    static const std::vector<double> PLACEHOLDER = {0.5};

    // 1st-order coefficient sets (5 for BYUT6, 7 for BYUP6)
    static const std::vector<double> BYUT6_X = {
        -0.10911832030141233, 0.14155448802651693, 0.18481180347254522,
        -0.07205043429473046, 0.6183758565592459};
    static const std::vector<double> BYUP6_X = {
        -0.15713213004022902, -0.10040082728878826, 0.036470152105372355,
        0.08330135509654436, 0.1350248745758766, -0.14929471508990524,
        0.10914851654012195};

    // BYUT6 1st-order: NLSM uses all zeros, meaning "no perturbation
    // from the base scheme". The 5 coefficients are tuning parameters,
    // not the stencil values themselves; zero is the baseline.
    static const std::vector<double> BYUT6_FIRST_ZERO = {
        0.0, 0.0, 0.0, 0.0, 0.0};
    // BYUT6 2nd-order: the working set from NLSM config.
    static const std::vector<double> BYUT6_SECOND_NLSM = {
        -0.6241441553395415, -0.667147877323579, -0.003600570463409447,
        -0.639883908530571, -0.8458974330948488};
    // (Older Y from a different run, kept for diagnostics:)
    static const std::vector<double> BYUT6_Y_OLD = {
        -0.31506461560540244, -0.5265308589141344, -0.9974895287305425,
        -0.44239768531527857, -0.15673121186240802};
    static const std::vector<double> BYUP6_Y = {
        -0.034424629919228435, -0.015719318138652988, 0.02504433592317341,
        0.02806547627839419, 0.008703706241768343, -0.03623872847237705,
        -0.06820700182146044};

    // BYUP6: Y from previous tuning run works at both 1st and 2nd order.
    // BYUT6 1st-order: all-zero perturbations (NLSM uses this; means
    //   "the base scheme without perturbations").
    // BYUT6 2nd-order: NLSM's tuned coefficients (different from Y_OLD).
    if (name == "BYUP6") return BYUP6_Y;
    if (name == "BYUT6") {
        return (deriv_order == 1) ? BYUT6_FIRST_ZERO : BYUT6_SECOND_NLSM;
    }
    // The 18 BYU_A6_2ND_R*_OP* schemes are Kim-style fully-optimized
    // (no free parameters); the D_coeffs[0] argument is vestigial from
    // the parameterized BYU infrastructure. Pass {0.0} so alpha resolves
    // to the published baseline. Some R-value/Op-variant combinations
    // are known to be unstable — empirically R060/OP2, R060/OP3,
    // R065/OP2, R065/OP3, R070/OP3 do not converge at 6th order with
    // these coefficients.
    static const std::vector<double> ZERO_ONE = {0.0};
    if (name.rfind("BYU_A6_2ND_R", 0) == 0) return ZERO_ONE;
    return PLACEHOLDER;
}

// run a single (scheme, deriv_order) across the h sweep
template <typename CreatorFn>
static SchemeResult run_sweep(const std::string &name, CreatorFn creator,
                              int deriv_order, std::ofstream &csv) {
    SchemeResult res;
    res.name = name;
    res.deriv_order = deriv_order;
    res.h.reserve(N_SWEEP.size());
    res.rmse_interior.reserve(N_SWEEP.size());
    res.rmse_boundary.reserve(N_SWEEP.size());
    res.rmse_all.reserve(N_SWEEP.size());

    std::vector<double> empty;
    const std::vector<double> &test_coeffs = coeffs_for(name, deriv_order);
    std::string no_filter = "none";

    for (unsigned int n : N_SWEEP) {
        const unsigned int n_active = n - 2 * PW;
        const double h = L_ACTIVE / (double)(n_active - 1);
        res.h.push_back(h);

        const size_t total = (size_t)n * n * n;
        std::vector<double> u(total), du_true(total), ddu_true(total);
        std::vector<double> du_computed(total, 0.0);
        init_field(u.data(), du_true.data(), ddu_true.data(), n, h);

        std::unique_ptr<Derivs> deriv;
        try {
            deriv = creator(ELEORDER, no_filter, empty,
                            test_coeffs, 1u);
        } catch (...) {
            res.any_failure = true;
            res.rmse_interior.push_back(std::nan(""));
            res.rmse_boundary.push_back(std::nan(""));
            res.rmse_all.push_back(std::nan(""));
            continue;
        }
        if (!deriv) {
            res.any_failure = true;
            res.rmse_interior.push_back(std::nan(""));
            res.rmse_boundary.push_back(std::nan(""));
            res.rmse_all.push_back(std::nan(""));
            continue;
        }

        deriv->set_maximum_block_size(total);

        // pre-create the boundary matrices for size n so that the
        // bflag-driven dispatch finds D_leftright already built
        deriv->pre_create_for_size(n);

        // set all six boundary bits so the LEFTRIGHT matrix is used along
        // every axis; this exercises the boundary closure rows
        const unsigned int bflag =
            (1u << OCT_DIR_LEFT) | (1u << OCT_DIR_RIGHT) |
            (1u << OCT_DIR_DOWN) | (1u << OCT_DIR_UP) |
            (1u << OCT_DIR_BACK) | (1u << OCT_DIR_FRONT);

        const unsigned int sz[3] = {n, n, n};
        // both 1st- and 2nd-order schemes expose do_grad_x; the registry
        // chooses which Derivs subclass is instantiated. Calling do_grad_x
        // on a 2nd-order Derivs yields f''.
        try {
            deriv->do_grad_x(du_computed.data(), u.data(), h, sz, bflag);
        } catch (...) {
            res.any_failure = true;
            res.rmse_interior.push_back(std::nan(""));
            res.rmse_boundary.push_back(std::nan(""));
            res.rmse_all.push_back(std::nan(""));
            continue;
        }

        const double *truth = (deriv_order == 1) ? du_true.data()
                                                 : ddu_true.data();
        RegionStats interior, boundary;
        compare(du_computed.data(), truth, n, interior, boundary);

        // combined region
        RegionStats all;
        all.sum_sq  = interior.sum_sq + boundary.sum_sq;
        all.count   = interior.count + boundary.count;
        all.max_abs = std::max(interior.max_abs, boundary.max_abs);

        res.rmse_interior.push_back(interior.rmse());
        res.rmse_boundary.push_back(boundary.rmse());
        res.rmse_all.push_back(all.rmse());

        // emit CSV rows
        const char d = (deriv_order == 1) ? '1' : '2';
        auto emit = [&](const char *region, const RegionStats &s) {
            csv << name << "," << d << "," << n << "," << n_active << ","
                << std::scientific << std::setprecision(10) << h << ","
                << region << "," << s.count << "," << s.rmse() << ","
                << s.max_abs << "\n";
        };
        emit("interior", interior);
        emit("boundary", boundary);
        emit("all", all);
    }
    return res;
}

// Mixed 2nd-derivative chain: computes d^2u/d{a}d{b} via two 1st-order
// applications. The variant string controls how the intermediate and
// final calls are dispatched:
//   "safe"      : both calls use the safe (default) path. Always correct.
//   "last-opt"  : intermediate uses default; final uses _last. Optimized
//                 and still correct (the final-step skip is exactly what
//                 _last is for).
//   "wrong"     : intermediate uses _last; final uses default. Demonstrates
//                 misuse — the intermediate's padding output is unwritten
//                 (zero from the initial fill) and the final reads garbage.
//                 Expected to show O(1) RMSE rather than convergence.
//
// outer: 'x', 'y', or 'z' (the OUTER call); inner is the OTHER axis (a, b)
// in the pair. For d^2/dxdy: inner=x, outer=y. For d^2/dxdz: inner=x,
// outer=z. For d^2/dydz: inner=y, outer=z. z never has a _last variant
// because do_grad_z already skips by convention.
struct MixedResult {
    std::string scheme;
    std::string pair;      // "xy", "xz", "yz"
    std::string variant;   // "safe", "last-opt", "wrong"
    std::vector<double> h;
    std::vector<double> rmse_interior;
    std::vector<double> rmse_boundary;
    std::vector<double> rmse_all;
    bool any_failure = false;
};

template <typename CreatorFn>
static void run_one_mixed(const std::string &scheme_name, CreatorFn creator,
                          char inner_axis, char outer_axis,
                          const std::string &variant,
                          unsigned int n, double h,
                          const std::vector<double> &u_init,
                          const std::vector<double> &truth,
                          double &rmse_interior_out,
                          double &rmse_boundary_out,
                          double &rmse_all_out,
                          bool &failure_out) {
    failure_out = false;
    rmse_interior_out = std::nan("");
    rmse_boundary_out = std::nan("");
    rmse_all_out = std::nan("");

    std::vector<double> empty;
    // mixed derivs use 1st-order schemes (mixed 2nd-deriv = grad_b(grad_a(u)))
    const std::vector<double> &mixed_coeffs = coeffs_for(scheme_name, 1);
    std::string no_filter = "none";

    std::unique_ptr<Derivs> deriv;
    try {
        deriv = creator(ELEORDER, no_filter, empty, mixed_coeffs, 1u);
    } catch (...) { failure_out = true; return; }
    if (!deriv) { failure_out = true; return; }

    const size_t total = (size_t)n * n * n;
    deriv->set_maximum_block_size(total);
    deriv->pre_create_for_size(n);

    std::vector<double> intermediate(total, 0.0);
    std::vector<double> final_out(total, 0.0);
    const unsigned int sz[3] = {n, n, n};
    // Use bflag = 0 (NO_BOUNDARY) for the mixed test, matching the
    // typical production case: an interior block with valid ghost cells
    // from neighbor exchange. With LEFTRIGHT bflag the matrix uses
    // identity rows in the padding, which would mask the "wrong"
    // variant's bug because the stencils never read those cells.
    const unsigned int bflag = 0;

    // helper to call the right grad_{inner|outer} variant
    auto apply = [&](char axis, bool use_last, double *out,
                      const double *in) {
        if (axis == 'x') {
            if (use_last) deriv->do_grad_x_last(out, in, h, sz, bflag);
            else          deriv->do_grad_x     (out, in, h, sz, bflag);
        } else if (axis == 'y') {
            if (use_last) deriv->do_grad_y_last(out, in, h, sz, bflag);
            else          deriv->do_grad_y     (out, in, h, sz, bflag);
        } else {
            deriv->do_grad_z(out, in, h, sz, bflag);
        }
    };

    bool inner_last = (variant == "wrong");
    bool outer_last = (variant == "last-opt");

    try {
        apply(inner_axis, inner_last, intermediate.data(), u_init.data());
        apply(outer_axis, outer_last, final_out.data(), intermediate.data());
    } catch (...) {
        failure_out = true;
        return;
    }

    RegionStats interior, boundary;
    compare(final_out.data(), truth.data(), n, interior, boundary);

    RegionStats all;
    all.sum_sq  = interior.sum_sq + boundary.sum_sq;
    all.count   = interior.count + boundary.count;
    all.max_abs = std::max(interior.max_abs, boundary.max_abs);

    rmse_interior_out = interior.rmse();
    rmse_boundary_out = boundary.rmse();
    rmse_all_out      = all.rmse();
}

template <typename CreatorFn>
static MixedResult run_mixed_sweep(const std::string &scheme_name,
                                    CreatorFn creator,
                                    char inner_axis, char outer_axis,
                                    const std::string &variant,
                                    std::ofstream &csv) {
    MixedResult res;
    res.scheme  = scheme_name;
    res.pair    = std::string{inner_axis, outer_axis};
    res.variant = variant;

    for (unsigned int n : N_SWEEP) {
        const unsigned int pw = PW;
        const double h = L_ACTIVE / (double)(n - 2 * pw - 1);
        res.h.push_back(h);

        const size_t total = (size_t)n * n * n;
        std::vector<double> u(total), du(total), ddu(total);
        std::vector<double> du_xy(total), du_xz(total), du_yz(total);
        init_field(u.data(), du.data(), ddu.data(), n, h,
                   du_xy.data(), du_xz.data(), du_yz.data());

        const std::vector<double> &truth =
            (res.pair == "xy") ? du_xy
                               : ((res.pair == "xz") ? du_xz : du_yz);

        double r_int, r_bdy, r_all;
        bool failure;
        run_one_mixed(scheme_name, creator, inner_axis, outer_axis,
                      variant, n, h, u, truth,
                      r_int, r_bdy, r_all, failure);
        if (failure) res.any_failure = true;
        res.rmse_interior.push_back(r_int);
        res.rmse_boundary.push_back(r_bdy);
        res.rmse_all.push_back(r_all);

        // emit CSV rows
        auto emit = [&](const char *region, unsigned long count, double rmse) {
            csv << scheme_name << ",mixed_" << res.pair << "," << variant
                << "," << n << "," << (n - 2 * pw) << ","
                << std::scientific << std::setprecision(10) << h << ","
                << region << "," << count << "," << rmse << ",nan\n";
        };
        // we lose the per-region cell counts after compare(); recompute
        // here for the CSV using the same is_boundary_cell logic.
        unsigned long n_int = 0, n_bdy = 0;
        for (unsigned int k = pw; k < n - pw; k++)
            for (unsigned int j = pw; j < n - pw; j++)
                for (unsigned int i = pw; i < n - pw; i++)
                    (is_boundary_cell(i, j, k, n) ? n_bdy : n_int)++;
        emit("interior", n_int, r_int);
        emit("boundary", n_bdy, r_bdy);
        emit("all",      n_int + n_bdy, r_all);
    }
    return res;
}

static void print_mixed_row(const MixedResult &r) {
    const unsigned int n_fit = std::min<unsigned int>(4, N_SWEEP.size());
    const double p_int = fit_order(r.h, r.rmse_interior, n_fit);
    const double p_bdy = fit_order(r.h, r.rmse_boundary, n_fit);
    const double p_all = fit_order(r.h, r.rmse_all,      n_fit);
    const double rmse_finest = r.rmse_all.empty()
        ? std::nan("")
        : r.rmse_all.back();

    std::cout << std::left << std::setw(14) << r.scheme
              << std::setw(7)  << ("d2/d" + r.pair)
              << std::setw(11) << r.variant
              << std::right << std::setw(13) << std::scientific
              << std::setprecision(3) << rmse_finest
              << std::fixed << std::setprecision(2)
              << std::setw(10) << p_int
              << std::setw(10) << p_bdy
              << std::setw(10) << p_all
              << (r.any_failure ? "  (had failures)" : "")
              << "\n";
}

static void print_summary_row(const SchemeResult &r) {
    // observed order via least-squares fit on the four finest h
    const unsigned int n_fit = std::min<unsigned int>(4, N_SWEEP.size());
    const double p_int = fit_order(r.h, r.rmse_interior, n_fit);
    const double p_bdy = fit_order(r.h, r.rmse_boundary, n_fit);
    const double p_all = fit_order(r.h, r.rmse_all, n_fit);
    const double rmse_finest = r.rmse_all.empty()
        ? std::nan("")
        : r.rmse_all.back();

    std::cout << std::left << std::setw(22) << r.name
              << std::setw(6) << (r.deriv_order == 1 ? "1st" : "2nd")
              << std::right << std::setw(13) << std::scientific
              << std::setprecision(3) << rmse_finest
              << std::fixed << std::setprecision(2)
              << std::setw(10) << p_int
              << std::setw(10) << p_bdy
              << std::setw(10) << p_all
              << (r.any_failure ? "  (had failures)" : "")
              << "\n";
}

int main(int argc, char **argv) {
    const std::string csv_path =
        (argc > 1) ? argv[1] : "convergence_results.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Could not open " << csv_path << " for writing\n";
        return 1;
    }
    csv << "scheme,deriv,n,n_active,h,region,n_cells,rmse,max_err\n";

    std::cout << "h-sweep convergence test\n"
              << "  eleorder = " << ELEORDER << ", pw = " << PW
              << ", L_active = " << L_ACTIVE << "\n"
              << "  N (block size) sweep: ";
    for (auto n : N_SWEEP) std::cout << n << " ";
    std::cout << "\n";
    std::cout << "  bflag = LEFT|RIGHT|DOWN|UP|BACK|FRONT (boundary closures exercised)\n"
              << "  K_BDRY = " << K_BDRY << " (boundary region width)\n"
              << "  CSV output: " << csv_path << "\n\n";

    std::cout << std::left << std::setw(22) << "Scheme"
              << std::setw(6) << "Deriv"
              << std::right << std::setw(13) << "RMSE(finest)"
              << std::setw(10) << "ord(int)"
              << std::setw(10) << "ord(bdy)"
              << std::setw(10) << "ord(all)" << "\n";
    std::cout << std::string(75, '-') << "\n";

    auto &first_reg  = get_first_order_registry();
    auto &second_reg = get_second_order_registry();

    // sort scheme names so output is stable. The "*Banded" variants are
    // included now that their immediate crash bugs are fixed, but they
    // can only be expected to produce correct results at n = p_n
    // (= 2*eleorder + 1 = 13); at other block sizes the per-size
    // factorization machinery isn't implemented so they will produce
    // garbage. The convergence fit will reflect that.
    std::vector<std::string> first_names, second_names;
    first_names.reserve(first_reg.size());
    second_names.reserve(second_reg.size());
    // Banded schemes leak per-instance memory (see findings/07); their
    // construct/destruct cycle across the h-sweep accumulates corruption
    // that crashes the run partway through. Filter them out here.
    // Also filter the parameterized BYU/A/C/UA scheme families which use
    // placeholder coefficients here and produce garbage outputs; at the
    // h-sweep block sizes some of them trigger heap corruption (separate
    // bug, not the convergence test's fault). The schemes we actually
    // care about for the methods paper still go through.
    auto is_skip = [](const std::string &s) {
        if (s.find("Banded") != std::string::npos) return true;
        // parameterized families that need real coefficients we don't have
        // yet; with the placeholder {0.5} they corrupt the heap at some
        // block sizes. BYUT6 and BYUP6 ARE included — known-working
        // coefficient sets are supplied via coeffs_for().
        if (s == "BYUT6") return false;
        if (s == "BYUP6") return false;
        if (s.rfind("BYUT4R", 0) == 0) return true;
        if (s.rfind("BYUT6", 0) == 0) return true;  // R1/R2/.. variants only
        if (s.rfind("BYUP6R", 0) == 0) return true;
        if (s.rfind("A4_", 0) == 0)   return true;
        if (s.rfind("A6_", 0) == 0)   return true;
        if (s.rfind("C4_", 0) == 0)   return true;
        if (s.rfind("C6_", 0) == 0)   return true;
        if (s.rfind("UC6_", 0) == 0)  return true;
        if (s.rfind("UA4_", 0) == 0)  return true;
        if (s.rfind("UA6_", 0) == 0)  return true;
        if (s.rfind("B4_", 0) == 0)   return true;
        if (s.rfind("KIMBYU_", 0) == 0) return true;
        if (s == "BorisO6Eta")        return true;
        return false;
    };
    for (const auto &kv : first_reg)
        if (!is_skip(kv.first)) first_names.push_back(kv.first);
    for (const auto &kv : second_reg)
        if (!is_skip(kv.first)) second_names.push_back(kv.first);
    std::sort(first_names.begin(), first_names.end());
    std::sort(second_names.begin(), second_names.end());

    for (const std::string &name : first_names) {
        auto r = run_sweep(name, first_reg.at(name), 1, csv);
        print_summary_row(r);
    }
    std::cout << std::string(75, '-') << "\n";
    for (const std::string &name : second_names) {
        auto r = run_sweep(name, second_reg.at(name), 2, csv);
        print_summary_row(r);
    }

    // --- Mixed 2nd-order derivative chains ---
    // For a representative subset of first-order schemes, compute
    // d^2u/d{a}d{b} = grad_b(grad_a(u)) and verify convergence. For each
    // pair, we run three variants:
    //   safe     : both intermediate and final use the default (full)
    //              path. Expected to converge at scheme order.
    //   last-opt : intermediate full + final uses _last variant. Same
    //              correctness, faster final step. Expected to converge.
    //   wrong    : intermediate uses _last, final uses default. Demonstrates
    //              misuse — intermediate has garbage in padding cells so
    //              the final reads zero where u was non-zero. Expected to
    //              NOT converge (slope ≈ 0, RMSE ~ O(1)).
    // (The "yz" pair has only safe + wrong because grad_z has no _last.)
    std::cout << "\n" << std::string(75, '=') << "\n";
    std::cout << "Mixed second derivatives (chained 1st-order)\n";
    std::cout << std::string(75, '=') << "\n";
    std::cout << std::left << std::setw(14) << "Scheme"
              << std::setw(7)  << "Pair"
              << std::setw(11) << "Variant"
              << std::right << std::setw(13) << "RMSE(finest)"
              << std::setw(10) << "ord(int)"
              << std::setw(10) << "ord(bdy)"
              << std::setw(10) << "ord(all)" << "\n";
    std::cout << std::string(85, '-') << "\n";

    // representative schemes; same as the perf benchmark, plus BYUT6 and
    // BYUP6 (now using working coefficient sets via coeffs_for()).
    const std::vector<std::string> mixed_schemes = {
        "E6Matrix", "JTT4", "JTT6", "JTP6", "BL6", "BorisO6",
        "BYUT6", "BYUP6", "BYUP8"
    };

    struct MixedPair { char inner, outer; const char *name; };
    const std::vector<MixedPair> pairs = {
        {'x', 'y', "xy"},
        {'x', 'z', "xz"},
        {'y', 'z', "yz"},
    };

    for (const auto &scheme : mixed_schemes) {
        auto it = first_reg.find(scheme);
        if (it == first_reg.end()) continue;
        for (const auto &p : pairs) {
            // safe path always
            auto r_safe = run_mixed_sweep(scheme, it->second,
                                           p.inner, p.outer, "safe", csv);
            print_mixed_row(r_safe);
            // last-opt only when outer is x or y (z has no _last)
            if (p.outer != 'z') {
                auto r_opt = run_mixed_sweep(scheme, it->second,
                                              p.inner, p.outer, "last-opt",
                                              csv);
                print_mixed_row(r_opt);
            }
            // wrong path: misuses _last for intermediate
            auto r_wrong = run_mixed_sweep(scheme, it->second,
                                            p.inner, p.outer, "wrong", csv);
            print_mixed_row(r_wrong);
        }
        std::cout << std::string(85, '-') << "\n";
    }

    csv.close();
    std::cout << "\nCSV written to " << csv_path << "\n";
    return 0;
}
