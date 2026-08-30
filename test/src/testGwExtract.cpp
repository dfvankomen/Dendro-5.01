// Gate for the GR module's gravitational-wave extraction (GR/include/).
//
// Two jobs. First, it forces `dendro_gr::extractFarFieldPsi4` to instantiate,
// so `GR/` is compile-checked by the build rather than only wired into it --
// nothing in dendrolib compiled a GR header before this test existed.
//
// Second, it pins the structural invariants of the Lebedev/SWSH table that
// `nrswsh.h` ships. Spin-weighted spherical harmonics factor as
//
//     sYlm(theta, phi) = f_lm(theta) * exp(i*m*phi),   f_lm real,
//
// so the table's 5810 complex entries per mode carry only `ntheta` distinct
// real numbers. This test asserts that factorization against the generated
// values. It is the deletion gate for the expanded table: any compact
// replacement must reproduce these numbers, and unlike regenerating with
// SymPyGR/cog it needs no Python toolchain to run.
// Exit code 0 = pass.
#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <vector>

#include "gw_extract.h"

// force the whole template body through the compiler
template void dendro_gr::extractFarFieldPsi4<double>(
    const ot::Mesh*, const double**, unsigned int, double,
    const dendro_gr::GWExtractionConfig&);

// LEBEDEV_NUM_PTS is a #define, so it needs no using-declaration
using dendro_gr::LEBEDEV_PHI;
using dendro_gr::LEBEDEV_SWSH;
using dendro_gr::LEBEDEV_THETA;
using dendro_gr::LEBEDEV_W;

static const double TOL = 1e-13;  // roundoff on values sympy emitted at 1e-20

// the table stores l = 2 .. lmax contiguously, (2l+1) modes each
static unsigned int lmax_from_mode_count(unsigned int nmodes) {
    unsigned int acc = 0;
    for (unsigned int l = 2; l <= 64; l++) {
        acc += 2 * l + 1;
        if (acc == nmodes) return l;
        if (acc > nmodes) break;
    }
    return 0;
}

int main(int argc, char** argv) {
    const unsigned int NPTS   = LEBEDEV_NUM_PTS;
    const unsigned int NMODES = sizeof(LEBEDEV_SWSH) / sizeof(LEBEDEV_SWSH[0]);
    const unsigned int LMAX   = lmax_from_mode_count(NMODES);

    printf("[gwExtract] table: %u points, %u modes", NPTS, NMODES);
    if (!LMAX) {
        printf("\n  FAIL: %u modes is not sum_{l=2}^{lmax}(2l+1) for any lmax\n",
               NMODES);
        return 1;
    }
    printf(", l = 2 .. %u\n", LMAX);

    // distinct theta values -- the compression factor the compact form buys
    std::map<long long, unsigned int> theta_id;
    std::vector<unsigned int> theta_of_pt(NPTS);
    for (unsigned int k = 0; k < NPTS; k++) {
        const long long key = (long long)llround(LEBEDEV_THETA[k] * 1e12);
        auto it = theta_id.find(key);
        if (it == theta_id.end())
            it = theta_id.emplace(key, (unsigned int)theta_id.size()).first;
        theta_of_pt[k] = it->second;
    }
    const unsigned int NTHETA = (unsigned int)theta_id.size();
    printf("[gwExtract] distinct theta: %u  (expanded %.1fx)\n", NTHETA,
           (double)NPTS / (double)NTHETA);

    // weights must be positive and integrate the sphere to 1
    double wsum = 0.0;
    for (unsigned int k = 0; k < NPTS; k++) {
        if (!(LEBEDEV_W[k] > 0.0)) {
            printf("  FAIL: non-positive quadrature weight at point %u\n", k);
            return 1;
        }
        wsum += LEBEDEV_W[k];
    }
    if (std::fabs(wsum - 1.0) > 1e-12) {
        printf("  FAIL: quadrature weights sum to %.17g, expected 1\n", wsum);
        return 1;
    }
    printf("[gwExtract] weight sum: 1 %+.2e\n", wsum - 1.0);

    // swshTableIndex must enumerate every table slot exactly once, in order --
    // this is what lets a non-contiguous l_modes list index the table correctly
    {
        std::vector<char> hit(NMODES, 0);
        unsigned int expect = 0;
        for (unsigned int l = 2; l <= LMAX; l++) {
            for (unsigned int ms = 0; ms <= 2 * l; ms++, expect++) {
                const unsigned int idx = dendro_gr::swshTableIndex(l, ms);
                if (idx != expect || idx >= NMODES || hit[idx]) {
                    printf("  FAIL: swshTableIndex(%u,%u) = %u, expected %u\n", l,
                           ms, idx, expect);
                    return 1;
                }
                hit[idx] = 1;
            }
        }
        printf("[gwExtract] swshTableIndex: bijective over %u slots\n", expect);
    }

    // the factorization itself
    double worst_spread = 0.0, worst_imag = 0.0;
    unsigned int lm = 0;
    for (unsigned int l = 2; l <= LMAX; l++) {
        for (int m = -(int)l; m <= (int)l; m++, lm++) {
            // f_lm(theta) sampled once per distinct theta, then cross-checked
            std::vector<DendroComplex> f(NTHETA);
            std::vector<char> seen(NTHETA, 0);

            for (unsigned int k = 0; k < NPTS; k++) {
                const DendroComplex g =
                    LEBEDEV_SWSH[lm][k] *
                    std::polar(1.0, -(double)m * LEBEDEV_PHI[k]);
                const unsigned int t = theta_of_pt[k];
                if (!seen[t]) {
                    f[t]    = g;
                    seen[t] = 1;
                } else {
                    worst_spread = std::max(worst_spread, std::abs(g - f[t]));
                }
                worst_imag = std::max(worst_imag, std::fabs(g.imag()));
            }
        }
    }

    printf("[gwExtract] worst theta-spread : %.3e\n", worst_spread);
    printf("[gwExtract] worst |imag|       : %.3e\n", worst_imag);

    if (worst_spread > TOL || worst_imag > TOL) {
        printf("  FAIL: sYlm does not factor as f(theta)*exp(i*m*phi) to %.1e\n",
               TOL);
        return 1;
    }

    printf("[gwExtract] PASS -- %u modes factor to %.1e; compact form needs "
           "%u x %u reals (%.1f KB) vs the expanded table\n",
           NMODES, TOL, NMODES, NTHETA,
           NMODES * NTHETA * sizeof(double) / 1024.0);
    return 0;
}
