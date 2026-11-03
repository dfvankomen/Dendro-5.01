// After a failed AH3 find the reseeded search radius must never shrink.
// Uses AEH_BHaHAHA::common_horizon_reseed_radius(), no mesh or solve.
// Exit code 0 = pass.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <limits>

#include "aeh_bhahaha.h"

using dendro_aeh::AEH_BHaHAHA;
static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { printf("  FAIL: %s\n", what); failures++; }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const double R = 8.0, pad = 0.25;
    const double r_geom_frozen = 2.0;  // from frozen post-merger AH1/AH2

    // 1) failure handler sets the guess to R; the reseed must keep it at R
    check(AEH_BHaHAHA::common_horizon_reseed_radius(r_geom_frozen, true, 3.0,
                                                    R, pad, R) == R,
          "full-sphere guess from the failure handler survives the reseed");

    // 2) sequence of failed finds, never decreasing, >= last good r3 + pad
    double r_prev = 3.0, r3 = 3.0;
    for (int k = 0; k < 6; ++k) {
        double r = AEH_BHaHAHA::common_horizon_reseed_radius(
            r_geom_frozen, true, r3, r_prev, pad, R);
        printf("  failed find %d: r_prev=%.3f -> r_max=%.3f (r_geom=%.3f)\n",
               k, r_prev, r, r_geom_frozen);
        check(r >= r_prev, "radius non-decreasing across failed finds");
        check(r >= std::min(r3 + pad, R), "radius >= last good r3 + padding");
        check(r <= R, "radius capped at MAX_SEARCH_RADIUS");
        r_prev = r;
        r3     = 0.0;  // history wiped by the reseed
    }

    // 3) first activation / sanity wipe: plain geometric value as before
    check(AEH_BHaHAHA::common_horizon_reseed_radius(r_geom_frozen, false, 3.0,
                                                    R, pad, R) == r_geom_frozen,
          "no failure -> geometric radius unchanged");
    // 4) cap and bad inputs
    check(AEH_BHaHAHA::common_horizon_reseed_radius(20.0, false, 0, 0, pad,
                                                    R) == R, "cap at R");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double r = AEH_BHaHAHA::common_horizon_reseed_radius(nan, true, nan, -1.0,
                                                         pad, R);
    check(std::isfinite(r) && r > 0.0 && r <= R, "non-finite inputs");

    printf(failures ? "testAehCommonReseed: FAILED (%d)\n"
                    : "testAehCommonReseed: PASS%.0d\n", failures);
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
