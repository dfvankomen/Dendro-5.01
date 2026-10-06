/**
 * @file punctureTracker.h
 * @brief Moving-puncture (black-hole) position tracking.
 *
 * Integrates the puncture coordinates along the shift, dx^i/dt = -beta^i,
 * with forward Euler. Theory-agnostic: needs only the mesh, the zipped
 * evolution variables, and the indices of the three shift components. Mirrors
 * the GW-extraction pattern (dendro_gr::extractFarFieldPsi4) -- the solver
 * wires the config and calls into here.
 *
 * The puncture generally lives off-grid and may be owned by a single MPI rank;
 * shift interpolation uses the shared octree interpolator
 * (ot::da::interpolateToCoords) and the result is reduced to root and
 * broadcast so every rank ends with identical updated locations.
 */

#ifndef DENDRO_GR_PUNCTURE_TRACKER_H
#define DENDRO_GR_PUNCTURE_TRACKER_H

#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "daUtils.h"
#include "dendro.h"
#include "mesh.h"
#include "point.h"

namespace dendro_gr {

/**
 * @brief Advance puncture locations by one forward-Euler step of dx^i/dt =
 *        -beta^i.
 *
 * @param pMesh         active octree mesh.
 * @param zipVars       zipped evolution variables (indexed by the var enum).
 * @param beta_idx      indices of beta^{0,1,2} within @p zipVars.
 * @param in            current puncture locations (size @p num_bhs).
 * @param out           updated puncture locations (size @p num_bhs); may alias
 *                      a different buffer than @p in.
 * @param num_bhs       number of punctures.
 * @param grid_limits   {min,max} of the octree (integer) grid.
 * @param domain_limits {min,max} of the physical computational domain.
 * @param dt            time increment since the last update.
 */
inline void trackPunctures(const ot::Mesh* pMesh, double** zipVars,
                           const unsigned int beta_idx[3], const Point* in,
                           Point* out, unsigned int num_bhs,
                           const Point grid_limits[2],
                           const Point domain_limits[2], double dt) {
    const unsigned int total = num_bhs * 3;

    std::vector<double> beta0(num_bhs, 0.0), beta1(num_bhs, 0.0),
        beta2(num_bhs, 0.0);
    std::vector<unsigned int> vidx0, vidx1, vidx2;
    std::vector<double> beta_interleaved(total, 0.0);
    std::vector<double> bh_pts(total, 0.0);

    for (unsigned int b = 0; b < num_bhs; ++b) {
        bh_pts[b * 3 + 0] = in[b].x();
        bh_pts[b * 3 + 1] = in[b].y();
        bh_pts[b * 3 + 2] = in[b].z();
    }

    // only ranks that own a puncture point return a non-empty validIndices
    if (pMesh->isActive()) {
        ot::da::interpolateToCoords(pMesh, zipVars[beta_idx[0]], bh_pts.data(),
                                    total, grid_limits, domain_limits,
                                    beta0.data(), vidx0);
        ot::da::interpolateToCoords(pMesh, zipVars[beta_idx[1]], bh_pts.data(),
                                    total, grid_limits, domain_limits,
                                    beta1.data(), vidx1);
        ot::da::interpolateToCoords(pMesh, zipVars[beta_idx[2]], bh_pts.data(),
                                    total, grid_limits, domain_limits,
                                    beta2.data(), vidx2);
        for (unsigned int b : vidx0) {
            beta_interleaved[b * 3 + 0] = beta0[b];
            beta_interleaved[b * 3 + 1] = beta1[b];
            beta_interleaved[b * 3 + 2] = beta2[b];
        }
    }

    // sum the (rank-local) shifts to root and broadcast to everyone
    std::vector<double> global_beta(total, 0.0);
    const int root_rank = 0;
    MPI_Reduce(beta_interleaved.data(), global_beta.data(), total, MPI_DOUBLE,
               MPI_SUM, root_rank, pMesh->getMPIGlobalCommunicator());
    MPI_Bcast(global_beta.data(), total, MPI_DOUBLE, root_rank,
              pMesh->getMPIGlobalCommunicator());

    for (unsigned int b = 0; b < num_bhs; ++b) {
        out[b] = Point(in[b].x() - global_beta[b * 3 + 0] * dt,
                       in[b].y() - global_beta[b * 3 + 1] * dt,
                       in[b].z() - global_beta[b * 3 + 2] * dt);
    }
}

/**
 * @brief Append the current puncture locations to "<prefix>_BHLocations.dat"
 *        (rank 0 only; tab-separated, full double precision).
 */
inline void writeBHTrajectory(const ot::Mesh* pMesh, const Point* locs,
                              unsigned int num_bhs, unsigned int step,
                              double time, const std::string& prefix) {
    if (pMesh->getMPIRankGlobal() != 0) return;

    std::ofstream f(prefix + "_BHLocations.dat", std::ofstream::app);
    if (!f.is_open()) return;

    if (step == 0) {
        f << "TimeStep\ttime";
        for (unsigned int b = 0; b < num_bhs; ++b)
            f << "\tbh" << b << "_x\tbh" << b << "_y\tbh" << b << "_z";
        f << "\n";
    }
    f << std::setprecision(std::numeric_limits<double>::max_digits10) << step
      << "\t" << time;
    for (unsigned int b = 0; b < num_bhs; ++b)
        f << "\t" << locs[b].x() << "\t" << locs[b].y() << "\t" << locs[b].z();
    f << "\n";
}

}  // namespace dendro_gr

#endif  // DENDRO_GR_PUNCTURE_TRACKER_H
