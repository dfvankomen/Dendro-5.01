// boundary_conditions.h  --  outer-boundary conditions on unzipped blocks
//
// Solvers apply these per block after the rhs, for blocks with bflag != 0.
// Coordinates are physical: pmin/pmax are the block's padded extent, as in
// ot::BlockGeometry::ptmin/ptmax, so node (i, j, k) sits at pmin + (i, j, k) h.
#pragma once

#include <stdexcept>

#include "dendro.h"

namespace dendro_bc {

// Calls f(pp, x, y, z) once for every node lying on a physical face of the
// block (edges and corners once). Faces only: the nodes behind them keep the
// solver's own stencils and closures.
template <typename F>
inline void for_each_face_node(const double* pmin, const double* pmax,
                               const unsigned int* sz, unsigned int bflag,
                               unsigned int pw, F&& f) {
    const unsigned int nx = sz[0], ny = sz[1], nz = sz[2];
    const double hx = (pmax[0] - pmin[0]) / (nx - 1);
    const double hy = (pmax[1] - pmin[1]) / (ny - 1);
    const double hz = (pmax[2] - pmin[2]) / (nz - 1);
    auto on = [bflag](unsigned int dir) { return (bflag & (1u << dir)) != 0; };

    for (unsigned int k = pw; k < nz - pw; k++) {
        const bool zf = (on(OCT_DIR_BACK) && k == pw) ||
                        (on(OCT_DIR_FRONT) && k == nz - pw - 1);
        for (unsigned int j = pw; j < ny - pw; j++) {
            const bool yf = (on(OCT_DIR_DOWN) && j == pw) ||
                            (on(OCT_DIR_UP) && j == ny - pw - 1);
            for (unsigned int i = pw; i < nx - pw; i++) {
                const bool xf = (on(OCT_DIR_LEFT) && i == pw) ||
                                (on(OCT_DIR_RIGHT) && i == nx - pw - 1);
                if (!(xf || yf || zf)) continue;
                f(i + nx * (j + ny * k), pmin[0] + i * hx, pmin[1] + j * hy,
                  pmin[2] + k * hz);
            }
        }
    }
}

// Dirichlet in time: on every physical-face node, rhs[v][pp] is replaced by
// the prescribed time derivative, dfdt(x, y, z, out) filling out[0..nvars).
// Integrated by the solver's own time stepper, the face then follows the
// prescribed boundary data to the stepper's accuracy (stage times included,
// provided dfdt uses the stage time).
template <typename G>
inline void dirichlet_rhs(double* const* rhs, unsigned int nvars,
                          const double* pmin, const double* pmax,
                          const unsigned int* sz, unsigned int bflag,
                          unsigned int pw, G&& dfdt) {
    if (bflag == 0) return;
    double buf[64];
    if (nvars > 64)
        throw std::runtime_error("dendro_bc::dirichlet_rhs: nvars > 64");
    for_each_face_node(pmin, pmax, sz, bflag, pw,
                       [&](unsigned int pp, double x, double y, double z) {
                           dfdt(x, y, z, buf);
                           for (unsigned int v = 0; v < nvars; v++)
                               rhs[v][pp] = buf[v];
                       });
}

}  // namespace dendro_bc
