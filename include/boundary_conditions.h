// boundary_conditions.h  --  outer-boundary conditions on unzipped blocks
//
// Solvers apply these per block after the rhs, for blocks with bflag != 0.
// Coordinates are physical: pmin/pmax are the block's padded extent, as in
// ot::BlockGeometry::ptmin/ptmax, so node (i, j, k) sits at pmin + (i, j, k) h.
#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "dendro.h"

namespace dendro_bc {

// Calls f(pp, x, y, z, faces) once for every node lying on a physical face of
// the block (edges and corners once); `faces` has bit OCT_DIR_* set for each
// face the node lies on. Faces only: the nodes behind them keep the solver's
// own stencils and closures.
template <typename F>
inline void for_each_face_node_on(const double* pmin, const double* pmax,
                                  const unsigned int* sz, unsigned int bflag,
                                  unsigned int pw, F&& f) {
    const unsigned int nx = sz[0], ny = sz[1], nz = sz[2];
    const double hx = (pmax[0] - pmin[0]) / (nx - 1);
    const double hy = (pmax[1] - pmin[1]) / (ny - 1);
    const double hz = (pmax[2] - pmin[2]) / (nz - 1);
    auto on = [bflag](unsigned int dir) { return (bflag & (1u << dir)) != 0; };

    for (unsigned int k = pw; k < nz - pw; k++) {
        unsigned int zf = 0;
        if (on(OCT_DIR_BACK) && k == pw) zf |= 1u << OCT_DIR_BACK;
        if (on(OCT_DIR_FRONT) && k == nz - pw - 1) zf |= 1u << OCT_DIR_FRONT;
        for (unsigned int j = pw; j < ny - pw; j++) {
            unsigned int yf = zf;
            if (on(OCT_DIR_DOWN) && j == pw) yf |= 1u << OCT_DIR_DOWN;
            if (on(OCT_DIR_UP) && j == ny - pw - 1) yf |= 1u << OCT_DIR_UP;
            for (unsigned int i = pw; i < nx - pw; i++) {
                unsigned int faces = yf;
                if (on(OCT_DIR_LEFT) && i == pw) faces |= 1u << OCT_DIR_LEFT;
                if (on(OCT_DIR_RIGHT) && i == nx - pw - 1)
                    faces |= 1u << OCT_DIR_RIGHT;
                if (!faces) continue;
                f(i + nx * (j + ny * k), pmin[0] + i * hx, pmin[1] + j * hy,
                  pmin[2] + k * hz, faces);
            }
        }
    }
}

// As above without the face mask.
template <typename F>
inline void for_each_face_node(const double* pmin, const double* pmax,
                               const unsigned int* sz, unsigned int bflag,
                               unsigned int pw, F&& f) {
    for_each_face_node_on(pmin, pmax, sz, bflag, pw,
                          [&](unsigned int pp, double x, double y, double z,
                              unsigned int) { f(pp, x, y, z); });
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


// ---------------------------------------------------------------------------
// Data-free conditions, one variable at a time. Apply after the interior rhs.

// Hold: the face keeps whatever value it has (f_t = 0).
inline void static_rhs(double* f_rhs, const double* pmin, const double* pmax,
                       const unsigned int* sz, unsigned int bflag,
                       unsigned int pw) {
    if (bflag == 0) return;
    for_each_face_node(pmin, pmax, sz, bflag, pw,
                       [&](unsigned int pp, double, double, double) {
                           f_rhs[pp] = 0.0;
                       });
}

// Outflow (first-order characteristic, incoming = 0): f_t = -v . grad f, with
// vel[face] the outgoing characteristic velocity through that face, indexed by
// OCT_DIR_LEFT..OCT_DIR_FRONT: c n for speed c, K n / sqrt(n.K.n) for
// u_tt = K:grad grad u. Exact for waves meeting the face head-on; oblique ones
// reflect partially, as with every local absorbing condition. Edge and corner
// nodes use the mean velocity of their faces.
inline void outflow_rhs(double* f_rhs, const double* dxf, const double* dyf,
                        const double* dzf, const double* pmin,
                        const double* pmax, const unsigned int* sz,
                        unsigned int bflag, unsigned int pw,
                        const double vel[6][3]) {
    if (bflag == 0) return;
    for_each_face_node_on(
        pmin, pmax, sz, bflag, pw,
        [&](unsigned int pp, double, double, double, unsigned int faces) {
            double v[3] = {0.0, 0.0, 0.0};
            unsigned int n = 0;
            for (unsigned int d = 0; d < 6; d++) {
                if (!(faces & (1u << d))) continue;
                v[0] += vel[d][0];
                v[1] += vel[d][1];
                v[2] += vel[d][2];
                n++;
            }
            f_rhs[pp] = -(v[0] * dxf[pp] + v[1] * dyf[pp] + v[2] * dzf[pp]) / n;
        });
}

// Neumann with a time-independent normal derivative (zero flux, or any fixed
// flux the initial data already has): the sixth-order one-sided derivative
// sum_k c_k f_k along the inward normal is held constant, so on the face
// f_t = -(sum_{k>=1} c_k f_t,k) / c_0, using the interior rhs of the six nodes
// behind it. Needs 7 nodes across the block interior. Edge and corner nodes
// average over their faces. All updates read the incoming rhs, so the result
// does not depend on face order.
inline void neumann_rhs(double* f_rhs, const double* pmin, const double* pmax,
                        const unsigned int* sz, unsigned int bflag,
                        unsigned int pw) {
    if (bflag == 0) return;
    static const double c[7] = {-147.0, 360.0, -450.0, 400.0,
                                -225.0, 72.0,  -10.0};
    const unsigned int nx = sz[0], ny = sz[1];
    const unsigned int n[3] = {sz[0], sz[1], sz[2]};
    for (unsigned int a = 0; a < 3; a++)
        if (n[a] < 2 * pw + 7)
            throw std::runtime_error(
                "dendro_bc::neumann_rhs: block interior narrower than 7 nodes");
    const unsigned int stride[3] = {1u, nx, nx * ny};

    std::vector<std::pair<unsigned int, double>> upd;
    for_each_face_node_on(
        pmin, pmax, sz, bflag, pw,
        [&](unsigned int pp, double, double, double, unsigned int faces) {
            double acc = 0.0;
            unsigned int m = 0;
            for (unsigned int d = 0; d < 6; d++) {
                if (!(faces & (1u << d))) continue;
                // OCT_DIR_LEFT/RIGHT = x, DOWN/UP = y, BACK/FRONT = z; the
                // low face of each pair steps inward in +axis
                const unsigned int axis = d / 2;
                const long step = (d % 2 == 0) ? (long)stride[axis]
                                               : -(long)stride[axis];
                double s = 0.0;
                for (unsigned int k = 1; k < 7; k++)
                    s += c[k] * f_rhs[(long)pp + (long)k * step];
                acc += -s / c[0];
                m++;
            }
            upd.emplace_back(pp, acc / m);
        });
    for (const auto& u : upd) f_rhs[u.first] = u.second;
}

// Sommerfeld (radiative): f_t = -(x f_x + y f_y + z f_z + falloff (f - f_inf))/r
// on a padding-width slab at each flagged face. Moved verbatim from the
// generated solvers (dendrosym, 2021), so results are unchanged.
inline void sommerfeld_rhs(double* f_rhs, const double* f, const double* dxf,
                           const double* dyf, const double* dzf,
                           const double* pmin, const double* pmax,
                           const double f_falloff, const double f_asymptotic,
                           const unsigned int* sz, const unsigned int bflag,
                           const unsigned int pw) {

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double hx = (pmax[0] - pmin[0]) / (nx - 1);
    double hy = (pmax[1] - pmin[1]) / (ny - 1);
    double hz = (pmax[2] - pmin[2]) / (nz - 1);

    const unsigned int PW = pw;

    unsigned int ib = PW;
    unsigned int jb = PW;
    unsigned int kb = PW;
    unsigned int ie = nx - PW;
    unsigned int je = ny - PW;
    unsigned int ke = nz - PW;

    double x, y, z;
    unsigned int pp;
    double inv_r;

    // apply on each boundary face that's flagged
    if (bflag & (1u << OCT_DIR_LEFT)) {
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < je; j++) {
                y = pmin[1] + j * hy;
                for (unsigned int i = ib; i < ib + PW; i++) {
                    x = pmin[0] + i * hx;
                    pp = i + nx * (j + ny * k);
                    inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                    f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp]
                                + f_falloff * (f[pp] - f_asymptotic));
                }
            }
        }
    }

    if (bflag & (1u << OCT_DIR_RIGHT)) {
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < je; j++) {
                y = pmin[1] + j * hy;
                for (unsigned int i = ie - PW; i < ie; i++) {
                    x = pmin[0] + i * hx;
                    pp = i + nx * (j + ny * k);
                    inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                    f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp]
                                + f_falloff * (f[pp] - f_asymptotic));
                }
            }
        }
    }

    if (bflag & (1u << OCT_DIR_DOWN)) {
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < jb + PW; j++) {
                y = pmin[1] + j * hy;
                for (unsigned int i = ib; i < ie; i++) {
                    x = pmin[0] + i * hx;
                    pp = i + nx * (j + ny * k);
                    inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                    f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp]
                                + f_falloff * (f[pp] - f_asymptotic));
                }
            }
        }
    }

    if (bflag & (1u << OCT_DIR_UP)) {
        for (unsigned int k = kb; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = je - PW; j < je; j++) {
                y = pmin[1] + j * hy;
                for (unsigned int i = ib; i < ie; i++) {
                    x = pmin[0] + i * hx;
                    pp = i + nx * (j + ny * k);
                    inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                    f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp]
                                + f_falloff * (f[pp] - f_asymptotic));
                }
            }
        }
    }

    if (bflag & (1u << OCT_DIR_BACK)) {
        for (unsigned int k = kb; k < kb + PW; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < je; j++) {
                y = pmin[1] + j * hy;
                for (unsigned int i = ib; i < ie; i++) {
                    x = pmin[0] + i * hx;
                    pp = i + nx * (j + ny * k);
                    inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                    f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp]
                                + f_falloff * (f[pp] - f_asymptotic));
                }
            }
        }
    }

    if (bflag & (1u << OCT_DIR_FRONT)) {
        for (unsigned int k = ke - PW; k < ke; k++) {
            z = pmin[2] + k * hz;
            for (unsigned int j = jb; j < je; j++) {
                y = pmin[1] + j * hy;
                for (unsigned int i = ib; i < ie; i++) {
                    x = pmin[0] + i * hx;
                    pp = i + nx * (j + ny * k);
                    inv_r = 1.0 / sqrt(x * x + y * y + z * z);
                    f_rhs[pp] = -inv_r * (x * dxf[pp] + y * dyf[pp] + z * dzf[pp]
                                + f_falloff * (f[pp] - f_asymptotic));
                }
            }
        }
    }
}

}  // namespace dendro_bc
