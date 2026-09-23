/**
 * @file boundary_conditions.h
 * @brief Outer boundary conditions for Dendro solvers.
 *
 * Rhs form (dirichlet_rhs, static_rhs, outflow_rhs, robin_rhs, neumann_rhs,
 * sommerfeld_rhs): applied to one unzipped block after its interior rhs, on
 * blocks with bflag != 0. pmin/pmax are the block's padded physical extent (as in
 * ot::BlockGeometry), so node (i, j, k) sits at pmin + (i, j, k) h; pw is the
 * padding width. Face-node kinds act only on nodes lying on a physical face; the
 * nodes behind keep the solver's own stencils and closures.
 *
 * Value form (face_nodes, set_face_values): overwrites a zipped state on the
 * domain's face nodes, for boundary data without a closed-form time derivative.
 * Solvers call it on every stage state before its rhs, at the stage time, and on
 * the final state of each step.
 */
#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "dendro.h"
#include "mesh.h"

namespace dendro_bc {

/** Coordinate axis (0, 1, 2) of face OCT_DIR_LEFT..OCT_DIR_FRONT. */
inline unsigned int face_axis(unsigned int face) { return face / 2; }

/** Index step from a node on `face` to its neighbour one node inward. */
inline long inward_step(unsigned int face, const unsigned int stride[3]) {
    const long s = (long)stride[face_axis(face)];
    return (face % 2 == 0) ? s : -s;
}

/**
 * Calls f(pp, x, y, z, faces) once for every node lying on a physical face of
 * the block, edges and corners once; `faces` has bit OCT_DIR_* set for each face
 * the node lies on.
 */
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

/** for_each_face_node_on without the face mask: f(pp, x, y, z). */
template <typename F>
inline void for_each_face_node(const double* pmin, const double* pmax,
                               const unsigned int* sz, unsigned int bflag,
                               unsigned int pw, F&& f) {
    for_each_face_node_on(pmin, pmax, sz, bflag, pw,
                          [&](unsigned int pp, double x, double y, double z,
                              unsigned int) { f(pp, x, y, z); });
}

/**
 * Dirichlet in time: on every face node, rhs[v][pp] = dfdt(x, y, z, out)[v] for
 * v < nvars. The time stepper then carries the face along the prescribed data to
 * its own accuracy, provided dfdt uses the stage time.
 */
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

/** Hold: every face node keeps its value (f_t = 0). */
inline void static_rhs(double* f_rhs, const double* pmin, const double* pmax,
                       const unsigned int* sz, unsigned int bflag,
                       unsigned int pw) {
    if (bflag == 0) return;
    for_each_face_node(pmin, pmax, sz, bflag, pw,
                       [&](unsigned int pp, double, double, double) {
                           f_rhs[pp] = 0.0;
                       });
}

/**
 * Outflow, the first-order characteristic condition with zero incoming data:
 * f_t = -v . grad f, where vel[face] is the outgoing characteristic velocity
 * through that face (c n for speed c; K n / sqrt(n.K.n) for u_tt = K:grad grad u).
 * Exact for waves meeting the face head-on; oblique ones reflect partially, as
 * with any local absorbing condition. Edge and corner nodes use the mean velocity
 * of their faces.
 */
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

/**
 * Robin, a f + b d_n f held at its initial value (d_n the outward normal
 * derivative), with a[face], b[face] per face. The sixth-order one-sided
 * derivative over the face node and the six behind it turns the condition into
 * f_t,0 = b sum_{k>=1} c_k f_t,k / (60 h a - b c_0), using the interior rhs of
 * those nodes; b = 0 holds f, a = 0 is Neumann. Needs 7 nodes across the block
 * interior. Edge and corner nodes average over their faces; every update reads
 * the incoming rhs, so the result does not depend on face order.
 */
inline void robin_rhs(double* f_rhs, const double* pmin, const double* pmax,
                      const unsigned int* sz, unsigned int bflag,
                      unsigned int pw, const double a[6], const double b[6]) {
    if (bflag == 0) return;
    static const double c[7] = {-147.0, 360.0, -450.0, 400.0,
                                -225.0, 72.0,  -10.0};
    const unsigned int stride[3] = {1u, sz[0], sz[0] * sz[1]};
    double h[3];
    for (unsigned int ax = 0; ax < 3; ax++) {
        if (sz[ax] < 2 * pw + 7)
            throw std::runtime_error(
                "dendro_bc::robin_rhs: block interior narrower than 7 nodes");
        h[ax] = (pmax[ax] - pmin[ax]) / (sz[ax] - 1);
    }

    std::vector<std::pair<unsigned int, double>> upd;
    for_each_face_node_on(
        pmin, pmax, sz, bflag, pw,
        [&](unsigned int pp, double, double, double, unsigned int faces) {
            double acc = 0.0;
            unsigned int m = 0;
            for (unsigned int d = 0; d < 6; d++) {
                if (!(faces & (1u << d))) continue;
                const long step = inward_step(d, stride);
                double s = 0.0;
                for (unsigned int k = 1; k < 7; k++)
                    s += c[k] * f_rhs[(long)pp + (long)k * step];
                acc += b[d] * s / (60.0 * h[face_axis(d)] * a[d] - b[d] * c[0]);
                m++;
            }
            upd.emplace_back(pp, acc / m);
        });
    for (const auto& u : upd) f_rhs[u.first] = u.second;
}

/** Neumann: d_n f held at its initial value (zero flux if it starts at zero). */
inline void neumann_rhs(double* f_rhs, const double* pmin, const double* pmax,
                        const unsigned int* sz, unsigned int bflag,
                        unsigned int pw) {
    static const double a[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    static const double b[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    robin_rhs(f_rhs, pmin, pmax, sz, bflag, pw, a, b);
}

/**
 * Sommerfeld (radiative): f_t = -(x f_x + y f_y + z f_z + falloff (f - f_inf)) / r
 * on a padding-width slab at each flagged face, for fields falling off as
 * f_inf + O(1/r^falloff) from a central source. The generated solvers' original
 * condition (dendrosym, 2021), unchanged.
 */
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

/** A domain face node of a zipped vector, from face_nodes(). */
struct FaceNode {
    unsigned int node;   /**< CG index into a zipped vector */
    double ox, oy, oz;   /**< octree coordinates; the solver maps them to physical */
    unsigned int faces;  /**< bit OCT_DIR_* per domain face the node lies on */
};

/**
 * Every locally owned CG node on a physical face of the domain. Face membership
 * comes from integer octant data, not coordinates. Recompute after a remesh.
 */
inline std::vector<FaceNode> face_nodes(ot::Mesh* mesh) {
    std::vector<FaceNode> out;
    if (!mesh->isActive()) return out;
    const ot::TreeNode* pNodes = &(*(mesh->getAllElements().begin()));
    const unsigned int eo      = mesh->getElementOrder();
    const unsigned int* e2n_cg = &(*(mesh->getE2NMapping().begin()));
    const unsigned int* e2n_dg = &(*(mesh->getE2NMapping_DG().begin()));
    const unsigned int nPe     = mesh->getNumNodesPerElement();
    const unsigned int nb      = mesh->getNodeLocalBegin();
    const unsigned int ne      = mesh->getNodeLocalEnd();
    const unsigned int top     = 1u << m_uiMaxDepth;
    std::vector<char> seen(mesh->getDegOfFreedom(), 0);

    for (unsigned int e = mesh->getElementLocalBegin();
         e < mesh->getElementLocalEnd(); e++)
        for (unsigned int n = 0; n < nPe; n++) {
            const unsigned int cg = e2n_cg[e * nPe + n];
            if (cg < nb || cg >= ne || seen[cg]) continue;
            seen[cg] = 1;
            unsigned int owner, ii, jj, kk;
            mesh->dg2eijk(e2n_dg[e * nPe + n], owner, ii, jj, kk);
            const ot::TreeNode& o  = pNodes[owner];
            const unsigned int len = 1u << (m_uiMaxDepth - o.getLevel());
            unsigned int faces     = 0;
            if (o.minX() == 0 && ii == 0) faces |= 1u << OCT_DIR_LEFT;
            if (o.minX() + len == top && ii == eo) faces |= 1u << OCT_DIR_RIGHT;
            if (o.minY() == 0 && jj == 0) faces |= 1u << OCT_DIR_DOWN;
            if (o.minY() + len == top && jj == eo) faces |= 1u << OCT_DIR_UP;
            if (o.minZ() == 0 && kk == 0) faces |= 1u << OCT_DIR_BACK;
            if (o.minZ() + len == top && kk == eo) faces |= 1u << OCT_DIR_FRONT;
            if (!faces) continue;
            const double h = (double)len / eo;
            out.push_back({cg, o.minX() + ii * h, o.minY() + jj * h,
                           o.minZ() + kk * h, faces});
        }
    return out;
}

/** f[node] = g(ox, oy, oz) on every face node (one variable of a zipped state). */
template <typename G>
inline void set_face_values(double* f, const std::vector<FaceNode>& nodes,
                            G&& g) {
    for (const auto& n : nodes) f[n.node] = g(n.ox, n.oy, n.oz);
}

}  // namespace dendro_bc
