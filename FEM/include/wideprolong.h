/**
 * @file wideprolong.h
 * @brief Wider-than-one-element 1D parent->child prolongation operators.
 *
 * Today's prolongation (RefElement::ip_1D_*) interpolates a child's nodes from
 * the single parent element's eleOrder+1 nodes, so it is a degree-eleOrder
 * Lagrange interpolant and carries O(h^(p+1)) error. A solver RHS then takes a
 * second derivative through those values, leaving every DOF whose stencil
 * crosses a 2:1 jump one order short of the interior scheme.
 *
 * These operators fix that by drawing the 1D stencil from an extended coarse
 * node array that runs into the neighbouring coarse elements. Measured in
 * test/src/testProlongationOrder.cpp for eleOrder=6, error at the first fine
 * node past the jump relative to the interface-free interior:
 *
 *     7 coarse nodes (today)   94000x
 *     8                         2350x
 *     9                           61x
 *    10                          2.5x     <- the default width, p+4
 *
 * Restoring the *order* only needs 8 nodes; reaching interior *amplitude*
 * needs 10, which is why the default is p+4 rather than the p+3 that an order
 * argument alone would suggest.
 *
 * The extension is one-sided in general. A coarse element abutting a fine
 * block cannot extend toward the block -- that side is finer, so no coarse
 * nodes exist there -- so callers pass how many extra coarse nodes are
 * actually reachable on each side and the window slides to whatever is
 * available. With no extension on either side the result is bit-comparable to
 * the narrow operator, which is what keeps the flag-off path honest.
 */

#ifndef DENDRO_WIDEPROLONG_H
#define DENDRO_WIDEPROLONG_H

#include <cstddef>
#include <vector>

namespace dendro {
namespace wideprolong {

/**
 * Default 1D stencil width in coarse nodes. p+4 is the measured point at which
 * the interface error drops to the interior level; see the table above.
 */
inline unsigned int stencil_width(unsigned int eleOrder) {
    return eleOrder + 4;
}

/**
 * Build the 1D wide prolongation matrix for one child half.
 *
 * @param[in]  eleOrder element order p; a child has p+1 nodes.
 * @param[in]  child    0 for the lower half, 1 for the upper half.
 * @param[in]  ext_lo   extra coarse nodes reachable below the element.
 * @param[in]  ext_hi   extra coarse nodes reachable above the element.
 * @param[in]  width    requested stencil width, clamped to what exists.
 * @param[out] op       row-major (p+1) x n_in; op[i*n_in + j] weights
 *                      extended input node j into child node i.
 * @param[out] n_in     p+1 + ext_lo + ext_hi.
 *
 * Extended input node j sits at parent-local coordinate (j - ext_lo)/p, where
 * the parent element spans [0,1]. Child c node i sits at c/2 + i/(2p).
 */
void build_1d(unsigned int eleOrder, unsigned int child, unsigned int ext_lo,
              unsigned int ext_hi, unsigned int width,
              std::vector<double> &op, unsigned int &n_in);

/**
 * General form: build the 1D matrix from explicit node coordinates.
 *
 * @param[in] xs  extended node coordinates in parent-element units, where the
 *                parent spans [0,1] and its own nodes sit at j/eleOrder.
 *                Must be strictly increasing.
 *
 * build_1d is the special case of a uniformly spaced extension. This form
 * exists because a neighbour at a different refinement level contributes
 * nodes at a different spacing -- 2x for a coarser neighbour -- so the
 * extended array is graded rather than uniform. Lagrange weights are
 * indifferent to spacing; only the coordinates change.
 */
void build_1d_at(unsigned int eleOrder, unsigned int child,
                 const std::vector<double> &xs, unsigned int width,
                 std::vector<double> &op);

/**
 * Apply a rectangular (n_out x n_in) operator along one axis of a 3D block.
 *
 * The existing DENDRO_TENSOR_*_APPLY_ELEM kernels take a single M and assume
 * a square operator, which a widened stencil is not, so these are separate
 * rather than a modification of those.
 *
 * A is row-major with A[o*n_in + i] weighting input i into output o. Block
 * layout is x-contiguous throughout: idx = k*ny*nx + j*nx + i.
 */
void apply_x(unsigned int n_in, unsigned int n_out, unsigned int ny,
             unsigned int nz, const double *A, const double *X, double *Y);
void apply_y(unsigned int n_in, unsigned int n_out, unsigned int nx,
             unsigned int nz, const double *A, const double *X, double *Y);
void apply_z(unsigned int n_in, unsigned int n_out, unsigned int nx,
             unsigned int ny, const double *A, const double *X, double *Y);

/** Scratch elements needed for each of apply_3d's two work buffers. */
size_t scratch_size(unsigned int eleOrder, unsigned int nx_in,
                    unsigned int ny_in, unsigned int nz_in);

/**
 * Tensor-product wide prolongation: extended coarse cube -> one child's
 * (eleOrder+1)^3 nodes. Sweeps x, then y, then z, mirroring
 * RefElement::I3D_Parent2Child so the structure is unchanged and only the
 * input extent widens.
 *
 * With every axis unextended this reproduces I3D_Parent2Child exactly, which
 * is what makes the flag-off path defensible.
 *
 * @param w1,w2 caller-supplied scratch, each at least scratch_size() long.
 */
void apply_3d(unsigned int eleOrder, const double *opx, unsigned int nx_in,
              const double *opy, unsigned int ny_in, const double *opz,
              unsigned int nz_in, const double *in, double *out, double *w1,
              double *w2);

}  // namespace wideprolong
}  // namespace dendro

#endif  // DENDRO_WIDEPROLONG_H
