// Oracle: E6 first derivatives of sin(2 pi x) sin(2 pi y) sin(2 pi z), taken
// per block of a real mesh (2:1 jumps; unzip padding; the block's own boundary
// flag), converge from one refinement to the next: order >= 5.0 at points 3+
// spacings from the domain boundary, >= 3.5 within 3. Catches: lost bflags,
// wrong block spacing. Usage: mpirun -np <n> testDerivOnMesh

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "derivatives.h"
#include "mesh.h"
#include "meshUtils.h"

// max error of d/dx, d/dy, d/dz over block interiors: [0] points 3+ spacings
// from the domain boundary (E6 closures cover the first 3 rows), [1] the rest.
// Per point, not per block: at level 2 every block touches the boundary.
static void blockErrors(unsigned int level, MPI_Comm comm, double err[2]) {
    const unsigned int p = 6;
    // uniform at level, one level finer where x < 1/2 (2:1 jumps at x = 1/2)
    std::vector<ot::TreeNode> leaves;
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (!rank) {
        const unsigned int n = 1u << level, h = 1u << (m_uiMaxDepth - level);
        for (unsigned int k = 0; k < n; k++)
            for (unsigned int j = 0; j < n; j++)
                for (unsigned int i = 0; i < n; i++) {
                    ot::TreeNode t(i * h, j * h, k * h, level, m_uiDim,
                                   m_uiMaxDepth);
                    if (2 * i < n)
                        t.addChildren(leaves);
                    else
                        leaves.push_back(t);
                }
    }
    ot::Mesh* mesh = ot::createMesh(leaves.data(), leaves.size(), p, comm, 1,
                                    ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ,
                                    0.1, DENDRO_DEFAULT_SF_K);
    mesh->setDomainBounds(Point(0, 0, 0), Point(1, 1, 1));
    const double tp = 2.0 * M_PI;
    std::function<void(double, double, double, double*)> f =
        [tp](double x, double y, double z, double* v) {
            v[0] = std::sin(tp * x) * std::sin(tp * y) * std::sin(tp * z);
        };
    double e[2] = {0.0, 0.0};
    if (mesh->isActive()) {
        double* u  = mesh->createCGVector<double>(f, 1);
        double* uz = mesh->createUnZippedVector<double>(1);
        mesh->readFromGhostBegin(u, 1);
        mesh->readFromGhostEnd(u, 1);
        mesh->unzip(u, uz, 1);
        dendroderivs::DendroDerivatives dd("E6", "E6", p);
        const auto& blks = mesh->getLocalBlockList();
        const auto& el   = mesh->getAllElements();
        const double G   = (double)(1u << m_uiMaxDepth);
        std::vector<double> d(1);
        for (const auto& b : blks) {
            const unsigned int sz[3] = {b.getAllocationSzX(),
                                        b.getAllocationSzY(),
                                        b.getAllocationSzZ()};
            const size_t tot         = (size_t)sz[0] * sz[1] * sz[2];
            if (d.size() < tot) d.resize(tot);
            dd.set_maximum_block_size(tot);
            const unsigned int pw    = b.get1DPadWidth();
            const unsigned int lev   = el[b.getLocalElementBegin()].getLevel();
            const double hx          = (1u << (m_uiMaxDepth - lev)) / (G * p);
            const double x0          = b.getBlockNode().minX() / G - pw * hx,
                         y0          = b.getBlockNode().minY() / G - pw * hx,
                         z0          = b.getBlockNode().minZ() / G - pw * hx;
            const unsigned int bflag = b.getBlkNodeFlag();
            const double* in         = uz + b.getOffset();
            for (int axis = 0; axis < 3; axis++) {
                if (axis == 0) dd.grad_x(d.data(), in, hx, sz, bflag);
                if (axis == 1) dd.grad_y(d.data(), in, hx, sz, bflag);
                if (axis == 2) dd.grad_z(d.data(), in, hx, sz, bflag);
                for (unsigned int k = pw; k < sz[2] - pw; k++)
                    for (unsigned int j = pw; j < sz[1] - pw; j++)
                        for (unsigned int i = pw; i < sz[0] - pw; i++) {
                            const double x = x0 + i * hx, y = y0 + j * hx,
                                         z   = z0 + k * hx;
                            const double sx  = std::sin(tp * x),
                                         sy  = std::sin(tp * y),
                                         sz_ = std::sin(tp * z);
                            const double ex =
                                axis == 0   ? tp * std::cos(tp * x) * sy * sz_
                                : axis == 1 ? tp * sx * std::cos(tp * y) * sz_
                                            : tp * sx * sy * std::cos(tp * z);
                            const double got =
                                d[i + sz[0] * (j + (size_t)sz[1] * k)];
                            const double wall =
                                std::min({x, 1 - x, y, 1 - y, z, 1 - z});
                            // grid points sit at whole spacings: > 2.5 is 3+
                            const int at = wall > 2.5 * hx ? 0 : 1;
                            e[at]        = std::max(e[at], std::fabs(got - ex));
                        }
            }
        }
        delete[] u;
        delete[] uz;
    }
    MPI_Allreduce(e, err, 2, MPI_DOUBLE, MPI_MAX, comm);
    delete mesh;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    m_uiMaxDepth = 8;
    _InitializeHcurve(m_uiDim);

    double coarse[2], fine[2];
    blockErrors(2, comm, coarse);
    blockErrors(3, comm, fine);
    const double ordIn  = std::log2(coarse[0] / fine[0]);
    const double ordBdy = std::log2(coarse[1] / fine[1]);
    // floors on log2(coarse max error / fine max error), not error tolerances.
    // deriv644_x (include/derivatives/derivs_explicit.h): order 6 inside, 4 in
    // closures. Floors: 6 - 1.0 (padding at 2:1 jumps) and 4 - 0.5
    const bool ok       = ordIn >= 5.0 && ordBdy >= 3.5;
    if (!rank)
        std::printf(
            "testDerivOnMesh np %d: interior max error %.3e -> %.3e "
            "(order %.2f, floor 5.0), boundary %.3e -> %.3e (order "
            "%.2f, floor 3.5): %s\n",
            npes, coarse[0], fine[0], ordIn, coarse[1], fine[1], ordBdy,
            ok ? "PASS" : "FAIL");
    MPI_Finalize();
    return ok ? 0 : 1;
}
