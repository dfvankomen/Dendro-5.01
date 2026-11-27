// Oracle: INJECTION intergrid transfer of a tensor-degree-eleOrder polynomial
// (dof = 2) onto a remeshed grid that both split and coarsened octants equals
// the polynomial at every new local node, to roundoff. Catches: wrong child
// interpolation (split) or injection index (coarsen), dof stride, NaN output.
// Usage: mpirun -np <n> testInterGridTransfer [maxDepth=6]

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <set>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    m_uiMaxDepth = (argc > 1) ? std::atoi(argv[1]) : 6;
    _InitializeHcurve(m_uiDim);
    const unsigned int dof = 2;

    int failures           = 0;
    for (unsigned int p : {4u, 6u}) {
        // domain [0,1]^3; degree p in each of x, y, z
        std::function<void(double, double, double, double*)> poly =
            [p](double x, double y, double z, double* v) {
                for (unsigned int d = 0; d < 2; d++)
                    v[d] = (1.0 + 0.3 * (d + 1) * x + std::pow(x, p)) *
                           (1.0 - y + 0.5 * std::pow(y, p)) *
                           (2.0 - 0.7 * std::pow(z, p - 1) + std::pow(z, p));
            };
        std::function<double(double, double, double)> fr = [](double x,
                                                              double y,
                                                              double z) {
            const double s  = 1.0 / (1u << m_uiMaxDepth);
            // refined in the coarsen region (x > 1/2), so coarsening has room
            const double dx = x * s - 0.75, dy = y * s - 0.5, dz = z * s - 0.5;
            return std::exp(-80.0 * (dx * dx + dy * dy + dz * dz));
        };
        std::vector<ot::TreeNode> tmp;
        function2Octree(fr, tmp, m_uiMaxDepth, 1e-3, p, comm);
        ot::Mesh* mA =
            ot::createMesh(tmp.data(), tmp.size(), p, comm, 1, ot::SM_TYPE::FDM,
                           DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);
        mA->setDomainBounds(Point(0, 0, 0), Point(1, 1, 1));

        // by element center: coarsen where x > 1/2, else split where y < 1/2
        if (mA->isActive()) {
            const auto& e           = mA->getAllElements();
            const unsigned int len0 = 1u << m_uiMaxDepth;
            std::vector<unsigned int> flags;
            for (unsigned int i = mA->getElementLocalBegin();
                 i < mA->getElementLocalEnd(); i++) {
                const unsigned int h  = 1u << (m_uiMaxDepth - e[i].getLevel());
                const unsigned int cx = e[i].minX() + h / 2,
                                   cy = e[i].minY() + h / 2;
                flags.push_back(cx > len0 / 2   ? OCT_COARSE
                                : cy < len0 / 2 ? OCT_SPLIT
                                                : OCT_NO_CHANGE);
            }
            mA->setMeshRefinementFlags(flags);
        }
        ot::Mesh* mB =
            mA->ReMesh(DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);

        // gather both octrees on rank 0; a new octant not in the old tree is
        // coarsened (an old octant shares its anchor at a finer level) or split
        // (an old octant at a coarser level contains it): both must occur
        auto gather = [&](const ot::Mesh* m) {
            std::vector<unsigned int> loc;  // x, y, z, level per octant
            if (m->isActive()) {
                const auto& e = m->getAllElements();
                for (unsigned int i = m->getElementLocalBegin();
                     i < m->getElementLocalEnd(); i++)
                    loc.insert(loc.end(), {e[i].minX(), e[i].minY(),
                                           e[i].minZ(), e[i].getLevel()});
            }
            int n = loc.size();
            std::vector<int> cnt(npes), off(npes, 0);
            MPI_Gather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, comm);
            std::vector<unsigned int> all;
            if (!rank) {
                for (int q = 1; q < npes; q++) off[q] = off[q - 1] + cnt[q - 1];
                all.resize(off[npes - 1] + cnt[npes - 1]);
            }
            MPI_Gatherv(loc.data(), n, MPI_UNSIGNED, all.data(), cnt.data(),
                        off.data(), MPI_UNSIGNED, 0, comm);
            return all;
        };
        const std::vector<unsigned int> octA = gather(mA), octB = gather(mB);
        long long changed[2] = {0, 0};  // coarsened, split
        if (!rank) {
            std::set<std::array<unsigned int, 4>> inA;
            for (size_t i = 0; i < octA.size(); i += 4)
                inA.insert({octA[i], octA[i + 1], octA[i + 2], octA[i + 3]});
            for (size_t i = 0; i < octB.size(); i += 4) {
                const unsigned int x = octB[i], y = octB[i + 1],
                                   z = octB[i + 2], l = octB[i + 3];
                if (inA.count({x, y, z, l})) continue;
                for (unsigned int f = l + 1; f <= m_uiMaxDepth; f++)
                    if (inA.count({x, y, z, f})) {
                        changed[0]++;
                        break;
                    }
                for (int c = (int)l - 1; c >= 0; c--) {
                    const unsigned int m = ~((1u << (m_uiMaxDepth - c)) - 1);
                    if (inA.count({x & m, y & m, z & m, (unsigned int)c})) {
                        changed[1]++;
                        break;
                    }
                }
            }
        }
        MPI_Bcast(changed, 2, MPI_LONG_LONG, 0, comm);

        double* in  = mA->createCGVector<double>(poly, dof);
        double* out = mB->createCGVector<double>(0.0, dof);
        if (mA->isActive()) {
            mA->readFromGhostBegin(in, dof);
            mA->readFromGhostEnd(in, dof);
        }
        mA->interGridTransfer(in, out, mB,
                              ot::INTERGRID_TRANSFER_MODE::INJECTION, dof);

        double err = 0.0, fmax = 0.0;
        if (mB->isActive()) {
            double* expect  = mB->createCGVector<double>(poly, dof);
            const size_t nN = mB->getDegOfFreedom();
            for (unsigned int d = 0; d < dof; d++)
                for (unsigned int i = mB->getNodeLocalBegin();
                     i < mB->getNodeLocalEnd(); i++) {
                    // a NaN output must fail: std::max would drop it
                    const double e =
                        std::fabs(out[d * nN + i] - expect[d * nN + i]);
                    err  = std::isnan(e) ? INFINITY : std::max(err, e);
                    fmax = std::max(fmax, std::fabs(expect[d * nN + i]));
                }
            delete[] expect;
        }
        double err_g = 0.0, fmax_g = 0.0;
        MPI_Allreduce(&err, &err_g, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&fmax, &fmax_g, 1, MPI_DOUBLE, MPI_MAX, comm);
        // relative to max|f|; degree-p polynomials are exact, so a roundoff
        // bound
        const double rel = err_g / fmax_g;
        const bool ok    = rel < 1e-12 && changed[0] > 0 && changed[1] > 0;
        if (!rank)
            std::printf(
                "  np %d eOrder %u: %lld coarsened and %lld split new "
                "octants, relative max error %.3e: %s\n",
                npes, p, changed[0], changed[1], rel, ok ? "ok" : "FAIL");
        if (!ok) failures++;
        delete[] in;
        delete[] out;
        delete mB;
        delete mA;
    }
    if (!rank)
        std::printf("testInterGridTransfer np %d: %s\n", npes,
                    failures ? "FAIL" : "PASS");
    MPI_Finalize();
    return failures ? 1 : 0;
}
