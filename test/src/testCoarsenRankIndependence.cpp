// Oracle: from the same octree and per-element flags, ReMesh gives the same
// global octant list at any rank count as a one-rank build (MPI_COMM_SELF), for
// coarsen-all and mixed coarsen/split, and each changes the mesh. A regression
// guard: it passes without the sibling fix too (no group spans 3 ranks here).
// Usage: mpirun -np <n> testCoarsenRankIndependence [grain=8]

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"

using Oct = std::array<unsigned int, 4>;  // x, y, z, level

static unsigned int flagFor(const ot::TreeNode& e, bool mixed) {
    if (!mixed) return OCT_COARSE;
    const unsigned int half = 1u << (m_uiMaxDepth - 1);
    const unsigned int len  = 1u << (m_uiMaxDepth - e.getLevel());
    const unsigned int cx = e.minX() + len / 2, cy = e.minY() + len / 2;
    if (cx < half) return OCT_COARSE;
    if (cy < half / 2) return OCT_SPLIT;
    return OCT_NO_CHANGE;
}

static std::vector<Oct> localOcts(const ot::Mesh* m) {
    std::vector<Oct> v;
    if (!m->isActive()) return v;
    const auto& e = m->getAllElements();
    for (unsigned int i = m->getElementLocalBegin();
         i < m->getElementLocalEnd(); i++)
        v.push_back({e[i].minX(), e[i].minY(), e[i].minZ(), e[i].getLevel()});
    return v;
}

// octants in after but not before, by region: [0] x < 1/2 (coarsen region),
// [1] x >= 1/2 and y < 1/4 (split region in the mixed case)
static std::array<std::size_t, 2> newOcts(const std::vector<Oct>& before,
                                          const std::vector<Oct>& after) {
    std::vector<Oct> gained;
    std::set_difference(after.begin(), after.end(), before.begin(),
                        before.end(), std::back_inserter(gained));
    const unsigned int half = 1u << (m_uiMaxDepth - 1);
    std::array<std::size_t, 2> n{0, 0};
    for (const Oct& o : gained) {
        const unsigned int len = 1u << (m_uiMaxDepth - o[3]);
        const unsigned int cx = o[0] + len / 2, cy = o[1] + len / 2;
        if (cx < half)
            n[0]++;
        else if (cy < half / 2)
            n[1]++;
    }
    return n;
}

// sorted global octant list on rank 0 of comm (empty elsewhere)
static std::vector<Oct> gatherOcts(const std::vector<Oct>& loc, MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    int n = 4 * (int)loc.size();
    std::vector<int> cnt(npes), off(npes, 0);
    MPI_Gather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, comm);
    for (int p = 1; p < npes; p++) off[p] = off[p - 1] + cnt[p - 1];
    std::vector<Oct> all(rank ? 0 : (off[npes - 1] + cnt[npes - 1]) / 4);
    MPI_Gatherv(loc.data(), n, MPI_UNSIGNED, all.data(), cnt.data(), off.data(),
                MPI_UNSIGNED, 0, comm);
    std::sort(all.begin(), all.end());
    return all;
}

// builds the octree on comm, remeshes once with the flags; returns the
// gathered initial and remeshed octant lists and the dropped-coarsen count
static void run(MPI_Comm comm, unsigned int grain, bool mixed,
                std::vector<Oct>& before, std::vector<Oct>& after,
                DendroIntL& dropped) {
    std::function<double(double, double, double)> fr = [](double x, double y,
                                                          double z) {
        const double s = 1.0 / (1u << m_uiMaxDepth);
        auto g         = [&](double cx) {
            const double dx = x * s - cx, dy = y * s - 0.5, dz = z * s - 0.5;
            return std::exp(-60.0 * (dx * dx + dy * dy + dz * dz));
        };
        return g(0.3) + g(0.7);
    };
    std::vector<ot::TreeNode> tmp;
    function2Octree(fr, tmp, m_uiMaxDepth, 1e-3, 4, comm);
    ot::Mesh* mesh =
        ot::createMesh(tmp.data(), tmp.size(), 4, comm, 1, ot::SM_TYPE::FDM,
                       grain, 0.1, DENDRO_DEFAULT_SF_K);
    before         = gatherOcts(localOcts(mesh), comm);
    DendroIntL loc = 0;
    if (mesh->isActive()) {
        const auto& e = mesh->getAllElements();
        std::vector<unsigned int> flags;
        for (unsigned int i = mesh->getElementLocalBegin();
             i < mesh->getElementLocalEnd(); i++)
            flags.push_back(flagFor(e[i], mixed));
        mesh->setMeshRefinementFlags(flags);
        loc = mesh->getNumCoarsenDroppedAtRankBoundary();
    }
    MPI_Allreduce(&loc, &dropped, 1, MPI_LONG_LONG, MPI_SUM, comm);
    ot::Mesh* next = mesh->ReMesh(grain, 0.1, DENDRO_DEFAULT_SF_K);
    after          = gatherOcts(localOcts(next), comm);
    if (next != mesh) delete next;
    delete mesh;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    m_uiMaxDepth             = 6;
    const unsigned int grain = (argc > 1) ? std::atoi(argv[1]) : 8;
    _InitializeHcurve(m_uiDim);

    int failures = 0;
    for (bool mixed : {false, true}) {
        std::vector<Oct> b, a, b1, a1;
        DendroIntL dropped = 0, dropped1 = 0;
        run(MPI_COMM_WORLD, grain, mixed, b, a, dropped);
        if (!rank) {
            run(MPI_COMM_SELF, grain, mixed, b1, a1, dropped1);
            const auto g     = newOcts(b, a);
            const bool moved = mixed ? (g[0] > 0 && g[1] > 0) : g[0] > 0;
            const bool ok    = b == b1 && a == a1 && dropped == 0 && moved;
            std::printf(
                "  np %d %s: before %zu (1 rank %zu), after %zu (1 rank %zu), "
                "new octants %zu coarsen region / %zu split region, %lld "
                "coarsen request(s) dropped at a rank boundary: %s\n",
                npes, mixed ? "mixed" : "coarsen-all", b.size(), b1.size(),
                a.size(), a1.size(), g[0], g[1], (long long)dropped,
                ok ? "ok" : "MISMATCH");
            if (!ok) failures++;
        }
    }
    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank)
        std::printf("testCoarsenRankIndependence np %d grain %u: %s\n", npes,
                    grain, failures ? "FAIL" : "PASS");
    MPI_Finalize();
    return failures ? 1 : 0;
}
