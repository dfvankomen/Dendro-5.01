/**
 * @file testOctreeStructure.cpp
 * @brief Structural invariants of the octree that function2Octree produces.
 *
 * Everything downstream assumes the element list is a partition of the domain:
 * the unzip index arithmetic, the e2e/e2n maps and the 2:1 padding logic all
 * break in different ways if it is not. Nothing checked it.
 *
 *   - total volume of the local elements equals the domain volume (completeness)
 *   - no element is an ancestor of another, and no two overlap (disjointness)
 *   - the list is sorted, so the SFC ordering the search relies on holds
 *   - every level is within [1, maxDepth]
 *
 * Volume is accumulated in __int128 because a depth-6 domain is already 2^54.
 *
 * Usage: testOctreeStructure [maxDepth] [waveletTol]
 */

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"
#include "octUtils.h"

namespace {

int check(const char* name, bool ok, int rank, const char* detail = "") {
    if (!rank)
        std::printf("    %-26s %s %s\n", name, ok ? "ok" : "FAIL", detail);
    return ok ? 0 : 1;
}

bool overlaps(const ot::TreeNode& a, const ot::TreeNode& b) {
    return a.maxX() > b.minX() && b.maxX() > a.minX() && a.maxY() > b.minY() &&
           b.maxY() > a.minY() && a.maxZ() > b.minZ() && b.maxZ() > a.minZ();
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    _InitializeHcurve(m_uiDim);
    m_uiMaxDepth             = (argc > 1) ? std::atoi(argv[1]) : 6;
    const double wavelet_tol = (argc > 2) ? std::atof(argv[2]) : 1e-3;

    int failures = 0;
    for (unsigned int eOrder : {4u, 6u, 8u}) {
        const double scale = 1.0 / (double)(1u << m_uiMaxDepth);
        std::function<double(double, double, double)> refine_fn =
            [scale](double x, double y, double z) {
                const double dx = x * scale - 0.5, dy = y * scale - 0.5,
                             dz = z * scale - 0.5;
                return std::exp(-120.0 * (dx * dx + dy * dy + dz * dz));
            };

        std::vector<ot::TreeNode> tree;
        function2Octree(refine_fn, tree, m_uiMaxDepth, wavelet_tol, eOrder,
                        MPI_COMM_WORLD);

        unsigned long long n = tree.size();
        MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        if (!rank)
            std::printf("  eOrder=%u octants=%llu (rank 0 holds %zu)\n", eOrder,
                        n, tree.size());

        __int128 vol = 0;
        bool levels_ok = true, sorted_ok = true, disjoint_ok = true;
        for (size_t i = 0; i < tree.size(); i++) {
            const unsigned int lev = tree[i].getLevel();
            if (lev < 1 || lev > m_uiMaxDepth) levels_ok = false;
            const __int128 len = (__int128)1 << (m_uiMaxDepth - lev);
            vol += len * len * len;
            if (i && !(tree[i - 1] < tree[i])) sorted_ok = false;
            if (i && (tree[i - 1].isAncestor(tree[i]) ||
                      tree[i].isAncestor(tree[i - 1]) ||
                      overlaps(tree[i - 1], tree[i])))
                disjoint_ok = false;
        }

        // a sorted, pairwise-disjoint list only needs neighbour comparisons, but
        // the volume sum is what proves nothing is missing
        long double vol_ld = (long double)vol;
        long double total  = 0.0L;
        MPI_Allreduce(&vol_ld, &total, 1, MPI_LONG_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        const long double side = (long double)((__int128)1 << m_uiMaxDepth);
        const long double want = side * side * side;

        char detail[128];
        std::snprintf(detail, sizeof(detail), "(volume %.6Lf of domain)",
                      total / want);
        failures += check("domain fully covered", total == want, rank, detail);
        failures += check("levels within range", levels_ok, rank);
        failures += check("SFC sorted", sorted_ok, rank);
        failures += check("pairwise disjoint", disjoint_ok, rank);
    }

    int total = failures;
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (!rank)
        std::printf("\noctree structure: %s\n", total ? "FAIL" : "PASS");
    MPI_Finalize();
    return total ? 1 : 0;
}
