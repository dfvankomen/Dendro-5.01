// Oracle: function2Octree past m_uiMaxDepth samples only inside the domain and
// returns a sorted, unique, covering octree that sibling enforcement keeps; a
// uniform tree is 8 level-1 blocks at maxDepth 10-12. Catches: child shift -1
// at max depth, 32-bit block-volume wrap, null TreeNode& (UBSan only).
// Usage: testOctUtilsEdges [maxDepth=4]

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "TreeNode.h"
#include "block.h"
#include "dendro.h"
#include "octUtils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    _InitializeHcurve(m_uiDim);
    m_uiMaxDepth      = (argc > 1) ? std::atoi(argv[1]) : 4;
    const double side = (double)(1u << m_uiMaxDepth);

    int failures      = 0;
    auto check        = [&](bool ok, const char* what) {
        int bad = ok ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (!rank) {
            printf("    %-48s %s\n", what, bad ? "FAIL" : "ok");
            fflush(stdout);
        }
        failures += bad;
    };

    // splits one octant per depth, so refinement reaches m_uiMaxDepth + 1
    bool outside = false;
    auto spike   = [&](double x, double y, double z) {
        if (x < 0 || y < 0 || z < 0 || x > side || y > side || z > side)
            outside = true;
        return (x == 0.0 && y == 0.0 && z == 0.0) ? 1.0 : 0.0;
    };

    auto check_tree = [&](const std::vector<ot::TreeNode>& tree) {
        DendroUInt_128 vol = 0;
        bool sorted = true, levels = true;
        for (size_t i = 0; i < tree.size(); i++) {
            const unsigned int lev = tree[i].getLevel();
            if (lev < 1 || lev > m_uiMaxDepth) levels = false;
            const DendroUInt_128 len = DendroUInt_128(1)
                                       << (m_uiMaxDepth - lev);
            vol += len * len * len;
            if (i && !(tree[i - 1] < tree[i])) sorted = false;
        }
        long double v = (long double)vol, total = 0.0L;
        MPI_Allreduce(&v, &total, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        check(total == (long double)side * side * side, "domain covered once");
        check(levels, "levels within [1, maxDepth]");
        check(sorted, "sorted and unique");
    };

    for (unsigned int eOrder : {4u, 6u}) {
        if (!rank)
            printf("  npes=%d maxDepth=%u eOrder=%u\n", npes, m_uiMaxDepth,
                   eOrder);

        std::vector<ot::TreeNode> tree;
        outside = false;
        std::function<double(double, double, double)> f1 =
            [&](double x, double y, double z) { return spike(x, y, z); };
        function2Octree(f1, tree, m_uiMaxDepth + 1, 1e-3, eOrder,
                        MPI_COMM_WORLD);
        check(!outside, "scalar overload samples inside the domain");
        check_tree(tree);

        tree.clear();
        outside                    = false;
        const unsigned int vidx[1] = {0};
        std::function<void(double, double, double, double*)> f2 =
            [&](double x, double y, double z, double* v) {
                v[0] = spike(x, y, z);
            };
        function2Octree(f2, 1, vidx, 1, tree, m_uiMaxDepth + 1, 1e-3, eOrder,
                        MPI_COMM_WORLD);
        check(!outside, "vector overload samples inside the domain");
        check_tree(tree);

        const std::vector<ot::TreeNode> before = tree;
        enforceSiblingsAreNotPartitioned(tree, MPI_COMM_WORLD);
        check_tree(tree);
        if (npes == 1)
            check(tree == before, "one rank: sibling enforcement is a no-op");
    }

    // a uniform level-3 tiling is the same 8 level-1 blocks at any maxDepth;
    // a level-l block volume 2^(3*(maxDepth-l)) needs 33+ bits at maxDepth-l=11
    const unsigned int savedDepth = m_uiMaxDepth;
    for (unsigned int D : {10u, 11u, 12u}) {
        m_uiMaxDepth = D;
        std::vector<ot::TreeNode> tree;
        createRegularOctree(tree, 3, m_uiDim, D, MPI_COMM_SELF);
        std::vector<ot::Block> blocks;
        unsigned int dmin, dmax;
        octree2BlockDecomposition(tree, blocks, D, dmin, dmax, 0, tree.size(),
                                  6);
        char what[64];
        std::snprintf(what, sizeof(what),
                      "maxDepth %u: uniform tree is 8 blocks", D);
        bool ok = blocks.size() == 8;
        for (const ot::Block& b : blocks)
            ok = ok && b.getBlockNode().getLevel() == 1;
        check(ok, what);
    }
    m_uiMaxDepth = savedDepth;

    if (!rank) printf("testOctUtilsEdges: %s\n", failures ? "FAIL" : "PASS");
    MPI_Finalize();
    return failures ? 1 : 0;
}
