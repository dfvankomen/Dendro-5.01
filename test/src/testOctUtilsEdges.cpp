// function2Octree driven past m_uiMaxDepth must stay inside the domain and
// return a valid octree.
//
// A unit spike at the origin corner splits exactly one octant per depth, so
// on 2 ranks the rank-0 serial phase refines all the way to max_ref_level =
// m_uiMaxDepth + 1 and on 1 rank the parallel phase does. Both overloads are
// checked: every fx sample lies in the domain, and the result is sorted,
// unique and covers the domain. enforceSiblingsAreNotPartitioned on the
// result must keep it so (the zero-count exchange is the common case).
//
// Usage: testOctUtilsEdges [maxDepth=4]
// Exit code 0 = pass.

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "TreeNode.h"
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

    bool outside = false;
    auto spike   = [&](double x, double y, double z) {
        if (x < 0 || y < 0 || z < 0 || x > side || y > side || z > side)
            outside = true;
        return (x == 0.0 && y == 0.0 && z == 0.0) ? 1.0 : 0.0;
    };

    auto check_tree = [&](const std::vector<ot::TreeNode>& tree) {
        __int128 vol = 0;
        bool sorted = true, levels = true;
        for (size_t i = 0; i < tree.size(); i++) {
            const unsigned int lev = tree[i].getLevel();
            if (lev < 1 || lev > m_uiMaxDepth) levels = false;
            const __int128 len = (__int128)1 << (m_uiMaxDepth - lev);
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

        // a rank with no straddling sibling group receives zero octants
        const std::vector<ot::TreeNode> before = tree;
        enforceSiblingsAreNotPartitioned(tree, MPI_COMM_WORLD);
        check_tree(tree);
        if (npes == 1)
            check(tree == before, "one rank: sibling enforcement is a no-op");
    }

    if (!rank) printf("testOctUtilsEdges: %s\n", failures ? "FAIL" : "PASS");
    MPI_Finalize();
    return failures ? 1 : 0;
}
