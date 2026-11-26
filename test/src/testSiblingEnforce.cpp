// Oracle: enforceSiblingsAreNotPartitioned leaves no sibling group cut and the
// octants unchanged (a group over three ranks; every boundary cutting one); on
// a cut partition with all elements coarsening, exactly 8 per cut group count
// as dropped. Catches: a single pass leaving the 3-rank group split; a dropped
// counter that misses or overcounts. Usage: mpirun -np <3..8>
// testSiblingEnforce

#include <mpi.h>

#include <cstdio>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "octUtils.h"
#include "parUtils.h"
#include "sfcSort.h"

// the uniform level-2 octree (64 leaves, 8 groups) in SFC order, on every rank
static std::vector<ot::TreeNode> level2() {
    std::vector<ot::TreeNode> in, out;
    const unsigned int step = 1u << (m_uiMaxDepth - 2);
    for (unsigned int k = 0; k < 4; k++)
        for (unsigned int j = 0; j < 4; j++)
            for (unsigned int i = 0; i < 4; i++)
                in.push_back(ot::TreeNode(i * step, j * step, k * step, 2,
                                          m_uiDim, m_uiMaxDepth));
    ot::TreeNode root(m_uiDim, m_uiMaxDepth);
    SFC::seqSort::SFC_treeSort(in.data(), in.size(), out, out, out,
                               m_uiMaxDepth, m_uiMaxDepth, root, ROOT_ROTATION,
                               1, TS_SORT_ONLY);
    return in;
}

static int runCase(const char* name, const std::vector<ot::TreeNode>& all,
                   const std::vector<unsigned int>& begin, MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    std::vector<ot::TreeNode> in(all.begin() + begin[rank],
                                 all.begin() + begin[rank + 1]);
    const bool wholeBefore = siblingGroupsAreWhole(in, comm);
    enforceSiblingsAreNotPartitioned(in, comm);
    const bool wholeAfter = siblingGroupsAreWhole(in, comm);

    DendroIntL n = in.size(), nAll = 0;
    par::Mpi_Allreduce(&n, &nAll, 1, MPI_SUM, comm);
    // the concatenation must still be all, in order
    std::vector<int> cnt(npes), off(npes, 0);
    int ln = (int)in.size() * sizeof(ot::TreeNode);
    MPI_Gather(&ln, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, comm);
    for (int p = 1; p < npes; p++) off[p] = off[p - 1] + cnt[p - 1];
    std::vector<ot::TreeNode> got(rank ? 0 : nAll);
    MPI_Gatherv(in.data(), ln, MPI_BYTE, got.data(), cnt.data(), off.data(),
                MPI_BYTE, 0, comm);
    int ok = !wholeBefore && wholeAfter && nAll == (DendroIntL)all.size();
    if (!rank) {
        for (size_t i = 0; ok && i < all.size(); i++)
            if (!(got[i] == all[i])) ok = 0;
        std::printf(
            "  %s: split before %d, whole after %d, %lld of %zu "
            "leaves in order: %s\n",
            name, !wholeBefore, wholeAfter, (long long)nAll, all.size(),
            ok ? "ok" : "FAIL");
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    return ok ? 0 : 1;
}

// a mesh built on a partition that cuts groups, through the restore
// constructor (no repartition), with every element asking to coarsen: each
// cut group's 8 requests are dropped and counted
static int dropCase(const std::vector<ot::TreeNode>& all,
                    const std::vector<unsigned int>& begin, MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    std::vector<ot::TreeNode> in(all.begin() + begin[rank],
                                 all.begin() + begin[rank + 1]);
    ot::Mesh mesh(in, 1, 2, npes, comm);
    std::vector<unsigned int> flags(mesh.getNumLocalMeshElements(), OCT_COARSE);
    mesh.setMeshRefinementFlags(flags);
    DendroIntL loc = mesh.getNumCoarsenDroppedAtRankBoundary(), got = 0;
    par::Mpi_Allreduce(&loc, &got, 1, MPI_SUM, comm);
    const DendroIntL want = 8 * (npes - 1);  // one cut group per boundary
    if (!rank)
        std::printf(
            "  cut partition, all coarsen: %lld request(s) counted as "
            "dropped (expect %lld): %s\n",
            (long long)got, (long long)want, got == want ? "ok" : "FAIL");
    return got == want ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    if (npes < 3 || npes > 8) {
        if (!rank)
            std::fprintf(stderr, "usage: mpirun -np <3..8> %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    m_uiMaxDepth = 6;
    _InitializeHcurve(m_uiDim);
    const std::vector<ot::TreeNode> all = level2();

    // 3 groups, 3 leaves per rank and the rest on the last: group 0 spans
    // ranks 0, 1, 2
    std::vector<ot::TreeNode> three(all.begin(), all.begin() + 24);
    std::vector<unsigned int> b3(npes + 1);
    for (int p = 0; p < npes; p++) b3[p] = 3 * p;
    b3[npes] = 24;
    // 64 leaves, each boundary 3 past a group boundary
    std::vector<unsigned int> b1(npes + 1);
    b1[0] = 0;
    for (int p = 1; p < npes; p++) b1[p] = 8 * ((8 * p) / npes) + 3;
    b1[npes]     = 64;

    int failures = runCase("group over three ranks", three, b3, MPI_COMM_WORLD);
    failures += runCase("every boundary cuts a group", all, b1, MPI_COMM_WORLD);
    failures += dropCase(all, b1, MPI_COMM_WORLD);
    if (!rank)
        std::printf("testSiblingEnforce np %d: %s\n", npes,
                    failures ? "FAIL" : "PASS");
    MPI_Finalize();
    return failures ? 1 : 0;
}
