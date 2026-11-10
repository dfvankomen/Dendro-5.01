// A mesh's coarsening must not depend on the rank count.
//
// setMeshRefinementFlags honors OCT_COARSE only when all 8 siblings sit on
// one rank, so a sibling group that the Mesh constructor's repartition splits
// across a rank boundary silently stays fine. That made the Brill-Lindquist
// initial mesh 7,092 elements at 128 ranks and 7,036 at 16 (BYU 15043183).
//
//   mpirun -np <n> testSiblingPartition [level=3] [grain=8]
//
// A uniform octree at <level> (8^level leaves) goes through createMesh; then
// (1) no sibling group may straddle a rank boundary, and (2) flagging every
// element OCT_COARSE and remeshing must give 8^(level-1) elements, as on one
// rank. The partitioner's 10% load tolerance lets splitters snap to coarse
// boundaries when ranks hold many elements, so the split only shows at a few
// dozen elements per rank (BL: ~55): level 3, grain 8 fails at 10 and 12 ranks
// on unfixed code and passes at 2-8. Keep the grain >= 8 so every active rank
// can hold a whole sibling group after the coarsening.

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
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

    m_uiMaxDepth = 6;
    _InitializeHcurve(m_uiDim);
    const unsigned int level = (argc > 1) ? std::atoi(argv[1]) : 3;
    const unsigned int grain = (argc > 2) ? std::atoi(argv[2]) : 8;
    const unsigned int eOrder = 4;
    const unsigned long expect = 1ul << (3 * (level - 1));
    const unsigned int step  = 1u << (m_uiMaxDepth - level);
    const unsigned int n1d   = 1u << level;

    std::vector<ot::TreeNode> leaves;
    if (!rank)
        for (unsigned int k = 0; k < n1d; k++)
            for (unsigned int j = 0; j < n1d; j++)
                for (unsigned int i = 0; i < n1d; i++)
                    leaves.push_back(ot::TreeNode(i * step, j * step, k * step, level,
                                                  m_uiDim, m_uiMaxDepth));

    ot::Mesh* mesh = ot::createMesh(leaves.data(), leaves.size(), eOrder, comm, 1,
                                    ot::SM_TYPE::FDM, grain, 0.1,
                                    DENDRO_DEFAULT_SF_K);

    // (1) the last local element and the next rank's first must not be siblings
    int straddles = 0;
    if (mesh->isActive()) {
        MPI_Comm ac = mesh->getMPICommunicator();
        const int ar = mesh->getMPIRank(), an = mesh->getMPICommSize();
        const auto& e = mesh->getAllElements();
        const unsigned int b = mesh->getElementLocalBegin(), f = mesh->getElementLocalEnd();
        ot::TreeNode first = e[b], last = e[f - 1], next;
        MPI_Status st;
        // send my first element to the previous rank, get the next rank's
        if (an > 1)
            MPI_Sendrecv(&first, sizeof(ot::TreeNode), MPI_BYTE, ar > 0 ? ar - 1 : MPI_PROC_NULL, 0,
                         &next, sizeof(ot::TreeNode), MPI_BYTE, ar + 1 < an ? ar + 1 : MPI_PROC_NULL, 0,
                         ac, &st);
        if (ar + 1 < an && last.getLevel() == next.getLevel() && last.getLevel() > 0 &&
            last.getParent() == next.getParent()) {
            std::printf("rank %d: sibling group of level %u split with rank %d\n", ar,
                        last.getLevel(), ar + 1);
            std::fflush(stdout);
            straddles = 1;
        }
    }
    int straddles_g = 0;
    MPI_Allreduce(&straddles, &straddles_g, 1, MPI_INT, MPI_SUM, comm);

    // (2) coarsen everything once: 8^level leaves must become 8^(level-1)
    unsigned long coarse = 0;
    if (mesh->isActive()) {
        std::vector<unsigned int> flags(mesh->getNumLocalMeshElements(), OCT_COARSE);
        mesh->setMeshRefinementFlags(flags);
    }
    ot::Mesh* coarser = mesh->ReMesh(grain, 0.1, DENDRO_DEFAULT_SF_K);
    if (coarser->isActive()) coarse = coarser->getNumLocalMeshElements();
    unsigned long coarse_g = 0;
    MPI_Allreduce(&coarse, &coarse_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    const bool ok = straddles_g == 0 && coarse_g == expect;
    if (!rank) {
        std::printf("testSiblingPartition np %d level %u grain %u: %d split sibling group(s), %lu "
                    "elements after one full coarsening (expect 0 and %lu): %s\n",
                    npes, level, grain, straddles_g, coarse_g, expect, ok ? "PASS" : "FAIL");
        std::fflush(stdout);
    }
    if (coarser != mesh) delete coarser;
    delete mesh;
    MPI_Finalize();
    return ok ? 0 : 1;
}
