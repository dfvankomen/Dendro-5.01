// Oracle: an FEM_DG mesh on 2+ active ranks builds, and each rank's DG nodal
// recv offsets are its element recv offsets times nodes per element.
// Catches: nodal offsets copied onto themselves ("DG node mismatch" abort);
// under ASan, buildF2EMap's heap overflow and the delete of unset splitters.
// Usage: mpirun -np <n >= 2> testDGScatterMap

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
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
    if (npes < 2) {
        if (!rank)
            std::fprintf(stderr, "usage: mpirun -np <n >= 2> %s\n", argv[0]);
        MPI_Abort(comm, 2);
    }

    m_uiMaxDepth              = 6;
    const unsigned int eOrder = 4;
    _InitializeHcurve(m_uiDim);

    const double d_min = -10.0, d_max = 10.0;
    std::function<double(double, double, double)> fr =
        [d_min, d_max](double x, double y, double z) {
            auto to = [d_min, d_max](double g) {
                return (g / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            };
            const double xx = to(x), yy = to(y), zz = to(z);
            return std::exp(-((xx + 2) * (xx + 2) + yy * yy + zz * zz)) +
                   std::exp(-((xx - 2) * (xx - 2) + yy * yy + zz * zz));
        };
    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(fr, tmpNodes, m_uiMaxDepth, 1e-3, eOrder, comm);
    ot::Mesh* mesh = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1, ot::SM_TYPE::FEM_DG,
        DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);

    int local_fail = 0;
    if (mesh->isActive() && mesh->getMPICommSize() > 1) {
        const unsigned int npe                = mesh->getNumNodesPerElement();
        const std::vector<unsigned int>& nOff = mesh->getNodalRecvOffsets();
        const std::vector<unsigned int>& eOff = mesh->getElementRecvOffsets();
        unsigned int bad                      = 0;
        for (unsigned int p = 0; p < mesh->getMPICommSize(); p++)
            if (nOff[p] != eOff[p] * npe) bad++;
        std::printf("  [rank %d] %u of %u recv offsets wrong, npe %u\n", rank,
                    bad, mesh->getMPICommSize(), npe);
        if (bad) local_fail = 1;
    } else if (!mesh->isActive()) {
        std::printf("  [rank %d] inactive\n", rank);
    } else {
        if (!rank)
            std::printf(
                "testDGScatterMap: mesh not split over 2+ active ranks\n");
        local_fail = 1;
    }

    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, comm);
    if (!rank)
        std::printf("testDGScatterMap: %s\n", global_fail ? "FAIL" : "PASS");
    delete mesh;
    MPI_Finalize();
    return global_fail ? 1 : 0;
}
