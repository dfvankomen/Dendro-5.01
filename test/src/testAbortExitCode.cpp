// Oracle: test/expect_abort.cmake passes only if mpirun exits nonzero and the
// output has "async ctx not found". readFromGhostEnd on a vector that had no
// Begin is a fatal library error. Catches: MPI_Abort(comm, 0); the launcher
// may then exit 0 (implementation-defined), and the fatal error passes.
// Usage: mpirun -np 2 testAbortExitCode

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
    if (npes != 2) {
        if (!rank) std::fprintf(stderr, "usage: mpirun -np 2 %s\n", argv[0]);
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
        tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1, ot::SM_TYPE::FDM,
        DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);

    const bool onTwo = mesh->isActive() && mesh->getMPICommSize() == 2;
    if (onTwo) {
        std::vector<double> vec(mesh->getDegOfFreedom(), 0.0);
        mesh->readFromGhostEnd(vec.data(), 1);
        std::printf("testAbortExitCode: library did NOT abort\n");
    } else if (!rank) {
        std::printf("testAbortExitCode: setup failed, mesh not on 2 ranks\n");
    }
    delete mesh;
    MPI_Finalize();
    return onTwo ? 0 : 1;
}
